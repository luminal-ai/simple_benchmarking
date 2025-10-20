"""
Client-side benchmark for vision-language models using COCO dataset.

Measures client-perceived tok/s and end-to-end latency using Poisson request distribution.
Also fetches server-side Prometheus metrics for comparison.

Now supports running multiple request rates (1.0, 10.0, 100.0) in sequence and
generating visuals to illustrate client-vs-server tradeoffs.

Usage:
python bench_client.py --url http://localhost:30000/v1/chat/completions --num-requests 100 --model moondream/moondream3-preview

or

python bench_client.py   --url http://localhost:3000/v1/chat/completions   --model moondream/moondream3-preview   --num-requests 100   --run-multi   --rates "1,10,100"   --output-dir ./out
# Optional overrides:
# --rates "1,10,100" --max-tokens 256 --output-dir ./out
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# NEW: plotting
import os
import matplotlib.pyplot as plt


@dataclass
class RequestInput:
    """Input data for a single request."""
    prompt: str
    image_url: str
    request_id: int
    prompt_length: int  # Length of text prompt in characters
    image_size_bytes: int  # Size of image in bytes


@dataclass
class RequestOutput:
    """Output and metrics for a single request."""
    request_id: int
    success: bool = False
    error: str = ""
    generated_text: str = ""
    ttft: float = 0.0  # Time to first token (seconds)
    e2e_latency: float = 0.0  # End-to-end latency (seconds)
    output_tokens: int = 0
    itl: List[float] = field(default_factory=list)  # Inter-token latencies


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics (client-side)."""
    # TTFT metrics (ms)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_avg_ms: float

    # E2E latency metrics (ms)
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    e2e_avg_ms: float

    # Tok/s metrics (per-request, not separated by prefill/decode)
    toks_p50: float
    toks_p95: float
    toks_p99: float
    toks_avg: float


@dataclass
class ServerMetrics:
    """Server-side metrics from Prometheus."""
    num_requests: int
    num_aborted_requests: int
    prompt_tokens: int
    generation_tokens: int
    cached_tokens: int

    # Histogram buckets for throughput
    prefill_throughput_buckets: Dict[float, float] = field(default_factory=dict)
    decode_throughput_buckets: Dict[float, float] = field(default_factory=dict)

    # Batch size samples (we'll collect this differently)
    avg_batch_size: float = 0.0


def parse_prometheus_metrics(text: str) -> Tuple[Dict[str, float], Dict[str, Dict[float, float]]]:
    metrics = {}
    histograms = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        bucket_match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*_bucket)\{.*?le="([\d.e+-]+|\\+Inf)".*?\}\s+([\d.e+-]+)', line)
        if bucket_match:
            metric_name = bucket_match.group(1).replace('_bucket', '')
            le_value = bucket_match.group(2)
            count = float(bucket_match.group(3))

            if le_value in ['+Inf', '\\+Inf']:
                le_value = float('inf')
            else:
                le_value = float(le_value)

            if metric_name not in histograms:
                histograms[metric_name] = {}
            histograms[metric_name][le_value] = count
            continue

        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{.*?\}\s+([\d.e+-]+)', line)
        if match:
            metric_name = match.group(1)
            value = float(match.group(2))
            metrics[metric_name] = value

    return metrics, histograms


def calculate_percentiles_from_histogram(buckets: Dict[float, float], percentiles: List[float]) -> Dict[float, float]:
    if not buckets:
        return {p: 0.0 for p in percentiles}

    sorted_buckets = sorted(buckets.items())
    total_count = sorted_buckets[-1][1]
    if total_count == 0:
        return {p: 0.0 for p in percentiles}

    results = {}
    for percentile in percentiles:
        target_count = (percentile / 100.0) * total_count
        prev_bound = 0.0
        prev_count = 0.0

        for bound, cum_count in sorted_buckets:
            if cum_count >= target_count:
                if cum_count == prev_count:
                    results[percentile] = bound
                else:
                    fraction = (target_count - prev_count) / (cum_count - prev_count)
                    results[percentile] = prev_bound + fraction * (bound - prev_bound)
                break
            prev_bound = bound
            prev_count = cum_count
        else:
            results[percentile] = sorted_buckets[-2][0] if len(sorted_buckets) > 1 else 0.0

    return results


async def fetch_server_metrics(base_url: str) -> Optional[ServerMetrics]:
    metrics_url = f"{base_url}/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Warning: Failed to fetch metrics from {metrics_url} (status {response.status})")
                    return None
                text = await response.text()
                parsed, histograms = parse_prometheus_metrics(text)
                server_metrics = ServerMetrics(
                    num_requests=int(parsed.get('luminal:num_requests_total', 0)),
                    num_aborted_requests=int(parsed.get('luminal:num_aborted_requests_total', 0)),
                    prompt_tokens=int(parsed.get('luminal:prompt_tokens_total', 0)),
                    generation_tokens=int(parsed.get('luminal:generation_tokens_total', 0)),
                    cached_tokens=int(parsed.get('luminal:cached_tokens_total', 0)),
                    prefill_throughput_buckets=histograms.get('luminal:input_throughput_histogram', {}),
                    decode_throughput_buckets=histograms.get('luminal:gen_throughput_histogram', {}),
                    avg_batch_size=parsed.get('luminal:avg_batch_size', 0.0),
                )
                return server_metrics
    except Exception as e:
        print(f"Warning: Failed to fetch server metrics: {e}")
        return None


def calculate_server_metrics_delta(before: ServerMetrics, after: ServerMetrics, duration_s: float) -> Dict[str, any]:
    delta_requests = after.num_requests - before.num_requests
    delta_aborted = after.num_aborted_requests - before.num_aborted_requests
    delta_prompt_tokens = after.prompt_tokens - before.prompt_tokens
    delta_generation_tokens = after.generation_tokens - before.generation_tokens
    delta_cached_tokens = after.cached_tokens - before.cached_tokens

    delta_prefill_buckets = {}
    all_prefill_bounds = set(before.prefill_throughput_buckets.keys()) | set(after.prefill_throughput_buckets.keys())
    for bound in all_prefill_bounds:
        delta_prefill_buckets[bound] = after.prefill_throughput_buckets.get(bound, 0) - before.prefill_throughput_buckets.get(bound, 0)

    delta_decode_buckets = {}
    all_decode_bounds = set(before.decode_throughput_buckets.keys()) | set(after.decode_throughput_buckets.keys())
    for bound in all_decode_bounds:
        delta_decode_buckets[bound] = after.decode_throughput_buckets.get(bound, 0) - before.decode_throughput_buckets.get(bound, 0)

    percentiles = [50, 95, 99]
    prefill_percentiles = calculate_percentiles_from_histogram(delta_prefill_buckets, percentiles)
    decode_percentiles = calculate_percentiles_from_histogram(delta_decode_buckets, percentiles)

    def calc_avg_from_histogram(buckets):
        if not buckets:
            return 0.0
        sorted_buckets = sorted(buckets.items())
        total_count = sorted_buckets[-1][1]
        if total_count == 0:
            return 0.0
        weighted_sum = 0.0
        prev_bound = 0.0
        prev_count = 0.0
        for bound, cum_count in sorted_buckets[:-1]:  # skip +Inf
            bucket_count = cum_count - prev_count
            if bucket_count > 0:
                midpoint = (prev_bound + bound) / 2
                weighted_sum += midpoint * bucket_count
            prev_bound = bound
            prev_count = cum_count
        return weighted_sum / total_count if total_count > 0 else 0.0

    prefill_avg = calc_avg_from_histogram(delta_prefill_buckets)
    decode_avg = calc_avg_from_histogram(delta_decode_buckets)

    avg_batch_size = after.avg_batch_size
    avg_cached_per_request = (delta_cached_tokens / delta_requests) if delta_requests > 0 else 0.0
    cache_hit_rate = (delta_cached_tokens / delta_prompt_tokens * 100) if delta_prompt_tokens > 0 else 0.0

    return {
        'requests': delta_requests,
        'aborted_requests': delta_aborted,
        'successful_requests': delta_requests - delta_aborted,
        'prompt_tokens': delta_prompt_tokens,
        'generation_tokens': delta_generation_tokens,
        'cached_tokens': delta_cached_tokens,
        'prefill_throughput': {
            'p50': prefill_percentiles.get(50, 0.0),
            'p95': prefill_percentiles.get(95, 0.0),
            'p99': prefill_percentiles.get(99, 0.0),
            'avg': prefill_avg,
        },
        'decode_throughput': {
            'p50': decode_percentiles.get(50, 0.0),
            'p95': decode_percentiles.get(95, 0.0),
            'p99': decode_percentiles.get(99, 0.0),
            'avg': decode_avg,
        },
        'cached_tokens_per_request': {'p50': None, 'p95': None, 'p99': None, 'avg': avg_cached_per_request},
        'cache_hit_rate_percent': cache_hit_rate,
        'avg_batch_size': avg_batch_size,
    }


def load_coco_dataset(num_samples: int, random_sample: bool = True) -> List[RequestInput]:
    try:
        from datasets import load_dataset
        import random
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading COCO-Caption2017 dataset from HuggingFace...")
    dataset = load_dataset("lmms-lab/COCO-Caption2017", split="val")
    print(f"Loaded {len(dataset)} images from COCO dataset")

    if len(dataset) > num_samples:
        if random_sample:
            indices = random.sample(range(len(dataset)), num_samples)
            sampled_dataset = dataset.select(indices)
        else:
            sampled_dataset = dataset.select(range(num_samples))
    else:
        print(f"Dataset has fewer than {num_samples} samples, using all {len(dataset)}")
        sampled_dataset = dataset

    requests = []
    for idx, example in enumerate(sampled_dataset):
        image = example.get("image")
        if image is None:
            continue
        import io
        import base64
        if image.mode in ["RGBA", "P", "LA"]:
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{img_str}"
        prompt = "Describe this image in detail."
        requests.append(RequestInput(
            prompt=prompt,
            image_url=image_data,
            request_id=idx,
            prompt_length=len(prompt),
            image_size_bytes=len(img_bytes)
        ))
    print(f"Created {len(requests)} requests from COCO dataset")
    return requests


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    request_input: RequestInput,
    model: str,
    max_tokens: int,
    pbar: Optional[tqdm] = None
) -> RequestOutput:
    output = RequestOutput(request_id=request_input.request_id)
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": request_input.image_url}},
            {"type": "text", "text": request_input.prompt}
        ]}
    ]
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}

    start_time = time.perf_counter()
    most_recent_timestamp = start_time
    generated_text = ""

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                output.error = f"HTTP {response.status}: {response.reason}"
                output.success = False
                if pbar: pbar.update(1)
                return output

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk_str = chunk_bytes.decode("utf-8")
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]
                if chunk_str == "[DONE]":
                    break
                try:
                    data = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    timestamp = time.perf_counter()
                    if output.ttft == 0.0:
                        output.ttft = timestamp - start_time
                    else:
                        output.itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
                    generated_text += content
                    output.output_tokens += 1

            output.e2e_latency = time.perf_counter() - start_time
            output.generated_text = generated_text
            output.success = True

    except Exception as e:
        output.error = str(e)
        output.success = False

    if pbar:
        pbar.update(1)
    return output


async def generate_requests(requests: List[RequestInput], request_rate: float):
    for request in requests:
        yield request
        if request_rate == float('inf'):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def calculate_metrics(outputs: List[RequestOutput], duration_s: float) -> BenchmarkMetrics:
    successful = [o for o in outputs if o.success]
    if not successful:
        raise ValueError("No successful requests to calculate metrics from")

    ttfts = [o.ttft for o in successful if o.ttft > 0]
    e2e_latencies = [o.e2e_latency for o in successful]
    toks_per_sec = [o.output_tokens / o.e2e_latency for o in successful if o.e2e_latency > 0 and o.output_tokens > 0]

    return BenchmarkMetrics(
        ttft_p50_ms=float(np.percentile(ttfts, 50) * 1000) if ttfts else 0,
        ttft_p95_ms=float(np.percentile(ttfts, 95) * 1000) if ttfts else 0,
        ttft_p99_ms=float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
        ttft_avg_ms=float(np.mean(ttfts) * 1000) if ttfts else 0,
        e2e_p50_ms=float(np.percentile(e2e_latencies, 50) * 1000),
        e2e_p95_ms=float(np.percentile(e2e_latencies, 95) * 1000),
        e2e_p99_ms=float(np.percentile(e2e_latencies, 99) * 1000),
        e2e_avg_ms=float(np.mean(e2e_latencies) * 1000),
        toks_p50=float(np.percentile(toks_per_sec, 50)) if toks_per_sec else 0,
        toks_p95=float(np.percentile(toks_per_sec, 95)) if toks_per_sec else 0,
        toks_p99=float(np.percentile(toks_per_sec, 99)) if toks_per_sec else 0,
        toks_avg=float(np.mean(toks_per_sec)) if toks_per_sec else 0,
    )


def print_metrics(client_metrics: BenchmarkMetrics, server_metrics: Optional[Dict[str, any]] = None, requests: Optional[List[RequestInput]] = None):
    print("\n" + "=" * 110)
    if server_metrics is None:
        print(" Benchmark Results (Client POV Only) ".center(110, "="))
    else:
        print(" Benchmark Results: Client POV vs Server POV ".center(110, "="))
    print("=" * 110)
    print(f"{'Metric':<30} {'P50':>15} {'P95':>15} {'P99':>15} {'Avg':>15}")
    print("-" * 110)
    print()
    print("CLIENT POV:")
    print(f"{'  TTFT (ms)':<30} {client_metrics.ttft_p50_ms:>15.2f} {client_metrics.ttft_p95_ms:>15.2f} {client_metrics.ttft_p99_ms:>15.2f} {client_metrics.ttft_avg_ms:>15.2f}")
    print(f"{'  E2E Latency (ms)':<30} {client_metrics.e2e_p50_ms:>15.2f} {client_metrics.e2e_p95_ms:>15.2f} {client_metrics.e2e_p99_ms:>15.2f} {client_metrics.e2e_avg_ms:>15.2f}")
    print(f"{'  Tok/s':<30} {client_metrics.toks_p50:>15.2f} {client_metrics.toks_p95:>15.2f} {client_metrics.toks_p99:>15.2f} {client_metrics.toks_avg:>15.2f}")

    if server_metrics is None:
        print("=" * 110)
        return

    print()
    print("SERVER POV:")
    print(f"{'  Decode Tok/s':<30} {server_metrics['decode_throughput']['p50']:>15.2f} {server_metrics['decode_throughput']['p95']:>15.2f} {server_metrics['decode_throughput']['p99']:>15.2f} {server_metrics['decode_throughput']['avg']:>15.2f}")
    print(f"{'  Prefill Tok/s':<30} {server_metrics['prefill_throughput']['p50']:>15.2f} {server_metrics['prefill_throughput']['p95']:>15.2f} {server_metrics['prefill_throughput']['p99']:>15.2f} {server_metrics['prefill_throughput']['avg']:>15.2f}")

    cached = server_metrics['cached_tokens_per_request']
    p50_str = 'N/A' if cached['p50'] is None else f"{cached['p50']:.2f}"
    p95_str = 'N/A' if cached['p95'] is None else f"{cached['p95']:.2f}"
    p99_str = 'N/A' if cached['p99'] is None else f"{cached['p99']:.2f}"
    print(f"{'  Cached Tokens/Req':<30} {p50_str:>15} {p95_str:>15} {p99_str:>15} {cached['avg']:>15.2f}")
    cache_rate = server_metrics['cache_hit_rate_percent']
    print(f"{'  Cache Hit Rate (%)':<30} {'N/A':>15} {'N/A':>15} {'N/A':>15} {cache_rate:>15.2f}")
    print(f"{'  Avg Batch Size':<30} {'N/A':>15} {'N/A':>15} {'N/A':>15} {server_metrics['avg_batch_size']:>15.2f}")
    print()
    print("-" * 110)
    print(f"Total Requests: {server_metrics['requests']:.0f} | Successful: {server_metrics['successful_requests']:.0f} | Aborted: {server_metrics['aborted_requests']:.0f}")
    print(f"Tokens - Prompt: {server_metrics['prompt_tokens']:.0f} | Generated: {server_metrics['generation_tokens']:.0f} | Cached: {server_metrics['cached_tokens']:.0f}")

    if requests:
        avg_prompt_len = np.mean([r.prompt_length for r in requests])
        avg_image_size_kb = np.mean([r.image_size_bytes / 1024 for r in requests])
        print(f"Input Stats - Avg Prompt Length: {avg_prompt_len:.0f} chars | Avg Image Size: {avg_image_size_kb:.2f} KB")
    print("=" * 110)


# ---------- NEW: helpers to run single & multi-rate, plus plotting ----------

async def run_single_benchmark(args, request_rate: float):
    """Runs one benchmark at a given request_rate and returns a result dict."""
    print(f"\n=== Running benchmark at {request_rate} req/s ===")
    base_url = args.url.rsplit('/v1/', 1)[0] if '/v1/' in args.url else args.url.rsplit('/', 1)[0]

    server_metrics_before = None
    if not args.disable_server_metrics:
        print("Fetching server metrics (before)...")
        server_metrics_before = await fetch_server_metrics(base_url)
        if server_metrics_before:
            print("✓ Server metrics captured (before)")
        else:
            print("⚠ Could not fetch server metrics - will show client POV only")

    # Load dataset
    requests = load_coco_dataset(args.num_requests, random_sample=True)
    if not requests:
        raise ValueError("No requests loaded from dataset")

    timeout = aiohttp.ClientTimeout(total=6*60*60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pbar = None if args.disable_tqdm else tqdm(total=len(requests), desc=f"Requests @{request_rate} rps")
        tasks = []
        benchmark_start = time.perf_counter()
        async for request in generate_requests(requests, request_rate):
            tasks.append(asyncio.create_task(
                send_chat_request(session=session, url=args.url, request_input=request,
                                  model=args.model, max_tokens=args.max_tokens, pbar=pbar)
            ))
        outputs = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start
        if pbar: pbar.close()

    server_metrics_after = None
    if not args.disable_server_metrics and server_metrics_before:
        print("Fetching server metrics (after)...")
        server_metrics_after = await fetch_server_metrics(base_url)
        if server_metrics_after:
            print("✓ Server metrics captured (after)")

    client_metrics = calculate_metrics(outputs, benchmark_duration)
    server_delta = None
    if server_metrics_before and server_metrics_after:
        server_delta = calculate_server_metrics_delta(server_metrics_before, server_metrics_after, benchmark_duration)

    print_metrics(client_metrics, server_delta, requests)

    # Build a compact result bundle
    bundle = {
        "request_rate": request_rate,
        "client_metrics": {
            "ttft_ms": {
                "p50": client_metrics.ttft_p50_ms,
                "p95": client_metrics.ttft_p95_ms,
                "p99": client_metrics.ttft_p99_ms,
                "avg": client_metrics.ttft_avg_ms,
            },
            "e2e_ms": {
                "p50": client_metrics.e2e_p50_ms,
                "p95": client_metrics.e2e_p95_ms,
                "p99": client_metrics.e2e_p99_ms,
                "avg": client_metrics.e2e_avg_ms,
            },
            "toks": {
                "p50": client_metrics.toks_p50,
                "p95": client_metrics.toks_p95,
                "p99": client_metrics.toks_p99,
                "avg": client_metrics.toks_avg,
            },
        },
        "server_metrics": server_delta,  # may be None
    }
    return bundle


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def plot_tradeoff(rates, e2e_avg_ms, decode_avg, outdir):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Client E2E Latency (ms)")
    ax1.plot(rates, e2e_avg_ms, marker="o", label="Client E2E (Avg)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Server Decode Tok/s (Avg)")
    ax2.plot(rates, decode_avg, marker="s", linestyle="--", label="Decode Tok/s (Avg)")
    fig.suptitle("Trade-off: User Latency vs Server Efficiency")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff.png"))
    plt.close(fig)


def plot_frontier(e2e_avg_ms, decode_avg, rates, outdir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(e2e_avg_ms, decode_avg)
    for x, y, r in zip(e2e_avg_ms, decode_avg, rates):
        ax.annotate(str(r), (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Client E2E Latency (ms)")
    ax.set_ylabel("Server Decode Tok/s (Avg)")
    fig.suptitle("Efficiency Frontier (Pareto) by Request Rate")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "frontier.png"))
    plt.close(fig)


def plot_series_vs_rate(rates, values, ylabel, title, filename, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates, values, marker="o")
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename))
    plt.close(fig)


def plot_client_latency_curves(rates, ttft_p50, ttft_p95, e2e_p50, e2e_p95, outdir):
    # TTFT
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(rates, ttft_p50, marker="o", label="TTFT P50")
    ax1.plot(rates, ttft_p95, marker="o", label="TTFT P95")
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.legend()
    fig1.suptitle("Client TTFT vs Request Rate")
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, "client_latency_ttft.png"))
    plt.close(fig1)

    # E2E
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(rates, e2e_p50, marker="o", label="E2E P50")
    ax2.plot(rates, e2e_p95, marker="o", label="E2E P95")
    ax2.set_xlabel("Request Rate (req/s)")
    ax2.set_ylabel("E2E Latency (ms)")
    ax2.legend()
    fig2.suptitle("Client E2E Latency vs Request Rate")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "client_latency_e2e.png"))
    plt.close(fig2)


# --- update run_multi_rate_benchmark to sleep between runs ---
async def run_multi_rate_benchmark(args, rates: List[float]) -> List[dict]:
    """Runs benchmarks across multiple request rates and returns list of result bundles."""
    results = []
    for idx, r in enumerate(rates):
        bundle = await run_single_benchmark(args, r)
        results.append(bundle)

        # Wait between tests unless this was the last one
        if idx < len(rates) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down for {wait_s} seconds before next rate...")
                await asyncio.sleep(wait_s)

    return results



def build_and_plot(results: List[dict], outdir: str):
    """Extracts series from results and generates plots + summary JSON."""
    _ensure_outdir(outdir)

    rates = [res["request_rate"] for res in results]

    # Client metrics
    e2e_avg_ms = [res["client_metrics"]["e2e_ms"]["avg"] for res in results]
    e2e_p50_ms = [res["client_metrics"]["e2e_ms"]["p50"] for res in results]
    e2e_p95_ms = [res["client_metrics"]["e2e_ms"]["p95"] for res in results]
    ttft_p50_ms = [res["client_metrics"]["ttft_ms"]["p50"] for res in results]
    ttft_p95_ms = [res["client_metrics"]["ttft_ms"]["p95"] for res in results]

    # Server metrics (may be missing)
    have_server = all(res.get("server_metrics") is not None for res in results)
    decode_avg = [res["server_metrics"]["decode_throughput"]["avg"] if res.get("server_metrics") else 0.0 for res in results]
    prefill_avg = [res["server_metrics"]["prefill_throughput"]["avg"] if res.get("server_metrics") else 0.0 for res in results]
    batch_size_avg = [res["server_metrics"]["avg_batch_size"] if res.get("server_metrics") else 0.0 for res in results]

    # Plots
    if have_server:
        plot_tradeoff(rates, e2e_avg_ms, decode_avg, outdir)
        plot_frontier(e2e_avg_ms, decode_avg, rates, outdir)
        plot_series_vs_rate(rates, batch_size_avg, "Avg Batch Size", "Avg Batch Size vs Request Rate", "batch_size_vs_rate.png", outdir)
        plot_series_vs_rate(rates, decode_avg, "Decode Tok/s (Avg)", "Server Decode Tok/s vs Request Rate", "decode_vs_rate.png", outdir)
    else:
        print("⚠ Server metrics unavailable for at least one rate; skipping server-side plots.")

    plot_client_latency_curves(rates, ttft_p50_ms, ttft_p95_ms, e2e_p50_ms, e2e_p95_ms, outdir)

    # Save summary JSON
    _save_json(os.path.join(outdir, "benchmark_summary.json"), {"results": results})


# --------------------------- original entrypoint w/ small changes ---------------------------

async def run_benchmark(args):
    """Back-compat single-run when user provides --request-rate explicitly."""
    print(f"Starting benchmark with {args.num_requests} requests at {args.request_rate} req/s")

    # Keep original single-run behavior for direct calls
    result = await run_single_benchmark(args, args.request_rate)

    # If output_file is specified, write single result
    if args.output_file:
        _save_json(args.output_file, {
            "url": args.url,
            "model": args.model,
            "num_requests": args.num_requests,
            "request_rate": result["request_rate"],
            "client_metrics": result["client_metrics"],
            "server_metrics": result["server_metrics"],
        })


def parse_rates(rates_str: Optional[str]) -> List[float]:
    if not rates_str:
        return [1.0, 10.0, 100.0]
    parts = [p.strip() for p in rates_str.split(",") if p.strip()]
    return [float(p) if p.lower() != "inf" else float("inf") for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Client-side benchmark for vision-language models")
    parser.add_argument("--url", type=str, required=True, help="API endpoint URL (e.g., http://localhost:30000/v1/chat/completions)")
    parser.add_argument("--model", type=str, default="default", help="Model name to use in requests")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Request rate (req/s) for single-run mode")
    parser.add_argument("--rates", type=str, help="Comma-separated request rates for multi-run (e.g., '1,10,100'). Defaults to 1,10,100 if provided without value.")
    parser.add_argument("--run-multi", action="store_true", help="Run benchmarks at multiple request rates (defaults to 1,10,100)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate per request")
    parser.add_argument("--output-file", type=str, help="Optional JSON file to save single-run results")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots and multi-run summary")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar")
    parser.add_argument("--disable-server-metrics", action="store_true", help="Disable fetching server-side Prometheus metrics")
    parser.add_argument(
        "--inter-run-wait-seconds",
        type=int,
        default=30,
        help="Seconds to wait between multi-run request rates to avoid crossover (default: 30)"
    )

    args = parser.parse_args()

    if args.run_multi:
        rates = parse_rates(args.rates) if args.rates else [1.0, 10.0, 100.0]
        results = asyncio.run(run_multi_rate_benchmark(args, rates))
        build_and_plot(results, args.output_dir)
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
