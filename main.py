"""
Main benchmark orchestration script.

Usage:
python main.py --url http://localhost:30000/v1/chat/completions --num-requests 100 --model moondream/moondream3-preview

or for multi-rate benchmarks:

python main.py --url http://localhost:3000/v1/chat/completions --model moondream/moondream3-preview --num-requests 100 --run-multi --rates "1,10,100" --output-dir ./out
"""

import argparse
import asyncio
import json
import os
import time
from typing import List, Optional

import aiohttp
from tqdm.asyncio import tqdm

from client_metrics import (
    BenchmarkMetrics,
    RequestInput,
    RequestOutput,
    calculate_metrics,
    generate_requests,
    load_coco_dataset,
    send_chat_request,
)
from server_metrics import (
    ServerMetrics,
    calculate_server_metrics_delta,
    fetch_server_metrics,
)
from metrics_printer import (
    plot_client_latency_curves,
    plot_distribution_skewness,
    plot_frontier,
    plot_series_vs_rate,
    plot_server_distribution_skewness,
    plot_tradeoff,
    print_metrics,
    print_prefill_decode_tables,
)


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
                                  model=args.model, max_tokens=args.max_tokens, api_key=args.api_key, pbar=pbar)
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

    # Generate distribution skewness plots
    outdir = args.output_dir if hasattr(args, 'output_dir') else "."
    rate_suffix = f"_rate_{request_rate}" if request_rate != float('inf') else "_rate_inf"
    plot_distribution_skewness(outputs, outdir, rate_suffix)
    if server_metrics_before and server_metrics_after:
        plot_server_distribution_skewness(server_metrics_before, server_metrics_after, server_delta, outdir, rate_suffix)

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
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, data):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


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
    """Parse comma-separated rates string into list of floats."""
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
    parser.add_argument("--api-key", type=str, help="Optional API key for authentication")
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
        print_prefill_decode_tables(results, args.num_requests)
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()

