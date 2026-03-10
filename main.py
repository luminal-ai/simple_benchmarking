"""
Main benchmark orchestration script.

Usage for vision models (moondream):
python main.py --url http://localhost:30000/v1/chat/completions --num-requests 100 --model moondream/moondream3-preview --model-type vision

Usage for text models (llama):
python main.py --url http://localhost:30000/v1/chat/completions --num-requests 100 --model meta-llama/Llama-3-8B --model-type text

For multi-rate benchmarks:
python main.py --url http://localhost:3000/v1/chat/completions --model meta-llama/Llama-3-8B --model-type text --num-requests 100 --run-multi --rates "1,10,100" --output-dir ./out
"""

import argparse
import asyncio
import json
import os
import random
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
    infer_queue_depths,
    load_coco_dataset,
    load_text_dataset,
    load_text_dataset_with_length,
    send_chat_request,
    summarize_by_queue_depth,
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
from convert_results import convert


QUEUE_DEPTH_TTFT_LIMIT_S = 10.0  # Stop ramping when avg TTFT exceeds this (seconds)


async def _run_queue_depth_step(
    session: aiohttp.ClientSession,
    args,
    n: int,
    requests: List[RequestInput],
) -> dict:
    """Fire N requests simultaneously and return metrics + inferred queue depths.

    All N requests are launched at once. The ones that land at the back of
    vLLM's internal prefill queue experience higher TTFT — that's the queue
    effect we're measuring.

    Returns a dict with client_metrics, queue_inference, and avg_ttft_s.
    """
    use_requests = random.sample(requests, min(n, len(requests)))

    pbar = None if args.disable_tqdm else tqdm(
        total=n, desc=f"N={n}"
    )

    benchmark_start = time.perf_counter()
    tasks = [
        asyncio.create_task(send_chat_request(
            session, args.url, req, args.model,
            args.max_tokens, args.api_key, pbar=pbar,
        ))
        for req in use_requests
    ]
    outputs = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter() - benchmark_start

    if pbar:
        pbar.close()

    # Compute metrics
    client_metrics_result = calculate_metrics(outputs, benchmark_duration)

    # Infer queue depths from timestamps
    inferred = infer_queue_depths(list(outputs))
    queue_summary = summarize_by_queue_depth(inferred) if inferred else []

    successful = [o for o in outputs if o.success]
    num_successful = len(successful)
    avg_prompt_tokens = (
        sum(o.prompt_tokens for o in successful) / num_successful
        if num_successful else 0
    )
    avg_completion_tokens = (
        sum(o.completion_tokens for o in successful) / num_successful
        if num_successful else 0
    )

    # Avg TTFT in seconds for stop condition
    avg_ttft_s = client_metrics_result.ttft_avg_ms / 1000.0 if num_successful else 0

    return {
        "request_rate": n,  # downstream uses this as batch_size
        "num_total": len(outputs),
        "num_successful": num_successful,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "client_metrics": {
            "ttft_ms": {
                "p50": client_metrics_result.ttft_p50_ms,
                "p95": client_metrics_result.ttft_p95_ms,
                "p99": client_metrics_result.ttft_p99_ms,
                "avg": client_metrics_result.ttft_avg_ms,
            },
            "e2e_ms": {
                "p50": client_metrics_result.e2e_p50_ms,
                "p95": client_metrics_result.e2e_p95_ms,
                "p99": client_metrics_result.e2e_p99_ms,
                "avg": client_metrics_result.e2e_avg_ms,
            },
            "itl_ms": {
                "p50": client_metrics_result.itl_p50_ms,
                "p95": client_metrics_result.itl_p95_ms,
                "p99": client_metrics_result.itl_p99_ms,
                "avg": client_metrics_result.itl_avg_ms,
            },
            "toks": {
                "p50": client_metrics_result.toks_p50,
                "p95": client_metrics_result.toks_p95,
                "p99": client_metrics_result.toks_p99,
                "avg": client_metrics_result.toks_avg,
            },
        },
        "server_metrics": None,
        "queue_inference": {
            "n_requests": n,
            "inferred": inferred,
            "summary_by_depth": queue_summary,
        },
        "avg_ttft_s": avg_ttft_s,
    }


async def run_queue_depth_discovery(
    args, input_len: int, max_requests: int = 256,
) -> dict:
    """Discover the maximum queue depth before TTFT exceeds the limit.

    Three-phase approach:

    Phase A — Scout (exponential): N = 1, 2, 4, 8, 16, 32, ...
        Quickly finds an upper bound where TTFT exceeds the limit.

    Phase B — Narrow (binary search): between last_good and first_bad
        Finds the exact crossover point.

    Phase C — Confirm (repeat): runs max_safe_n multiple times
        Ensures the result is repeatable, not a fluke.  If confirmation
        fails, drops max_safe_n down by 1 and retries.

    Returns a dict:
        {
            "input_length": int,
            "max_safe_n": int,         # highest N consistently under limit
            "ttft_limit": float,
            "scout_results": [...],
            "narrow_results": [...],
            "confirm_results": [...],
            "all_results": [...],       # everything sorted by N
        }
    """
    ttft_limit = getattr(args, 'queue_ttft_limit', QUEUE_DEPTH_TTFT_LIMIT_S)
    max_n = getattr(args, 'max_queue_depth', 256)
    confirm_runs = getattr(args, 'queue_confirm_runs', 3)
    cooldown_s = min(5, max(0, int(args.inter_run_wait_seconds)))

    requests = load_text_dataset_with_length(max_requests, input_len, random_sample=True)

    scout_results = []
    narrow_results = []
    confirm_results = []

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async def _warmup(session, n):
        """Fire a small burst to warm the engine, discard results."""
        warmup_n = max(1, min(n, 4))
        print(f"  Warming up (N={warmup_n})...")
        await _run_queue_depth_step(session, args, warmup_n, requests)

    async def _step(session, n, label=""):
        tag = f" [{label}]" if label else ""
        print(f"\n=== Queue depth{tag}: N={n}, input_length={input_len} ===")
        await _warmup(session, n)
        bundle = await _run_queue_depth_step(session, args, n, requests)
        print(f"  N={n}: avg TTFT = {bundle['avg_ttft_s']:.2f}s (limit: {ttft_limit}s)")
        return bundle

    async def _cooldown():
        if cooldown_s > 0:
            print(f"  Cooling down {cooldown_s}s...")
            await asyncio.sleep(cooldown_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ── Phase A: Scout (exponential) ──────────────────────────
        print(f"\n{'─'*50}")
        print(f"  Phase A: Scout (exponential) — input_length={input_len}")
        print(f"{'─'*50}")

        last_good_n = 0   # last N where TTFT was under limit
        first_bad_n = None  # first N where TTFT exceeded limit

        n = 1
        while n <= max_n:
            bundle = await _step(session, n, "scout")
            scout_results.append(bundle)

            if bundle["avg_ttft_s"] > ttft_limit:
                # Re-test to avoid a single fluke declaring the upper bound
                retest_over = 1  # already have 1 over-limit result
                retest_under = 0
                for rt in range(confirm_runs - 1):
                    await _cooldown()
                    rt_bundle = await _step(session, n, f"scout retest {rt+2}/{confirm_runs}")
                    scout_results.append(rt_bundle)
                    if rt_bundle["avg_ttft_s"] > ttft_limit:
                        retest_over += 1
                    else:
                        retest_under += 1

                if retest_over > retest_under:
                    first_bad_n = n
                    print(f"  Scout: N={n} confirmed over limit ({retest_over}/{confirm_runs} exceeded). Upper bound found.")
                    break
                else:
                    print(f"  Scout: N={n} was a fluke ({retest_under}/{confirm_runs} passed). Continuing.")
                    last_good_n = n
            else:
                last_good_n = n

            await _cooldown()
            next_n = n * 2
            if next_n > max_n:
                next_n = max_n
            if next_n <= n:
                # Can't go higher — we've tested max_n already
                break
            n = next_n

        # ── Phase B: Narrow (binary search) ───────────────────────
        if first_bad_n is not None and last_good_n > 0 and first_bad_n - last_good_n > 1:
            print(f"\n{'─'*50}")
            print(f"  Phase B: Narrow (binary search) — between {last_good_n} and {first_bad_n}")
            print(f"{'─'*50}")

            lo, hi = last_good_n, first_bad_n
            while hi - lo > 1:
                mid = (lo + hi) // 2
                await _cooldown()
                bundle = await _step(session, mid, "narrow")
                narrow_results.append(bundle)

                if bundle["avg_ttft_s"] > ttft_limit:
                    # Re-test to avoid fluke
                    retest_over = 1
                    retest_under = 0
                    for rt in range(confirm_runs - 1):
                        await _cooldown()
                        rt_bundle = await _step(session, mid, f"narrow retest {rt+2}/{confirm_runs}")
                        narrow_results.append(rt_bundle)
                        if rt_bundle["avg_ttft_s"] > ttft_limit:
                            retest_over += 1
                        else:
                            retest_under += 1
                    if retest_over > retest_under:
                        hi = mid
                    else:
                        lo = mid
                else:
                    lo = mid

            last_good_n = lo
            first_bad_n = hi
            print(f"  Narrow complete: max_safe_n={last_good_n}, first_bad_n={first_bad_n}")

        # Handle edge cases
        if first_bad_n is None:
            # Never exceeded limit — max_n is safe
            max_safe_n = last_good_n if last_good_n > 0 else max_n
        elif last_good_n == 0:
            # N=1 already exceeded limit
            max_safe_n = 0
        else:
            max_safe_n = last_good_n

        # ── Phase C: Confirm (repeat) ─────────────────────────────
        if max_safe_n > 0:
            print(f"\n{'─'*50}")
            print(f"  Phase C: Confirm — N={max_safe_n} × {confirm_runs} runs")
            print(f"{'─'*50}")

            for run_idx in range(confirm_runs):
                await _cooldown()
                bundle = await _step(session, max_safe_n, f"confirm {run_idx+1}/{confirm_runs}")
                confirm_results.append(bundle)

                if bundle["avg_ttft_s"] > ttft_limit:
                    print(f"  Confirm run {run_idx+1} FAILED (TTFT={bundle['avg_ttft_s']:.2f}s > {ttft_limit}s)")
                    # Drop down and retry
                    max_safe_n -= 1
                    if max_safe_n <= 0:
                        max_safe_n = 0
                        print(f"  Dropped to max_safe_n=0. Stopping confirmation.")
                        break
                    print(f"  Dropping to max_safe_n={max_safe_n}, restarting confirmation.")
                    confirm_results = []  # Reset confirms for new N
                    run_idx = -1  # Will be 0 on next iteration
                    # Restart the confirm loop with the new max_safe_n
                    break
            else:
                # All confirm runs passed — we're done
                avg_confirm_ttft = sum(r["avg_ttft_s"] for r in confirm_results) / len(confirm_results)
                print(f"  Confirmed: N={max_safe_n} avg TTFT={avg_confirm_ttft:.2f}s across {confirm_runs} runs")

            # If we broke out to retry with lower N, run the confirms again
            if len(confirm_results) < confirm_runs and max_safe_n > 0:
                confirm_results = []
                for run_idx in range(confirm_runs):
                    await _cooldown()
                    bundle = await _step(session, max_safe_n, f"confirm {run_idx+1}/{confirm_runs}")
                    confirm_results.append(bundle)
                    if bundle["avg_ttft_s"] > ttft_limit:
                        max_safe_n -= 1
                        if max_safe_n <= 0:
                            max_safe_n = 0
                            confirm_results = []
                            break

    # ── Assemble results ──────────────────────────────────────────
    all_results = sorted(
        scout_results + narrow_results + confirm_results,
        key=lambda r: r["request_rate"],
    )

    il_label = f"{input_len // 1024}k" if input_len >= 1024 else str(input_len)
    print(f"\n{'='*50}")
    print(f"  Queue Depth Discovery Complete — {il_label} context")
    print(f"  Max safe queue depth: {max_safe_n}")
    print(f"  TTFT limit: {ttft_limit}s")
    print(f"  Steps tested: {len(all_results)}")
    print(f"{'='*50}")

    return {
        "input_length": input_len,
        "max_safe_n": max_safe_n,
        "ttft_limit": ttft_limit,
        "scout_results": scout_results,
        "narrow_results": narrow_results,
        "confirm_results": confirm_results,
        "all_results": all_results,
    }


async def run_single_benchmark_concurrency(args, concurrency: int, preloaded_requests: Optional[List[RequestInput]] = None):
    """Runs one benchmark at fixed concurrency (exactly N requests in flight) and returns a result dict."""
    print(f"\n=== Running benchmark at concurrency={concurrency} ===")
    base_url = args.url.rsplit('/v1/', 1)[0] if '/v1/' in args.url else args.url.rsplit('/', 1)[0]

    server_metrics_before = None
    if not args.disable_server_metrics:
        print("Fetching server metrics (before)...")
        server_metrics_before = await fetch_server_metrics(base_url)
        if server_metrics_before:
            print("✓ Server metrics captured (before)")
        else:
            print("⚠ Could not fetch server metrics - will show client POV only")

    if preloaded_requests is not None:
        requests = preloaded_requests
    else:
        model_type = getattr(args, 'model_type', 'vision')
        if model_type == 'text':
            requests = load_text_dataset(args.num_requests, random_sample=True)
        else:
            requests = load_coco_dataset(args.num_requests, random_sample=True)

    if not requests:
        raise ValueError("No requests loaded from dataset")

    timeout = aiohttp.ClientTimeout(total=6*60*60)
    semaphore = asyncio.Semaphore(concurrency)
    outputs: List[RequestOutput] = []

    async def _throttled_request(session, req):
        async with semaphore:
            return await send_chat_request(
                session=session, url=args.url, request_input=req,
                model=args.model, max_tokens=args.max_tokens,
                api_key=args.api_key, pbar=pbar,
            )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        pbar = None if args.disable_tqdm else tqdm(total=len(requests), desc=f"Requests @c={concurrency}")
        benchmark_start = time.perf_counter()
        tasks = [asyncio.create_task(_throttled_request(session, req)) for req in requests]
        outputs = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start
        if pbar:
            pbar.close()

    server_metrics_after = None
    if not args.disable_server_metrics and server_metrics_before:
        server_metrics_after = await fetch_server_metrics(base_url)

    client_metrics = calculate_metrics(outputs, benchmark_duration)
    server_delta = None
    if server_metrics_before and server_metrics_after:
        server_delta = calculate_server_metrics_delta(server_metrics_before, server_metrics_after, benchmark_duration)

    print_metrics(client_metrics, server_delta, requests)

    outdir = args.output_dir if hasattr(args, 'output_dir') else "."
    rate_suffix = f"_conc_{concurrency}"
    plot_distribution_skewness(outputs, outdir, rate_suffix)

    successful = [o for o in outputs if o.success]
    num_successful = len(successful)
    avg_prompt_tokens = (
        sum(o.prompt_tokens for o in successful) / num_successful if num_successful else 0
    )
    avg_completion_tokens = (
        sum(o.completion_tokens for o in successful) / num_successful if num_successful else 0
    )

    bundle = {
        "request_rate": concurrency,  # Use concurrency as the "rate" for report compat
        "num_total": len(outputs),
        "num_successful": num_successful,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
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
            "itl_ms": {
                "p50": client_metrics.itl_p50_ms,
                "p95": client_metrics.itl_p95_ms,
                "p99": client_metrics.itl_p99_ms,
                "avg": client_metrics.itl_avg_ms,
            },
            "toks": {
                "p50": client_metrics.toks_p50,
                "p95": client_metrics.toks_p95,
                "p99": client_metrics.toks_p99,
                "avg": client_metrics.toks_avg,
            },
        },
        "server_metrics": server_delta,
    }

    return bundle


async def run_single_benchmark(args, request_rate: float, preloaded_requests: Optional[List[RequestInput]] = None):
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

    # Use preloaded requests if provided, otherwise load dataset
    if preloaded_requests is not None:
        requests = preloaded_requests
    else:
        model_type = getattr(args, 'model_type', 'vision')
        if model_type == 'text':
            print("Loading text dataset for text-only model...")
            requests = load_text_dataset(args.num_requests, random_sample=True)
        else:
            print("Loading COCO vision dataset for vision model...")
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

    # Compute per-request token averages for downstream report generation
    successful = [o for o in outputs if o.success]
    num_successful = len(successful)
    avg_prompt_tokens = (
        sum(o.prompt_tokens for o in successful) / num_successful
        if num_successful else 0
    )
    avg_completion_tokens = (
        sum(o.completion_tokens for o in successful) / num_successful
        if num_successful else 0
    )

    # Build a compact result bundle
    bundle = {
        "request_rate": request_rate,
        "num_total": len(outputs),
        "num_successful": num_successful,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
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
            "itl_ms": {
                "p50": client_metrics.itl_p50_ms,
                "p95": client_metrics.itl_p95_ms,
                "p99": client_metrics.itl_p99_ms,
                "avg": client_metrics.itl_avg_ms,
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


async def run_multi_rate_benchmark(args, rates: List[float], preloaded_requests: Optional[List[RequestInput]] = None) -> List[dict]:
    """Runs benchmarks across multiple request rates and returns list of result bundles."""
    results = []
    for idx, r in enumerate(rates):
        bundle = await run_single_benchmark(args, r, preloaded_requests)
        results.append(bundle)

        # Wait between tests unless this was the last one
        if idx < len(rates) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down for {wait_s} seconds before next rate...")
                await asyncio.sleep(wait_s)

    return results


async def run_multi_concurrency_benchmark(args, concurrencies: List[int], preloaded_requests: Optional[List[RequestInput]] = None) -> List[dict]:
    """Runs benchmarks across multiple fixed-concurrency levels."""
    results = []
    for idx, c in enumerate(concurrencies):
        bundle = await run_single_benchmark_concurrency(args, c, preloaded_requests)
        results.append(bundle)

        if idx < len(concurrencies) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down for {wait_s} seconds before next concurrency level...")
                await asyncio.sleep(wait_s)

    return results


def build_and_plot(results: List[dict], outdir: str, args):
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

    # Save summary JSON with metadata for report generation
    summary = {
        "url": args.url,
        "model": args.model,
        "model_type": getattr(args, 'model_type', 'text'),
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "rates": rates,
        "results": results,
    }
    _save_json(os.path.join(outdir, "benchmark_summary.json"), summary)

    # Auto-generate report_data.json and HTML report
    _generate_report(summary, outdir, args)


async def _collect_scenarios(runner_fn, args, x_values, input_lengths: List[int], label: str) -> List[dict]:
    """Common loop for collecting benchmark scenarios across input lengths.

    Parameters
    ----------
    runner_fn : async callable(args, x_values, preloaded_requests) -> List[dict]
        Either ``run_multi_rate_benchmark`` or ``run_multi_concurrency_benchmark``.
    args : argparse.Namespace
        CLI arguments.
    x_values : list
        The rates or concurrency levels to sweep.
    input_lengths : List[int]
        Token lengths to iterate over.
    label : str
        Human-readable label for progress messages (e.g. "concurrency" or "rate").

    Returns
    -------
    List[dict]
        One scenario dict per input length.
    """
    all_scenarios = []

    for il_idx, input_len in enumerate(input_lengths):
        il_label = f"{input_len // 1024}k" if input_len >= 1024 else str(input_len)
        print(f"\n{'='*60}")
        print(f"  Scenario {il_idx+1}/{len(input_lengths)}: input_length={input_len} tokens ({il_label})")
        print(f"{'='*60}")

        requests = load_text_dataset_with_length(args.num_requests, input_len, random_sample=True)
        results = await runner_fn(args, x_values, preloaded_requests=requests)

        all_scenarios.append({
            "input_length": input_len,
            "rates": [float(v) for v in x_values],  # compat with report pipeline
            "results": results,
        })

        print_prefill_decode_tables(results, args.num_requests)

        if il_idx < len(input_lengths) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down {wait_s}s before next input length...")
                await asyncio.sleep(wait_s)

    return all_scenarios


async def run_multi_scenario_benchmark(args, rates: List[float], input_lengths: List[int]):
    """Run benchmarks across multiple input lengths and request rates.

    Produces a multi-scenario benchmark_summary.json and generates the HTML report
    with context-length variation on the x-axis.
    """
    _ensure_outdir(args.output_dir)
    all_scenarios = await _collect_scenarios(
        run_multi_rate_benchmark, args, rates, input_lengths, label="rate",
    )

    # Save multi-scenario summary
    summary = {
        "url": args.url,
        "model": args.model,
        "model_type": getattr(args, 'model_type', 'text'),
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "input_lengths": input_lengths,
        "rates": rates,
        "scenarios": all_scenarios,
    }
    _save_json(os.path.join(args.output_dir, "benchmark_summary.json"), summary)

    # Generate report
    _generate_report(summary, args.output_dir, args)


async def run_multi_scenario_concurrency_benchmark(args, concurrencies: List[int], input_lengths: List[int]):
    """Run benchmarks across multiple input lengths and fixed concurrency levels.

    Like run_multi_scenario_benchmark but uses fixed concurrency (semaphore)
    instead of Poisson arrival rates. This ensures exactly N requests in flight.
    """
    _ensure_outdir(args.output_dir)
    all_scenarios = await _collect_scenarios(
        run_multi_concurrency_benchmark, args, concurrencies, input_lengths, label="concurrency",
    )

    summary = {
        "url": args.url,
        "model": args.model,
        "model_type": getattr(args, 'model_type', 'text'),
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "input_lengths": input_lengths,
        "rates": [float(c) for c in concurrencies],
        "scenarios": all_scenarios,
    }
    _save_json(os.path.join(args.output_dir, "benchmark_summary.json"), summary)
    _generate_report(summary, args.output_dir, args)


async def run_queue_test(args, input_lengths: List[int]):
    """Standalone queue depth test — no concurrency sweep, just discovery.

    For each input length, runs the 3-phase discovery (scout → narrow → confirm)
    and saves a queue_summary.json with per-scenario max_safe_n values.
    """
    _ensure_outdir(args.output_dir)
    max_queue_requests = getattr(args, 'max_queue_depth', 256)

    all_scenarios = []
    for il_idx, input_len in enumerate(input_lengths):
        il_label = f"{input_len // 1024}k" if input_len >= 1024 else str(input_len)
        print(f"\n{'#'*60}")
        print(f"  Queue Test: Scenario {il_idx+1}/{len(input_lengths)}: {il_label} context")
        print(f"{'#'*60}")

        result = await run_queue_depth_discovery(args, input_len, max_requests=max_queue_requests)
        all_scenarios.append(result)

        if il_idx < len(input_lengths) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down {wait_s}s before next input length...")
                await asyncio.sleep(wait_s)

    # Save summary
    summary = {
        "url": args.url,
        "model": args.model,
        "model_type": getattr(args, 'model_type', 'text'),
        "max_tokens": args.max_tokens,
        "ttft_limit": getattr(args, 'queue_ttft_limit', QUEUE_DEPTH_TTFT_LIMIT_S),
        "scenarios": [
            {
                "input_length": s["input_length"],
                "max_safe_n": s["max_safe_n"],
                "ttft_limit": s["ttft_limit"],
                "rates": [float(r["request_rate"]) for r in s["all_results"]],
                "results": s["all_results"],
            }
            for s in all_scenarios
        ],
    }
    _save_json(os.path.join(args.output_dir, "queue_summary.json"), summary)
    _generate_report(summary, args.output_dir, args)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  QUEUE DEPTH SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Context':<12} {'Max Safe N':<14} {'TTFT Limit'}")
    print(f"  {'─'*12} {'─'*14} {'─'*10}")
    for s in all_scenarios:
        il_label = f"{s['input_length'] // 1024}k" if s['input_length'] >= 1024 else str(s['input_length'])
        safe = str(s['max_safe_n']) if s['max_safe_n'] > 0 else "N/A (even N=1 exceeds)"
        print(f"  {il_label:<12} {safe:<14} {s['ttft_limit']}s")
    print(f"{'='*60}")


async def run_dual_mode_benchmark(args, concurrencies: List[int], input_lengths: List[int]):
    """Run both fixed-concurrency and queue-depth-discovery benchmarks.

    Phase 1 sweeps concurrency levels (peak performance) — all N requests start
    together behind a semaphore, measuring raw GPU capability.

    Phase 2 discovers queue depth limits (adaptive) — for each input length,
    fires N=1, 2, 3, ... requests simultaneously, measuring TTFT at each level.
    Stops when avg TTFT exceeds the limit (default 10s).  The result is a curve
    of queue_depth → TTFT that tells operators exactly when to scale.
    """
    _ensure_outdir(args.output_dir)

    # --- Phase 1: Peak Performance (Fixed Concurrency) ---
    print(f"\n{'#'*60}")
    print("  Phase 1/2: Peak Performance (Fixed Concurrency)")
    print(f"{'#'*60}")
    all_concurrency_scenarios = await _collect_scenarios(
        run_multi_concurrency_benchmark, args, concurrencies, input_lengths, label="concurrency",
    )

    # --- Phase 2: Queue Depth Discovery (3-phase adaptive) ---
    print(f"\n{'#'*60}")
    print("  Phase 2/2: Queue Depth Discovery (Scout → Narrow → Confirm)")
    print(f"  Stop condition: avg TTFT > {getattr(args, 'queue_ttft_limit', QUEUE_DEPTH_TTFT_LIMIT_S)}s")
    print(f"{'#'*60}")
    all_queue_scenarios = []
    max_queue_requests = getattr(args, 'max_queue_depth', 256)

    for il_idx, input_len in enumerate(input_lengths):
        discovery = await run_queue_depth_discovery(args, input_len, max_requests=max_queue_requests)
        all_queue_scenarios.append({
            "input_length": input_len,
            "max_safe_n": discovery["max_safe_n"],
            "rates": [float(r["request_rate"]) for r in discovery["all_results"]],
            "results": discovery["all_results"],
        })

        if il_idx < len(input_lengths) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down {wait_s}s before next input length...")
                await asyncio.sleep(wait_s)

    # Save combined summary
    summary = {
        "url": args.url,
        "model": args.model,
        "model_type": getattr(args, 'model_type', 'text'),
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "dual_mode": True,
        "input_lengths": input_lengths,
        "peak_performance": {
            "x_label": "Concurrent Users",
            "x_values": concurrencies,
            "scenarios": all_concurrency_scenarios,
        },
        "queue_depth": {
            "x_label": "Queue Depth (Simultaneous Requests)",
            "x_values": [float(r["request_rate"]) for r in all_queue_scenarios[0]["results"]] if all_queue_scenarios else [],
            "scenarios": all_queue_scenarios,
        },
    }
    _save_json(os.path.join(args.output_dir, "benchmark_summary.json"), summary)
    _generate_report(summary, args.output_dir, args)


def _generate_report(summary: dict, outdir: str, args):
    """Convert benchmark results and generate HTML report."""
    try:
        report_data = convert(
            summary,
            provider_name=getattr(args, 'provider', None),
            gpu_name=getattr(args, 'gpu', None),
        )
        report_json_path = os.path.join(outdir, "report_data.json")
        _save_json(report_json_path, report_data)

        report_dir = os.path.join(outdir, "report")
        os.makedirs(report_dir, exist_ok=True)

        if report_data.get("dual_mode"):
            # Dual-mode report: peak performance + sustained load
            from generate_report import generate_dual_html_report
            generate_dual_html_report(report_data, report_dir)
        else:
            # Single-mode report
            from generate_report import load_report_json, build_scenario_data, generate_html_report
            from generate_report import fig_throughput_tradeoff, fig_system_throughput, fig_per_user_speed
            from generate_report import fig_scaling_efficiency, fig_latency

            df, model_name, gpu_name = load_report_json(report_json_path)
            sd = build_scenario_data(df)

            fig_throughput_tradeoff(df, model_name, report_dir)
            fig_system_throughput(df, model_name, report_dir)
            fig_per_user_speed(df, model_name, report_dir)
            fig_scaling_efficiency(df, model_name, report_dir)
            fig_latency(df, model_name, report_dir)
            generate_html_report(df, model_name, gpu_name, sd, report_dir)

        print(f"\nHTML report: {os.path.join(report_dir, 'index.html')}")
    except Exception as e:
        print(f"\nWARNING: Report generation failed: {e}")
        import traceback
        traceback.print_exc()


def merge_phase1_and_phase2(peak_path: str, queue_path: str, args):
    """Merge separately-collected Phase 1 and Phase 2 JSON files into a dual-mode report.

    Phase 1 (benchmark_summary.json) provides peak_performance scenarios.
    Phase 2 (queue_summary.json) provides queue_depth scenarios.
    """
    import json as _json

    with open(peak_path) as f:
        peak = _json.load(f)
    with open(queue_path) as f:
        queue = _json.load(f)

    # Extract peak performance scenarios
    if peak.get("dual_mode"):
        # Already dual-mode — extract just peak_performance
        peak_scenarios = peak["peak_performance"]["scenarios"]
        peak_x_label = peak["peak_performance"].get("x_label", "Concurrent Users")
        peak_x_values = peak["peak_performance"].get("x_values", [])
    elif peak.get("scenarios") and isinstance(peak["scenarios"], list):
        # Multi-scenario format
        peak_scenarios = peak["scenarios"]
        peak_x_label = "Concurrent Users"
        rates = peak.get("rates") or peak.get("concurrencies") or []
        peak_x_values = rates
    else:
        raise ValueError(f"Unrecognized format in {peak_path}")

    # Extract queue scenarios
    queue_scenarios = queue.get("scenarios", [])
    if not queue_scenarios:
        raise ValueError(f"No scenarios found in {queue_path}")

    queue_x_values = [float(r["request_rate"]) for r in queue_scenarios[0].get("results", [])] if queue_scenarios else []

    # Build merged dual-mode summary
    merged = {
        "url": peak.get("url", queue.get("url", "")),
        "model": peak.get("model", queue.get("model", "Unknown")),
        "model_type": peak.get("model_type", queue.get("model_type", "text")),
        "num_requests": peak.get("num_requests", 100),
        "max_tokens": peak.get("max_tokens", queue.get("max_tokens", 256)),
        "dual_mode": True,
        "input_lengths": sorted(set(
            [s.get("input_length", 0) for s in peak_scenarios] +
            [s.get("input_length", 0) for s in queue_scenarios]
        )),
        "peak_performance": {
            "x_label": peak_x_label,
            "x_values": peak_x_values,
            "scenarios": peak_scenarios,
        },
        "queue_depth": {
            "x_label": "Queue Depth (Simultaneous Requests)",
            "x_values": queue_x_values,
            "scenarios": queue_scenarios,
        },
    }

    _ensure_outdir(args.output_dir)
    merged_path = os.path.join(args.output_dir, "benchmark_summary.json")
    _save_json(merged_path, merged)
    print(f"Merged dual-mode summary: {merged_path}")

    _generate_report(merged, args.output_dir, args)


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
    parser = argparse.ArgumentParser(description="Client-side benchmark for language models (vision and text)")
    parser.add_argument("--url", type=str, required=True, help="API endpoint URL (e.g., http://localhost:30000/v1/chat/completions)")
    parser.add_argument("--model", type=str, default="default", help="Model name to use in requests")
    parser.add_argument("--model-type", type=str, choices=["vision", "text"], default="vision",
                        help="Model type: 'vision' for vision-language models (e.g., moondream) or 'text' for text-only models (e.g., llama)")
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
    parser.add_argument("--provider", type=str, help="Provider name for report (e.g., 'Together AI')")
    parser.add_argument("--gpu", type=str, help="GPU/hardware name for report (e.g., '8xH100')")
    parser.add_argument("--input-lengths", type=str,
                        help="Comma-separated input token lengths to sweep (e.g., '512,2048,8192,32768'). "
                             "Generates separate scenarios per context length for the HTML report.")

    parser.add_argument(
        "--inter-run-wait-seconds",
        type=int,
        default=30,
        help="Seconds to wait between multi-run request rates to avoid crossover (default: 30)"
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        help="Comma-separated fixed concurrency levels (e.g., '1,2,4,8,16,32'). "
             "Uses fixed-concurrency mode: exactly N requests in flight at a time. "
             "Mutually exclusive with --rates."
    )
    parser.add_argument(
        "--queue-ttft-limit",
        type=float,
        default=QUEUE_DEPTH_TTFT_LIMIT_S,
        help=f"Stop queue depth discovery when avg TTFT exceeds this many seconds (default: {QUEUE_DEPTH_TTFT_LIMIT_S}s).",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=256,
        help="Maximum N to test during queue depth discovery (default: 256).",
    )
    parser.add_argument(
        "--queue-confirm-runs",
        type=int,
        default=3,
        help="Number of confirmation runs at the discovered max safe queue depth (default: 3).",
    )
    parser.add_argument(
        "--run-queue-test",
        action="store_true",
        help="Run standalone queue depth discovery. Uses exponential scouting, binary search narrowing, "
             "and confirmation runs to find the max queue depth before TTFT exceeds --queue-ttft-limit. "
             "Requires --input-lengths.",
    )
    parser.add_argument(
        "--run-dual",
        action="store_true",
        help="Run BOTH fixed-concurrency (peak performance) and queue-depth discovery "
             "benchmarks in one invocation. Requires --concurrencies and --input-lengths."
    )
    parser.add_argument(
        "--merge-results",
        nargs=2,
        metavar=("PEAK_JSON", "QUEUE_JSON"),
        help="Merge separately-collected Phase 1 (benchmark_summary.json) and "
             "Phase 2 (queue_summary.json) into a dual-mode report. "
             "Example: --merge-results results/benchmark_summary.json results/queue_summary.json",
    )

    args = parser.parse_args()

    if args.merge_results:
        merge_phase1_and_phase2(args.merge_results[0], args.merge_results[1], args)
        return

    if args.run_queue_test:
        # Standalone queue depth test
        if not args.input_lengths:
            parser.error("--run-queue-test requires --input-lengths (e.g., '1024,4096,16384,32768')")
        input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]
        asyncio.run(run_queue_test(args, input_lengths))
    elif args.run_dual:
        # Dual-mode requires concurrencies and input-lengths
        if not args.concurrencies:
            parser.error("--run-dual requires --concurrencies (e.g., '1,2,4,8,16,32')")
        if not args.input_lengths:
            parser.error("--run-dual requires --input-lengths (e.g., '512,2048,8192')")

        concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]
        input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]

        asyncio.run(run_dual_mode_benchmark(args, concurrencies, input_lengths))
    elif args.run_multi:
        if args.concurrencies:
            # Fixed-concurrency mode
            concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]
            if args.input_lengths:
                input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]
                asyncio.run(run_multi_scenario_concurrency_benchmark(args, concurrencies, input_lengths))
            else:
                results = asyncio.run(run_multi_concurrency_benchmark(args, concurrencies))
                build_and_plot(results, args.output_dir, args)
                print_prefill_decode_tables(results, args.num_requests)
        else:
            # Poisson rate mode (original behavior)
            rates = parse_rates(args.rates) if args.rates else [1.0, 10.0, 100.0]
            if args.input_lengths:
                input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]
                asyncio.run(run_multi_scenario_benchmark(args, rates, input_lengths))
            else:
                results = asyncio.run(run_multi_rate_benchmark(args, rates))
                build_and_plot(results, args.output_dir, args)
                print_prefill_decode_tables(results, args.num_requests)
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
