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


NUM_QUEUE_PROBES = 10  # Number of probe requests per queue-depth measurement


async def run_single_benchmark_queue_depth(
    args, queue_depth: int, preloaded_requests: List[RequestInput]
) -> dict:
    """Measure new-user experience when queue_depth users are actively decoding.

    Starts queue_depth background requests with high max_tokens, waits for all
    of them to receive their first token (confirming they are in the decode
    phase), then sends NUM_QUEUE_PROBES probe requests and measures their
    TTFT and decode speed.  This directly answers: "if K users are being served,
    what does a new arrival experience?"
    """
    print(f"\n=== Queue-depth benchmark: depth={queue_depth} ===")

    base_url = args.url.rsplit('/v1/', 1)[0] if '/v1/' in args.url else args.url.rsplit('/', 1)[0]

    # Split request pool: backgrounds + probes
    bg_requests = preloaded_requests[:queue_depth] if queue_depth > 0 else []
    probe_requests = preloaded_requests[queue_depth:queue_depth + NUM_QUEUE_PROBES]

    if not probe_requests:
        raise ValueError(
            f"Not enough preloaded requests for depth={queue_depth}. "
            f"Need at least {queue_depth + NUM_QUEUE_PROBES}, got {len(preloaded_requests)}."
        )

    # Background requests use high max_tokens so they stay alive during probes
    bg_max_tokens = max(args.max_tokens, 4096)

    server_metrics_before = None
    if not args.disable_server_metrics:
        server_metrics_before = await fetch_server_metrics(base_url)

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        if queue_depth == 0:
            # Baseline — no background load
            pbar = None if args.disable_tqdm else tqdm(
                total=len(probe_requests), desc="Probes @depth=0"
            )
            benchmark_start = time.perf_counter()
            probe_outputs = await asyncio.gather(*[
                send_chat_request(
                    session, args.url, req, args.model,
                    args.max_tokens, args.api_key, pbar=pbar,
                )
                for req in probe_requests
            ])
            benchmark_duration = time.perf_counter() - benchmark_start
            if pbar:
                pbar.close()
        else:
            # Start backgrounds and wait for them to enter decode phase
            bg_events = [asyncio.Event() for _ in bg_requests]
            print(f"  Starting {queue_depth} background requests (max_tokens={bg_max_tokens})...")
            bg_tasks = [
                asyncio.create_task(send_chat_request(
                    session, args.url, req, args.model, bg_max_tokens,
                    args.api_key, first_token_event=evt,
                ))
                for req, evt in zip(bg_requests, bg_events)
            ]

            # Wait until every background has received its first token
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[evt.wait() for evt in bg_events]),
                    timeout=300,  # 5-min safety timeout
                )
                print(f"  All {queue_depth} backgrounds are decoding. Sending probes...")
            except asyncio.TimeoutError:
                started = sum(1 for e in bg_events if e.is_set())
                print(f"  WARNING: Only {started}/{queue_depth} backgrounds started "
                      f"within timeout. Proceeding with probes...")

            # Brief settle so vLLM scheduler stabilises
            await asyncio.sleep(1.0)

            # Fire probes
            pbar = None if args.disable_tqdm else tqdm(
                total=len(probe_requests), desc=f"Probes @depth={queue_depth}"
            )
            benchmark_start = time.perf_counter()
            probe_outputs = await asyncio.gather(*[
                send_chat_request(
                    session, args.url, req, args.model,
                    args.max_tokens, args.api_key, pbar=pbar,
                )
                for req in probe_requests
            ])
            benchmark_duration = time.perf_counter() - benchmark_start
            if pbar:
                pbar.close()

            # Clean up background tasks
            print(f"  Cancelling {queue_depth} background requests...")
            for t in bg_tasks:
                t.cancel()
            await asyncio.gather(*bg_tasks, return_exceptions=True)

    # --- Metrics (probes only) ---
    server_metrics_after = None
    if not args.disable_server_metrics and server_metrics_before:
        server_metrics_after = await fetch_server_metrics(base_url)

    client_metrics_result = calculate_metrics(probe_outputs, benchmark_duration)

    server_delta = None
    if server_metrics_before and server_metrics_after:
        server_delta = calculate_server_metrics_delta(
            server_metrics_before, server_metrics_after, benchmark_duration
        )

    print_metrics(client_metrics_result, server_delta, probe_requests)

    successful = [o for o in probe_outputs if o.success]
    num_successful = len(successful)
    avg_prompt_tokens = (
        sum(o.prompt_tokens for o in successful) / num_successful
        if num_successful else 0
    )
    avg_completion_tokens = (
        sum(o.completion_tokens for o in successful) / num_successful
        if num_successful else 0
    )

    return {
        "request_rate": queue_depth,  # downstream uses this as batch_size
        "num_total": len(probe_outputs),
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
        "server_metrics": server_delta,
    }


async def run_queue_depth_sweep(
    args, queue_depths: List[int], preloaded_requests: List[RequestInput]
) -> List[dict]:
    """Run queue-depth benchmarks across multiple depth levels."""
    results = []
    for idx, depth in enumerate(queue_depths):
        bundle = await run_single_benchmark_queue_depth(args, depth, preloaded_requests)
        results.append(bundle)

        if idx < len(queue_depths) - 1:
            wait_s = max(0, int(args.inter_run_wait_seconds))
            if wait_s > 0:
                print(f"\nCooling down for {wait_s} seconds before next depth...")
                await asyncio.sleep(wait_s)

    return results


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

    # Infer queue depth from per-request timestamps
    inferred = infer_queue_depths(list(outputs))
    if inferred:
        bundle["queue_depth_summary"] = summarize_by_queue_depth(inferred)
        bundle["per_request_timestamps"] = [
            {
                "request_id": r["request_id"],
                "send_time": r["send_time"],
                "first_token_time": r["first_token_time"],
                "ttft": r["ttft"],
                "queue_depth_at_send": r["queue_depth_at_send"],
                "decoding_at_send": r["decoding_at_send"],
            }
            for r in inferred
        ]

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


async def run_dual_mode_benchmark(args, concurrencies: List[int], queue_depths: List[int], input_lengths: List[int]):
    """Run both fixed-concurrency and probe-under-load benchmarks in one invocation.

    Phase 1 sweeps concurrency levels (peak performance) — all N requests start
    together behind a semaphore, measuring raw GPU capability.

    Phase 2 sweeps queue depths (probe-under-load) — K background requests are
    actively decoding, then probe requests arrive and we measure the new-user
    experience.  This answers "if K users are being served, what does user K+1 see?"
    """
    _ensure_outdir(args.output_dir)

    # --- Phase 1: Peak Performance (Fixed Concurrency) ---
    print(f"\n{'#'*60}")
    print("  Phase 1/2: Peak Performance (Fixed Concurrency)")
    print(f"{'#'*60}")
    all_concurrency_scenarios = await _collect_scenarios(
        run_multi_concurrency_benchmark, args, concurrencies, input_lengths, label="concurrency",
    )

    # --- Phase 2: Queue Depth (Probe-Under-Load) ---
    print(f"\n{'#'*60}")
    print("  Phase 2/2: Queue Depth (Probe-Under-Load)")
    print(f"{'#'*60}")
    all_queue_scenarios = []
    max_depth = max(queue_depths)

    for il_idx, input_len in enumerate(input_lengths):
        il_label = f"{input_len // 1024}k" if input_len >= 1024 else str(input_len)
        print(f"\n{'='*60}")
        print(f"  Scenario {il_idx+1}/{len(input_lengths)}: input_length={input_len} tokens ({il_label})")
        print(f"{'='*60}")

        # Need enough requests for largest depth + probes
        total_needed = max_depth + NUM_QUEUE_PROBES + 5
        requests = load_text_dataset_with_length(
            total_needed, input_len, random_sample=True,
        )

        results = await run_queue_depth_sweep(args, queue_depths, requests)

        all_queue_scenarios.append({
            "input_length": input_len,
            "rates": [float(d) for d in queue_depths],
            "results": results,
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
            "x_label": "Active Users (Queue Depth)",
            "x_values": queue_depths,
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
        "--queue-depths",
        type=str,
        help="Comma-separated queue depths for probe-under-load test (e.g., '0,1,2,4,8,16,32'). "
             "At each depth, K background requests are actively decoding while probe requests "
             "measure the new-arrival experience.",
    )
    parser.add_argument(
        "--run-dual",
        action="store_true",
        help="Run BOTH fixed-concurrency (peak performance) and probe-under-load (queue depth) "
             "benchmarks in one invocation. Requires --concurrencies and --input-lengths. "
             "Use --queue-depths to specify depths (defaults to --concurrencies values with 0 prepended)."
    )

    args = parser.parse_args()

    if args.run_dual:
        # Dual-mode requires concurrencies and input-lengths
        if not args.concurrencies:
            parser.error("--run-dual requires --concurrencies (e.g., '1,2,4,8,16,32')")
        if not args.input_lengths:
            parser.error("--run-dual requires --input-lengths (e.g., '512,2048,8192')")

        concurrencies = [int(x.strip()) for x in args.concurrencies.split(",")]
        input_lengths = [int(x.strip()) for x in args.input_lengths.split(",")]

        # Queue depths: explicit or derived from concurrencies
        if args.queue_depths:
            queue_depths = [int(x.strip()) for x in args.queue_depths.split(",")]
        else:
            queue_depths = sorted(set([0] + concurrencies))

        asyncio.run(run_dual_mode_benchmark(args, concurrencies, queue_depths, input_lengths))
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
