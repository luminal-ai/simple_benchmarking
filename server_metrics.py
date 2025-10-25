"""
Server-side metrics collection from Prometheus endpoints.

Handles fetching, parsing, and calculating server-side metrics including
throughput histograms, queue metrics, and batch sizes.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp


@dataclass
class QueueMetricsSample:
    """A single sample of queue metrics at a point in time."""
    timestamp: float  # seconds since benchmark start
    num_queue_reqs: int
    avg_queue_latency: float  # seconds
    num_running_reqs: int
    token_usage: float


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

    # Histogram sum fields (for exact average calculation)
    prefill_throughput_sum: float = 0.0
    decode_throughput_sum: float = 0.0

    # Batch size samples (we'll collect this differently)
    avg_batch_size: float = 0.0

    # Queue metrics (NEW)
    num_queue_reqs: int = 0
    avg_request_queue_latency: float = 0.0  # seconds

    # Queue histogram buckets
    queue_length_buckets: Dict[float, float] = field(default_factory=dict)
    queue_latency_buckets: Dict[float, float] = field(default_factory=dict)

    # Queue histogram sum fields (for exact average calculation)
    queue_length_sum: float = 0.0
    queue_latency_sum: float = 0.0


def parse_prometheus_metrics(text: str) -> Tuple[Dict[str, float], Dict[str, Dict[float, float]]]:
    """Parse Prometheus text format metrics into metrics dict and histogram buckets."""
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


def calculate_percentiles_from_histogram(
    buckets: Dict[float, float],         # {upper_bound: cumulative_count}
    percentiles: List[float],
    first_lower_bound: Optional[float] = None
) -> Dict[float, float]:
    """Calculate percentiles from Prometheus histogram buckets using linear interpolation."""
    if not buckets:
        return {float(p): 0.0 for p in percentiles}

    # Sort by bound and validate cumulative monotonicity
    sorted_buckets = sorted(buckets.items())  # [(upper, cum_count), ...]
    uppers, cums = zip(*sorted_buckets)

    # Basic validation / fixes
    for i in range(1, len(cums)):
        if cums[i] < cums[i-1]:
            raise ValueError("Cumulative counts must be non-decreasing.")
    total = cums[-1]
    if total <= 0:
        return {float(p): 0.0 for p in percentiles}

    # Establish the lower bound for the first bucket
    if first_lower_bound is None:
        # Best-effort: treat the first bucket as a point/degenerate interval at its upper bound
        # (or set this to 0.0 if that matches your metric's support)
        first_lower = uppers[0]
    else:
        first_lower = first_lower_bound

    results: Dict[float, float] = {}
    for p in percentiles:
        # clamp p to [0,100]
        pp = max(0.0, min(100.0, float(p)))
        if pp == 0.0:
            results[p] = float(first_lower)
            continue
        if pp == 100.0:
            results[p] = float(uppers[-1])
            continue

        target = (pp / 100.0) * total

        prev_upper = first_lower
        prev_cum = 0.0

        for upper, cum in sorted_buckets:
            if cum >= target:
                if cum == prev_cum:  # avoid div-by-zero if a bucket added nothing
                    # Put the percentile at the upper boundary (could also choose prev_upper)
                    results[p] = float(upper)
                else:
                    frac = (target - prev_cum) / (cum - prev_cum)
                    results[p] = float(prev_upper + frac * (upper - prev_upper))
                break
            prev_upper, prev_cum = upper, cum

    return results


async def fetch_server_metrics(base_url: str) -> Optional[ServerMetrics]:
    """Fetch server-side metrics from Prometheus /metrics endpoint."""
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
                    prefill_throughput_sum=parsed.get('luminal:input_throughput_histogram_sum', 0.0),
                    decode_throughput_sum=parsed.get('luminal:gen_throughput_histogram_sum', 0.0),
                    avg_batch_size=parsed.get('luminal:avg_batch_size', 0.0),
                    num_queue_reqs=int(parsed.get('luminal:num_queue_reqs', 0)),
                    avg_request_queue_latency=parsed.get('luminal:avg_request_queue_latency', 0.0),
                    queue_length_buckets=histograms.get('luminal:num_queue_reqs_histogram', {}),
                    queue_latency_buckets=histograms.get('luminal:avg_request_queue_latency_histogram', {}),
                    queue_length_sum=parsed.get('luminal:num_queue_reqs_histogram_sum', 0.0),
                    queue_latency_sum=parsed.get('luminal:avg_request_queue_latency_histogram_sum', 0.0),
                )
                return server_metrics
    except Exception as e:
        print(f"Warning: Failed to fetch server metrics: {e}")
        return None


def total_from_cumhist_delta(delta_buckets: dict[float, float]) -> float:
    """Extract total count from cumulative histogram delta buckets."""
    if not delta_buckets:
        return 0.0
    # optional: sanity check monotonicity by upper bound
    last = -float("inf")
    for _, c in sorted(delta_buckets.items()):
        if c < last:  # not cumulative → you have per-bucket counts instead
            # fall back to sum for non-cumulative histograms
            return sum(delta_buckets.values())
        last = c
    return max(delta_buckets.values())  # total count for the window


def calculate_server_metrics_delta(before: ServerMetrics, after: ServerMetrics, duration_s: float) -> Dict[str, any]:
    """Calculate delta metrics between two server metric snapshots."""
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

    # Calculate average from histogram sum/count (exact, not approximated from buckets)
    delta_prefill_sum = after.prefill_throughput_sum - before.prefill_throughput_sum
    delta_decode_sum = after.decode_throughput_sum - before.decode_throughput_sum

    print(f" before: {before.prefill_throughput_sum} after {after.prefill_throughput_sum} delta {delta_prefill_sum}")

    # Get total count from the +Inf bucket
    prefill_count = total_from_cumhist_delta(delta_prefill_buckets)
    decode_count  = total_from_cumhist_delta(delta_decode_buckets)

    print(f" count: {prefill_count} count: {decode_count} | delta {delta_decode_buckets}")

    prefill_avg = delta_prefill_sum / prefill_count if prefill_count > 0 else 0.0
    decode_avg = delta_decode_sum / decode_count if decode_count > 0 else 0.0

    avg_batch_size = after.avg_batch_size
    avg_cached_per_request = (delta_cached_tokens / delta_requests) if delta_requests > 0 else 0.0
    cache_hit_rate = (delta_cached_tokens / delta_prompt_tokens * 100) if delta_prompt_tokens > 0 else 0.0

    # Calculate queue metrics deltas
    delta_queue_length_buckets = {}
    all_queue_bounds = set(before.queue_length_buckets.keys()) | set(after.queue_length_buckets.keys())
    for bound in all_queue_bounds:
        delta_queue_length_buckets[bound] = after.queue_length_buckets.get(bound, 0) - before.queue_length_buckets.get(bound, 0)

    delta_queue_latency_buckets = {}
    all_latency_bounds = set(before.queue_latency_buckets.keys()) | set(after.queue_latency_buckets.keys())
    for bound in all_latency_bounds:
        delta_queue_latency_buckets[bound] = after.queue_latency_buckets.get(bound, 0) - before.queue_latency_buckets.get(bound, 0)

    # Calculate percentiles from queue histograms
    queue_length_percentiles = calculate_percentiles_from_histogram(delta_queue_length_buckets, percentiles, first_lower_bound=0.0)
    queue_latency_percentiles = calculate_percentiles_from_histogram(delta_queue_latency_buckets, percentiles, first_lower_bound=0.0)

    # Calculate average from histogram sum/count (exact, not approximated from buckets)
    delta_queue_length_sum = after.queue_length_sum - before.queue_length_sum
    delta_queue_latency_sum = after.queue_latency_sum - before.queue_latency_sum

    queue_length_count = total_from_cumhist_delta(delta_queue_length_buckets)
    queue_latency_count = total_from_cumhist_delta(delta_queue_latency_buckets)

    queue_length_avg = delta_queue_length_sum / queue_length_count if queue_length_count > 0 else 0.0
    queue_latency_avg_ms = (delta_queue_latency_sum / queue_latency_count * 1000) if queue_latency_count > 0 else 0.0

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
        'queue_length': {
            'p50': queue_length_percentiles.get(50, 0.0),
            'p95': queue_length_percentiles.get(95, 0.0),
            'p99': queue_length_percentiles.get(99, 0.0),
            'avg': queue_length_avg,
        },
        'queue_latency_ms': {
            'p50': queue_latency_percentiles.get(50, 0.0) * 1000,  # Convert to ms
            'p95': queue_latency_percentiles.get(95, 0.0) * 1000,
            'p99': queue_latency_percentiles.get(99, 0.0) * 1000,
            'avg': queue_latency_avg_ms,
        },
    }


async def poll_queue_metrics(base_url: str, start_time: float, samples: List[QueueMetricsSample],
                             poll_interval: float = 5.0, stop_event: asyncio.Event = None):
    """Background task that polls queue metrics periodically."""
    while not stop_event.is_set():
        try:
            async with aiohttp.ClientSession() as session:
                metrics_url = f"{base_url}/metrics"
                async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        text = await response.text()
                        parsed, _ = parse_prometheus_metrics(text)

                        sample = QueueMetricsSample(
                            timestamp=time.perf_counter() - start_time,
                            num_queue_reqs=int(parsed.get('luminal:num_queue_reqs', 0)),
                            avg_queue_latency=parsed.get('luminal:avg_request_queue_latency', 0.0),
                            num_running_reqs=int(parsed.get('luminal:num_running_reqs', 0)),
                            token_usage=parsed.get('luminal:token_usage', 0.0),
                        )
                        samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to poll queue metrics: {e}")

        # Wait for poll_interval or until stop_event is set
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
        except asyncio.TimeoutError:
            pass  # Continue polling
