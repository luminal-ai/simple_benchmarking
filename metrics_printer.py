"""
Printing and visualization of benchmark metrics.

Handles console output formatting and generating plots for both client and server metrics.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from client_metrics import BenchmarkMetrics, RequestInput, RequestOutput
from server_metrics import QueueMetricsSample, ServerMetrics


def print_metrics(client_metrics: BenchmarkMetrics, server_metrics: Optional[Dict[str, any]] = None, requests: Optional[List[RequestInput]] = None):
    """Print formatted benchmark metrics table."""
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
    print("QUEUE METRICS:")
    print(f"{'  Queue Length (# reqs)':<30} {server_metrics['queue_length']['p50']:>15.1f} {server_metrics['queue_length']['p95']:>15.1f} {server_metrics['queue_length']['p99']:>15.1f} {server_metrics['queue_length']['avg']:>15.1f}")
    print(f"{'  Queue Latency (ms)':<30} {server_metrics['queue_latency_ms']['p50']:>15.2f} {server_metrics['queue_latency_ms']['p95']:>15.2f} {server_metrics['queue_latency_ms']['p99']:>15.2f} {server_metrics['queue_latency_ms']['avg']:>15.2f}")
    print()
    print("-" * 110)
    print(f"Total Requests: {server_metrics['requests']:.0f} | Successful: {server_metrics['successful_requests']:.0f} | Aborted: {server_metrics['aborted_requests']:.0f}")
    print(f"Tokens - Prompt: {server_metrics['prompt_tokens']:.0f} | Generated: {server_metrics['generation_tokens']:.0f} | Cached: {server_metrics['cached_tokens']:.0f}")

    if requests:
        avg_prompt_len = np.mean([r.prompt_length for r in requests])
        avg_image_size_kb = np.mean([r.image_size_bytes / 1024 for r in requests])
        print(f"Input Stats - Avg Prompt Length: {avg_prompt_len:.0f} chars | Avg Image Size: {avg_image_size_kb:.2f} KB")
    print("=" * 110)


def print_queue_metrics_summary(samples: List[QueueMetricsSample]):
    """Print summary statistics for queue metrics."""
    if not samples:
        print("No queue metrics samples collected")
        return

    queue_lengths = [s.num_queue_reqs for s in samples]
    queue_latencies = [s.avg_queue_latency * 1000 for s in samples if s.avg_queue_latency > 0]

    print("\n" + "=" * 80)
    print(" Queue Metrics Summary (Time-Series Data) ".center(80, "="))
    print("=" * 80)
    print(f"Samples collected: {len(samples)}")
    print()
    print("QUEUE LENGTH:")
    print(f"  Mean: {np.mean(queue_lengths):.1f} requests")
    print(f"  P50:  {np.percentile(queue_lengths, 50):.1f} requests")
    print(f"  P95:  {np.percentile(queue_lengths, 95):.1f} requests")
    print(f"  P99:  {np.percentile(queue_lengths, 99):.1f} requests")
    print(f"  Max:  {np.max(queue_lengths):.0f} requests")
    print()
    if queue_latencies:
        print("QUEUE LATENCY:")
        print(f"  Mean: {np.mean(queue_latencies):.2f} ms")
        print(f"  P50:  {np.percentile(queue_latencies, 50):.2f} ms")
        print(f"  P95:  {np.percentile(queue_latencies, 95):.2f} ms")
        print(f"  P99:  {np.percentile(queue_latencies, 99):.2f} ms")
        print(f"  Max:  {np.max(queue_latencies):.2f} ms")
    print("=" * 80)


def print_prefill_decode_tables(results: List[dict], num_requests: int):
    """Print formatted tables for prefill and decode metrics across multiple rates."""
    # Build rows for both tables
    prefill_rows = []
    decode_rows = []

    def _safe_get(d, path, default=0.0):
        cur = d
        try:
            for p in path:
                if cur is None:
                    return default
                cur = cur[p]
            return cur if cur is not None else default
        except Exception:
            return default

    def _fmt_int(x):
        try:
            return f"{float(x):.0f}"
        except Exception:
            return "0"

    def _fmt_sec(x_ms):
        try:
            return f"{float(x_ms)/1000.0:.1f}"
        except Exception:
            return "0.0"

    def _fmt_tokps(x):
        try:
            return f"{float(x):.0f}"
        except Exception:
            return "0"

    for res in results:
        r = res.get("request_rate", 0.0)
        cm = res.get("client_metrics", {})
        sm = res.get("server_metrics", None)

        # Client metrics
        e2e_p50_ms = _safe_get(cm, ["e2e_ms", "p50"], 0.0)
        e2e_avg_ms = _safe_get(cm, ["e2e_ms", "avg"], 0.0)
        client_tok_p50 = _safe_get(cm, ["toks", "p50"], 0.0)
        client_tok_avg = _safe_get(cm, ["toks", "avg"], 0.0)
        # Success %
        succ = _safe_get(sm, ["successful_requests"], 0.0)
        success_pct = f"{(succ / num_requests) * 100:.1f}%"

        # Server metrics (may be missing)
        prefill_p50 = _safe_get(sm, ["prefill_throughput", "p50"], 0.0)
        prefill_avg = _safe_get(sm, ["prefill_throughput", "avg"], 0.0)
        decode_p50  = _safe_get(sm, ["decode_throughput",  "p50"], 0.0)
        decode_avg  = _safe_get(sm, ["decode_throughput",  "avg"], 0.0)
        batch_avg   = _safe_get(sm, ["avg_batch_size"], 0.0)

        # Queue metrics
        queue_len_p50 = _safe_get(sm, ["queue_length", "p50"], 0.0)
        queue_len_avg = _safe_get(sm, ["queue_length", "avg"], 0.0)
        queue_lat_p50_ms = _safe_get(sm, ["queue_latency_ms", "p50"], 0.0)
        queue_lat_avg_ms = _safe_get(sm, ["queue_latency_ms", "avg"], 0.0)

        prefill_rows.append([
            _fmt_int(r),
            _fmt_int(prefill_p50),
            _fmt_int(prefill_avg),
            f"{float(batch_avg):.2f}",
            f"{float(queue_len_p50):.1f}",
            f"{float(queue_len_avg):.1f}",
            f"{float(queue_lat_p50_ms):.1f}",
            f"{float(queue_lat_avg_ms):.1f}",
            _fmt_sec(e2e_p50_ms),
            _fmt_sec(e2e_avg_ms),
            _fmt_tokps(client_tok_p50),
            _fmt_tokps(client_tok_avg),
            success_pct,
        ])

        decode_rows.append([
            _fmt_int(r),
            _fmt_int(decode_p50),
            _fmt_int(decode_avg),
            f"{float(batch_avg):.2f}",
            f"{float(queue_len_p50):.1f}",
            f"{float(queue_len_avg):.1f}",
            f"{float(queue_lat_p50_ms):.1f}",
            f"{float(queue_lat_avg_ms):.1f}",
            _fmt_sec(e2e_p50_ms),
            _fmt_sec(e2e_avg_ms),
            _fmt_tokps(client_tok_p50),
            _fmt_tokps(client_tok_avg),
            success_pct,
        ])

    # Pretty print with tabs
    def _print_table(title, rows):
        print()
        print(title)
        print("\t")
        print("Req /s\tP50\tAvg\tAvg Batch Size\tQueue Len P50\tQueue Len Avg\tQueue Lat P50 (ms)\tQueue Lat Avg (ms)\tClient E2E P50 (sec)\tClient E2E Avg (sec)\tClient Tok/s (P50)\tClient Tok/s (Avg)\tSuccess %")
        for row in rows:
            print("\t".join(row))

    _print_table("Prefill", prefill_rows)
    print("\t")
    _print_table("Decode", decode_rows)
    print()


# ==================== PLOTTING FUNCTIONS ====================

def convert_cumulative_to_bucket_counts(cumulative_buckets: Dict[float, float]) -> Tuple[List[float], List[float], List[float]]:
    """Convert cumulative histogram buckets to per-bucket counts."""
    if not cumulative_buckets:
        return [], [], []

    # Sort buckets by upper bound
    sorted_buckets = sorted(cumulative_buckets.items())

    # Extract edges and cumulative counts
    edges = [bound for bound, _ in sorted_buckets]
    cumulative_counts = [count for _, count in sorted_buckets]

    # Convert cumulative to per-bucket counts
    bucket_counts = []
    prev_count = 0.0
    for cum_count in cumulative_counts:
        bucket_counts.append(cum_count - prev_count)
        prev_count = cum_count

    # Create lower edges (starting from 0 or first edge)
    lower_edges = [0.0] + edges[:-1]

    # Calculate bucket centers for plotting
    bucket_centers = [(lower + upper) / 2 for lower, upper in zip(lower_edges, edges)]

    # Filter out +inf bucket for visualization
    filtered_centers = []
    filtered_counts = []
    filtered_edges = []

    for center, count, edge in zip(bucket_centers, bucket_counts, edges):
        if edge != float('inf'):
            filtered_centers.append(center)
            filtered_counts.append(count)
            filtered_edges.append(edge)

    return filtered_edges, filtered_counts, filtered_centers


def plot_distribution_skewness(outputs: List[RequestOutput], outdir: str, rate_suffix: str = ""):
    """Plot histograms showing distribution skewness for TTFT, E2E latency, and tok/s."""
    successful = [o for o in outputs if o.success]
    if not successful:
        print("⚠ No successful requests to plot distribution")
        return

    # Extract metrics
    ttfts_ms = [o.ttft * 1000 for o in successful if o.ttft > 0]
    e2e_latencies_ms = [o.e2e_latency * 1000 for o in successful]
    toks_per_sec = [o.output_tokens / o.e2e_latency for o in successful if o.e2e_latency > 0 and o.output_tokens > 0]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    metrics_data = [
        (ttfts_ms, "TTFT (ms)", "TTFT Distribution"),
        (e2e_latencies_ms, "E2E Latency (ms)", "E2E Latency Distribution"),
        (toks_per_sec, "Tok/s", "Tok/s Distribution"),
    ]

    for ax, (data, xlabel, title) in zip(axes, metrics_data):
        if not data:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title(title)
            continue

        # Calculate statistics
        p50 = float(np.percentile(data, 50))
        avg = float(np.mean(data))

        # Determine skewness
        if avg > p50:
            skew_type = "Right-skewed"
            skew_color = "red"
        elif avg < p50:
            skew_type = "Left-skewed"
            skew_color = "blue"
        else:
            skew_type = "Symmetric"
            skew_color = "green"

        # Plot histogram
        ax.hist(data, bins=30, alpha=0.7, color="gray", edgecolor="black")

        # Add vertical lines for p50 and avg
        ax.axvline(p50, color="orange", linestyle="--", linewidth=2, label=f"P50: {p50:.2f}")
        ax.axvline(avg, color="purple", linestyle="-", linewidth=2, label=f"Avg: {avg:.2f}")

        # Add skewness annotation
        ax.text(0.98, 0.95, f"{skew_type}",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color=skew_color,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = f"distribution_skewness{rate_suffix}.png"
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved distribution skewness plot to {filepath}")


def plot_server_distribution_skewness(server_metrics_before: ServerMetrics, server_metrics_after: ServerMetrics,
                                     server_delta: Dict[str, any], outdir: str, rate_suffix: str = ""):
    """Plot histograms showing distribution skewness for server-side metrics."""
    if server_delta is None:
        print("⚠ No server metrics available to plot distribution")
        return

    # Calculate delta buckets
    delta_prefill_buckets = {}
    all_prefill_bounds = set(server_metrics_before.prefill_throughput_buckets.keys()) | set(server_metrics_after.prefill_throughput_buckets.keys())
    for bound in all_prefill_bounds:
        delta_prefill_buckets[bound] = (server_metrics_after.prefill_throughput_buckets.get(bound, 0) -
                                       server_metrics_before.prefill_throughput_buckets.get(bound, 0))

    delta_decode_buckets = {}
    all_decode_bounds = set(server_metrics_before.decode_throughput_buckets.keys()) | set(server_metrics_after.decode_throughput_buckets.keys())
    for bound in all_decode_bounds:
        delta_decode_buckets[bound] = (server_metrics_after.decode_throughput_buckets.get(bound, 0) -
                                      server_metrics_before.decode_throughput_buckets.get(bound, 0))

    # Get p50 and avg from delta
    prefill = server_delta.get('prefill_throughput', {})
    decode = server_delta.get('decode_throughput', {})

    prefill_p50 = prefill.get('p50', 0.0)
    prefill_avg = prefill.get('avg', 0.0)
    decode_p50 = decode.get('p50', 0.0)
    decode_avg = decode.get('avg', 0.0)

    # Convert cumulative buckets to counts
    prefill_edges, prefill_counts, prefill_centers = convert_cumulative_to_bucket_counts(delta_prefill_buckets)
    decode_edges, decode_counts, decode_centers = convert_cumulative_to_bucket_counts(delta_decode_buckets)

    # Create figure with 2 subplots (prefill and decode)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    metrics_data = [
        (prefill_edges, prefill_counts, prefill_centers, prefill_p50, prefill_avg,
         "Throughput (tok/s)", "Prefill Throughput Distribution"),
        (decode_edges, decode_counts, decode_centers, decode_p50, decode_avg,
         "Throughput (tok/s)", "Decode Throughput Distribution"),
    ]

    for ax, (edges, counts, centers, p50, avg, xlabel, title) in zip(axes, metrics_data):
        if not edges or sum(counts) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title(title)
            continue

        # Determine skewness
        if avg > p50:
            skew_type = "Right-skewed"
            skew_color = "red"
        elif avg < p50:
            skew_type = "Left-skewed"
            skew_color = "blue"
        else:
            skew_type = "Symmetric"
            skew_color = "green"

        # Plot histogram using bucket data
        # Calculate widths for each bucket
        lower_edges = [0.0] + edges[:-1]
        widths = [upper - lower for lower, upper in zip(lower_edges, edges)]

        ax.bar(centers, counts, width=widths, alpha=0.7, color="gray", edgecolor="black", align='center')

        # Add vertical lines for p50 and avg
        if p50 > 0:
            ax.axvline(p50, color="orange", linestyle="--", linewidth=2, label=f"P50: {p50:.1f}")
        if avg > 0:
            ax.axvline(avg, color="purple", linestyle="-", linewidth=2, label=f"Avg: {avg:.1f}")

        # Add skewness annotation
        ax.text(0.98, 0.95, f"{skew_type}",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color=skew_color,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Add interpretation text
        if avg > p50:
            interpretation = f"Avg > P50: {avg:.1f} > {p50:.1f}\nRight tail with high throughput"
        elif avg < p50:
            interpretation = f"Avg < P50: {avg:.1f} < {p50:.1f}\nLeft tail with low throughput"
        else:
            interpretation = f"Avg ≈ P50: {avg:.1f} ≈ {p50:.1f}\nSymmetric distribution"

        ax.text(0.02, 0.95, interpretation,
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency (count)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    filename = f"server_distribution_skewness{rate_suffix}.png"
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved server distribution skewness plot to {filepath}")


def plot_queue_metrics_timeseries(samples: List[QueueMetricsSample], output_dir: str, rate_suffix: str):
    """Create time-series plots for queue metrics."""
    if not samples:
        print("No queue metrics samples to plot")
        return

    timestamps = [s.timestamp for s in samples]
    queue_lengths = [s.num_queue_reqs for s in samples]
    queue_latencies = [s.avg_queue_latency * 1000 for s in samples]  # Convert to ms
    running_reqs = [s.num_running_reqs for s in samples]
    token_usage = [s.token_usage * 100 for s in samples]  # Convert to percentage

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Queue Metrics Over Time (Rate: {rate_suffix})', fontsize=16)

    # Plot 1: Queue Length
    axes[0, 0].plot(timestamps, queue_lengths, 'b-', linewidth=2, label='Queue Length')
    axes[0, 0].fill_between(timestamps, queue_lengths, alpha=0.3)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Number of Requests in Queue')
    axes[0, 0].set_title('Queue Length Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Queue Latency
    axes[0, 1].plot(timestamps, queue_latencies, 'r-', linewidth=2, label='Avg Queue Latency')
    axes[0, 1].fill_between(timestamps, queue_latencies, alpha=0.3, color='red')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Queue Latency (ms)')
    axes[0, 1].set_title('Average Queue Latency Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Running Requests
    axes[1, 0].plot(timestamps, running_reqs, 'g-', linewidth=2, label='Running Requests')
    axes[1, 0].fill_between(timestamps, running_reqs, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Number of Running Requests')
    axes[1, 0].set_title('Running Requests (Batch Size) Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Token Usage
    axes[1, 1].plot(timestamps, token_usage, 'm-', linewidth=2, label='Token Usage')
    axes[1, 1].fill_between(timestamps, token_usage, alpha=0.3, color='magenta')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Token Usage (%)')
    axes[1, 1].set_title('KV Cache Token Usage Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 105])

    plt.tight_layout()
    outpath = os.path.join(output_dir, f"queue_timeseries{rate_suffix}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Queue time-series plot saved to: {outpath}")


def plot_queue_histogram(samples: List[QueueMetricsSample], output_dir: str, rate_suffix: str):
    """Create histogram showing distribution of queue lengths and latencies."""
    if not samples:
        print("No queue metrics samples for histogram")
        return

    queue_lengths = [s.num_queue_reqs for s in samples]
    queue_latencies = [s.avg_queue_latency * 1000 for s in samples if s.avg_queue_latency > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Queue Metrics Distribution (Rate: {rate_suffix})', fontsize=16)

    # Histogram 1: Queue Length Distribution
    axes[0].hist(queue_lengths, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(queue_lengths), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(queue_lengths):.1f}')
    axes[0].axvline(np.percentile(queue_lengths, 95), color='orange', linestyle='dashed', linewidth=2, label=f'P95: {np.percentile(queue_lengths, 95):.1f}')
    axes[0].set_xlabel('Queue Length (# requests)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Queue Length Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram 2: Queue Latency Distribution
    if queue_latencies:
        axes[1].hist(queue_latencies, bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(queue_latencies), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(queue_latencies):.1f}ms')
        axes[1].axvline(np.percentile(queue_latencies, 95), color='orange', linestyle='dashed', linewidth=2, label=f'P95: {np.percentile(queue_latencies, 95):.1f}ms')
        axes[1].set_xlabel('Queue Latency (ms)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Queue Latency Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No queue latency data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Queue Latency Distribution (No Data)')

    plt.tight_layout()
    outpath = os.path.join(output_dir, f"queue_histogram{rate_suffix}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Queue histogram saved to: {outpath}")


def plot_tradeoff(rates, e2e_avg_ms, decode_avg, outdir):
    """Plot trade-off between client E2E latency and server decode throughput."""
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
    """Plot efficiency frontier (Pareto curve) showing trade-off between latency and throughput."""
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
    """Plot a metric series vs request rate."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates, values, marker="o")
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename))
    plt.close(fig)


def plot_client_latency_curves(rates, ttft_p50, ttft_p95, e2e_p50, e2e_p95, outdir):
    """Plot client latency curves (TTFT and E2E) vs request rate."""
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

