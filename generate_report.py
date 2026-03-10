#!/usr/bin/env python3
"""
Generate benchmark report from vllm bench latency results.

Reads the new multi-scenario CSV format produced by bench.sh and generates:
  - PNG charts (throughput tradeoff, per-user speed, etc.)
  - An HTML report with inline data for capacity charts

Usage:
    python generate_report.py results/gpt-oss-120b/ --out-dir report/
    python generate_report.py results/gpt-oss-120b/summary.csv --out-dir report/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


# ── Style (dark, matches report HTML theme: --bg: #060b0a) ───────────────────
_BG = "#060b0a"
_BG_AXES = "#0a1210"
_BORDER = "rgba(255,255,255,0.08)"
_TEXT = "#eaf0ee"  # --text-primary  ≈ rgba(255,255,255,0.92)
_TEXT_MUTED = "#8b9a95"  # --text-muted    ≈ rgba(255,255,255,0.55)
_GRID = "#1a2420"  # subtle grid lines

plt.rcParams.update(
    {
        "figure.facecolor": _BG,
        "axes.facecolor": _BG_AXES,
        "axes.edgecolor": _GRID,
        "axes.labelcolor": _TEXT,
        "text.color": _TEXT,
        "xtick.color": _TEXT_MUTED,
        "ytick.color": _TEXT_MUTED,
        "grid.color": _GRID,
        "grid.alpha": 0.7,
        "legend.facecolor": _BG_AXES,
        "legend.edgecolor": _GRID,
        "legend.labelcolor": _TEXT,
        "font.family": "sans-serif",
        "font.size": 11,
    }
)

COLORS = {
    "prefill": "#3b82f6",
    "decode": "#34d399",  # --accent (emerald)
    "accent": "#f59e0b",
    "band_prefill": "#3b82f6",
    "band_decode": "#34d399",
}


def load_new_format(csv_path: str) -> pd.DataFrame:
    """Load new multi-scenario CSV (from bench.sh)."""
    df = pd.read_csv(csv_path)
    for col in [
        "input_len",
        "output_len",
        "batch_size",
        "avg_latency",
        "prefill_tps_per_user",
        "decode_tps_per_user",
        "prefill_tps_total",
        "decode_tps_total",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["batch_size", "avg_latency"])
    df = df.sort_values(["scenario", "batch_size"])
    return df


def load_old_format(csv_path: str) -> pd.DataFrame:
    """Load old single-scenario CSV (from bench_qwen3_30b_a3b.sh / benchmark_gpt_oss.sh)."""
    df = pd.read_csv(csv_path)
    # Drop debug rows
    df = df[df["system"].str.strip().str.len() > 0].copy()
    df = df[~df["system"].str.startswith(" ")].copy()
    for col in [
        "chips",
        "users",
        "interactivity_toks_per_s_per_user",
        "throughput_toks_per_s_total",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["users", "throughput_toks_per_s_total"])
    df["phase"] = df["config"].apply(
        lambda c: "prefill" if "prefill" in c else "decode"
    )

    # Convert to new format — treat as a single unnamed scenario
    prefill = df[df["phase"] == "prefill"].sort_values("users")
    decode = df[df["phase"] == "decode"].sort_values("users")
    rows = []
    for _, (pr, dr) in enumerate(zip(prefill.itertuples(), decode.itertuples())):
        bs = int(pr.users)
        pf_pu = pr.interactivity_toks_per_s_per_user
        dc_pu = dr.interactivity_toks_per_s_per_user
        pf_t = pr.throughput_toks_per_s_total
        dc_t = dr.throughput_toks_per_s_total
        # Estimate avg_latency from decode total throughput
        rows.append(
            {
                "scenario": "default",
                "input_len": 512,
                "output_len": 256,
                "batch_size": bs,
                "avg_latency": 0,  # not available
                "prefill_tps_per_user": pf_pu,
                "decode_tps_per_user": dc_pu,
                "prefill_tps_total": pf_t,
                "decode_tps_total": dc_t,
            }
        )
    return pd.DataFrame(rows)


def load_report_json(json_path: str) -> tuple[pd.DataFrame, str, str]:
    """Load report_data.json (produced by benchmark_endpoint.py) into a DataFrame.

    Returns (df, model_name, gpu_name).
    """
    with open(json_path) as f:
        data = json.load(f)

    model_name = data.get("model_name", "Unknown Model")
    gpu_name = data.get("gpu_name", "Unknown GPU")

    rows = []
    for scenario_name, sdata in data.get("scenarios", {}).items():
        input_len = sdata["input_len"]
        output_len = sdata["output_len"]
        for m in sdata["measured"]:
            rows.append(
                {
                    "scenario": scenario_name,
                    "input_len": input_len,
                    "output_len": output_len,
                    "batch_size": m["batch_size"],
                    "avg_latency": m.get("e2e", 0),
                    "prefill_tps_per_user": m["prefill_tps_per_user"],
                    "decode_tps_per_user": m["decode_tps_per_user"],
                    "prefill_tps_total": m["prefill_tps_total"],
                    "decode_tps_total": m["decode_tps_total"],
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["scenario", "batch_size"])
    return df, model_name, gpu_name


def detect_format(csv_path: str) -> str:
    """Detect CSV format by checking header."""
    with open(csv_path) as f:
        header = f.readline().strip()
    if header.startswith("scenario,"):
        return "new"
    return "old"


def load_csv(csv_path: str) -> pd.DataFrame:
    fmt = detect_format(csv_path)
    if fmt == "new":
        return load_new_format(csv_path)
    else:
        return load_old_format(csv_path)


# ── Chart generators (work on per-scenario data) ────────────────────────────


def fig_throughput_tradeoff(df: pd.DataFrame, model_name: str, out_dir: str):
    """Interactivity vs Throughput per chip (classic tradeoff curve)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = df["scenario"].unique()
    colors = _distinct_colors(len(scenarios))
    for i, scenario in enumerate(scenarios):
        g = df[df["scenario"] == scenario].sort_values("batch_size")
        color = colors[i]
        ax.plot(
            g["decode_tps_per_user"],
            g["decode_tps_total"],
            marker="o",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{_friendly_name(scenario)} decode",
        )
        for _, row in g.iterrows():
            ax.annotate(
                f"bs={int(row['batch_size'])}",
                (row["decode_tps_per_user"], row["decode_tps_total"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color=color,
                alpha=0.8,
            )

    ax.set_xlabel("Per-User Decode Speed (tok/s/user)")
    ax.set_ylabel("System Decode Throughput (tok/s)")
    ax.set_title(f"{model_name} — Interactivity vs Throughput Tradeoff")
    _legend_below(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_throughput_tradeoff.png"), dpi=250)
    plt.close(fig)


def fig_system_throughput(df: pd.DataFrame, model_name: str, out_dir: str):
    """Total system throughput vs batch size (bar chart) — first scenario."""
    scenario = df["scenario"].iloc[0]
    g = df[df["scenario"] == scenario].sort_values("batch_size")

    fig, ax = plt.subplots(figsize=(10, 6))
    batch_sizes = g["batch_size"].values
    x = np.arange(len(batch_sizes))
    width = 0.35

    bars_p = ax.bar(
        x - width / 2,
        g["prefill_tps_total"].values,
        width,
        label="Prefill",
        color=COLORS["prefill"],
        alpha=0.85,
    )
    bars_d = ax.bar(
        x + width / 2,
        g["decode_tps_total"].values,
        width,
        label="Decode",
        color=COLORS["decode"],
        alpha=0.85,
    )

    for bars in [bars_p, bars_d]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 15,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#667085",
            )

    ax.set_xlabel("Batch Size (concurrent users)")
    ax.set_ylabel("System Throughput (tok/s)")
    ax.set_title(f"{model_name} — System Throughput ({_friendly_name(scenario)})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    _legend_below(ax)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_system_throughput.png"), dpi=250)
    plt.close(fig)


def fig_per_user_speed(df: pd.DataFrame, model_name: str, out_dir: str):
    """Per-user generation speed vs batch size across scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = df["scenario"].unique()
    colors = _distinct_colors(len(scenarios))
    for i, scenario in enumerate(scenarios):
        g = df[df["scenario"] == scenario].sort_values("batch_size")
        color = colors[i]
        ax.plot(
            g["batch_size"],
            g["decode_tps_per_user"],
            marker="o",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{_friendly_name(scenario)}",
        )

    thresholds = [
        (50, "Excellent (50 tok/s)", "#10b981"),
        (25, "Good (25 tok/s)", "#f59e0b"),
        (15, "Acceptable (15 tok/s)", "#ef4444"),
    ]
    for val, label, color in thresholds:
        ax.axhline(y=val, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            df["batch_size"].max() * 0.95,
            val + 2,
            label,
            ha="right",
            fontsize=8,
            color=color,
            alpha=0.7,
        )

    ax.set_xlabel("Batch Size (concurrent users)")
    ax.set_ylabel("Per-User Decode Speed (tok/s/user)")
    ax.set_title(f"{model_name} — Per-User Decode Speed")
    _legend_below(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_per_user_speed.png"), dpi=250)
    plt.close(fig)


def fig_scaling_efficiency(df: pd.DataFrame, model_name: str, out_dir: str):
    """Scaling efficiency across scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = df["scenario"].unique()
    colors = _distinct_colors(len(scenarios))
    for i, scenario in enumerate(scenarios):
        g = df[df["scenario"] == scenario].sort_values("batch_size")
        bs1_throughput = g["decode_tps_total"].values[0]
        bs_vals = g["batch_size"].values
        ideal = bs1_throughput * bs_vals
        actual = g["decode_tps_total"].values
        efficiency = (actual / ideal) * 100

        color = colors[i]
        ax.plot(
            bs_vals,
            efficiency,
            marker="o",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{_friendly_name(scenario)}",
        )

    ax.axhline(y=100, color="#667085", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Batch Size (concurrent users)")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title(f"{model_name} — Decode Scaling Efficiency")
    ax.set_ylim(0, 115)
    _legend_below(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_scaling_efficiency.png"), dpi=250)
    plt.close(fig)


def fig_latency(df: pd.DataFrame, model_name: str, out_dir: str):
    """E2E latency breakdown for first scenario."""
    scenario = df["scenario"].iloc[0]
    g = df[df["scenario"] == scenario].sort_values("batch_size")

    fig, ax = plt.subplots(figsize=(10, 6))
    input_len = int(g["input_len"].iloc[0])
    output_len = int(g["output_len"].iloc[0])

    prefill_time = input_len / g["prefill_tps_per_user"].values
    decode_time = output_len / g["decode_tps_per_user"].values
    e2e = prefill_time + decode_time
    batch_sizes = g["batch_size"].values

    ax.bar(
        np.arange(len(batch_sizes)),
        prefill_time,
        width=0.6,
        label=f"Prefill ({input_len} tok)",
        color=COLORS["prefill"],
        alpha=0.85,
    )
    ax.bar(
        np.arange(len(batch_sizes)),
        decode_time,
        width=0.6,
        bottom=prefill_time,
        label=f"Decode ({output_len} tok)",
        color=COLORS["decode"],
        alpha=0.85,
    )

    for i, (xi, total) in enumerate(zip(np.arange(len(batch_sizes)), e2e)):
        ax.text(
            xi,
            total + 0.05,
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#667085",
        )

    ax.set_xlabel("Batch Size (concurrent users)")
    ax.set_ylabel("Estimated Latency (seconds)")
    ax.set_title(
        f"{model_name} — End-to-End Latency ({_friendly_name(scenario)})"
    )
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    _legend_below(ax)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_latency.png"), dpi=250)
    plt.close(fig)


# ── Helpers for inline HTML chart generation ─────────────────────────────

_BASE_COLORS = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4",
                 "#f43f5e", "#a3e635", "#38bdf8", "#fb923c", "#c084fc", "#2dd4bf"]


def _distinct_colors(n: int) -> list[str]:
    """Return *n* visually distinct colors, generating extras via HSL if needed."""
    if n <= len(_BASE_COLORS):
        return _BASE_COLORS[:n]
    # Generate additional colors evenly spaced in hue
    import colorsys
    colors = list(_BASE_COLORS)
    for i in range(n - len(_BASE_COLORS)):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for max spread
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.75)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# Keep backwards compat aliases
CONCURRENCY_COLORS = _BASE_COLORS
CONCURRENCY_MARKERS = ["o", "o", "o", "o", "o", "o"]


def fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=250,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _legend_below(ax, **kwargs):
    """Place legend below the axes so it never overlaps data."""
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(ax.get_legend_handles_labels()[1].__len__(), 4),
        framealpha=0.9,
        **kwargs,
    )


def _setup_ax(ax, xlabel: str, ylabel: str, title: str = ""):
    """Apply shared dark-theme styling to an axis."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


_SCENARIO_DISPLAY_NAMES = {
    "1k": "Chatbot",
    "4k": "RAG / QA",
    "16k": "Agentic",
    "32k": "Tool Calling Agentic",
}


def _friendly_name(scenario_id: str) -> str:
    """Map a scenario ID like '1k' or '1k-in-128-out' to a friendly display name."""
    if scenario_id in _SCENARIO_DISPLAY_NAMES:
        return _SCENARIO_DISPLAY_NAMES[scenario_id]
    # Try matching the prefix (e.g. '1k-in-128-out' -> '1k')
    prefix = scenario_id.split("-")[0]
    return _SCENARIO_DISPLAY_NAMES.get(prefix, scenario_id)


def _sorted_scenarios(scenarios_data: dict):
    """Return scenario IDs sorted by input_len."""
    return sorted(scenarios_data.keys(), key=lambda k: scenarios_data[k]["input_len"])


def _context_labels(scenarios_data: dict, sorted_ids: list) -> list[str]:
    """Build human-readable context labels using friendly scenario names."""
    labels = []
    for sid in sorted_ids:
        labels.append(_friendly_name(sid))
    return labels


def _get_metric(
    scenarios_data: dict, sorted_ids: list, metric_key: str, batch_size: int
) -> list:
    """Extract a metric value across all scenarios for a given batch size."""
    values = []
    for sid in sorted_ids:
        m = next(
            (
                p
                for p in scenarios_data[sid]["measured"]
                if p["batch_size"] == batch_size
            ),
            None,
        )
        values.append(float(m[metric_key]) if m and m.get(metric_key) is not None else 0.0)
    return values


def chart_throughput_range(scenarios_data: dict, batch_sizes: list) -> str:
    """Envelope chart: per-user decode speed range (bs=1 solid, bs=max dashed)."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    max_bs = max(batch_sizes)

    top = _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", 1)
    bottom = _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", max_bs)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.plot(
        x,
        top,
        color="#3b82f6",
        linewidth=2,
        marker="o",
        markersize=7,
        label="Single user (bs=1)",
        zorder=3,
    )
    ax.plot(
        x,
        bottom,
        color="#3b82f6",
        linewidth=2,
        marker="o",
        markersize=7,
        linestyle="--",
        label=f"Max concurrency (bs={max_bs})",
        zorder=3,
    )
    ax.fill_between(x, bottom, top, color="#3b82f6", alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    _setup_ax(ax, "Context Length", "Per-User Speed (tok/s)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_ttft_range(scenarios_data: dict, batch_sizes: list) -> str:
    """Envelope chart: TTFT range (bs=1 dashed, bs=max solid)."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    max_bs = max(batch_sizes)

    ttft_bs1 = _get_metric(scenarios_data, sorted_ids, "ttft", 1)
    ttft_max = _get_metric(scenarios_data, sorted_ids, "ttft", max_bs)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.plot(
        x,
        ttft_max,
        color="#f59e0b",
        linewidth=2,
        marker="o",
        markersize=7,
        label=f"Max concurrency (bs={max_bs})",
        zorder=3,
    )
    ax.plot(
        x,
        ttft_bs1,
        color="#f59e0b",
        linewidth=2,
        marker="o",
        markersize=7,
        linestyle="--",
        label="Single user (bs=1)",
        zorder=3,
    )
    ax.fill_between(x, ttft_bs1, ttft_max, color="#f59e0b", alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    _setup_ax(ax, "Context Length", "TTFT (seconds)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_system_throughput(scenarios_data: dict, batch_sizes: list) -> str:
    """Multi-line chart: system throughput per concurrency level."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    colors = _distinct_colors(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    for i, bs in enumerate(batch_sizes):
        vals = _get_metric(scenarios_data, sorted_ids, "decode_tps_total", bs)
        c = colors[i]
        m = CONCURRENCY_MARKERS[i % len(CONCURRENCY_MARKERS)]
        ax.plot(
            x,
            vals,
            color=c,
            linewidth=2,
            marker=m,
            markersize=7,
            label=f"{bs} Req{'s' if bs > 1 else ''}",
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    _setup_ax(ax, "Context Length", "System Throughput (tok/s)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_itl(scenarios_data: dict, batch_sizes: list) -> str:
    """Multi-line chart: inter-token latency (derived 1000/decode_tps)."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    colors = _distinct_colors(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    for i, bs in enumerate(batch_sizes):
        raw = _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", bs)
        vals = [1000.0 / v if v else None for v in raw]
        c = colors[i]
        m = CONCURRENCY_MARKERS[i % len(CONCURRENCY_MARKERS)]
        ax.plot(
            x,
            vals,
            color=c,
            linewidth=2,
            marker=m,
            markersize=7,
            label=f"{bs} Req{'s' if bs > 1 else ''}",
            zorder=3,
        )
    # 20ms threshold line
    ax.axhline(y=20, color="#eab308", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        len(labels) - 0.5,
        21,
        "20ms good",
        ha="right",
        fontsize=9,
        color="#eab308",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    _setup_ax(ax, "Context Length", "Inter-Token Latency (ms)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_decode_speed(scenarios_data: dict, batch_sizes: list) -> str:
    """Multi-line chart: per-user decode speed per concurrency level."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    colors = _distinct_colors(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    for i, bs in enumerate(batch_sizes):
        vals = _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", bs)
        c = colors[i]
        m = CONCURRENCY_MARKERS[i % len(CONCURRENCY_MARKERS)]
        ax.plot(
            x,
            vals,
            color=c,
            linewidth=2,
            marker=m,
            markersize=7,
            label=f"{bs} Req{'s' if bs > 1 else ''}",
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    _setup_ax(ax, "Context Length", "Decode Speed (tok/s)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_scaling_efficiency(scenarios_data: dict, batch_sizes: list) -> str:
    """Multi-line chart: scaling efficiency per concurrency level."""
    sorted_ids = _sorted_scenarios(scenarios_data)
    labels = _context_labels(scenarios_data, sorted_ids)
    colors = _distinct_colors(len(batch_sizes))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    for i, bs in enumerate(batch_sizes):
        vals = []
        for sid in sorted_ids:
            m1 = next(
                (p for p in scenarios_data[sid]["measured"] if p["batch_size"] == 1),
                None,
            )
            mbs = next(
                (p for p in scenarios_data[sid]["measured"] if p["batch_size"] == bs),
                None,
            )
            if m1 and mbs:
                vals.append(
                    (mbs["decode_tps_total"] / (m1["decode_tps_total"] * bs)) * 100
                )
            else:
                vals.append(None)
        c = colors[i]
        m = CONCURRENCY_MARKERS[i % len(CONCURRENCY_MARKERS)]
        ax.plot(
            x,
            vals,
            color=c,
            linewidth=2,
            marker=m,
            markersize=7,
            label=f"{bs} Req{'s' if bs > 1 else ''}",
            zorder=3,
        )
    # Ideal 100% line
    ax.axhline(y=100, color="#667085", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    _setup_ax(ax, "Context Length", "Scaling Efficiency (%)")
    _legend_below(ax)
    fig.tight_layout()
    return fig_to_base64(fig)


def _get_scenario_thresholds(scenario_data: dict) -> dict:
    """Return quality thresholds for capacity analysis.

    Left axis is ITL (ms) = 1000 / decode_tps_per_user.
    Right axis is TTFT (s).  Both go up = worse, so thresholds are maximums.
    Thresholds are held constant across all scenarios for consistent comparison.
    """
    return {
        "maxITL": 20,           # 20 ms max inter-token latency
        "rightMetric": "ttft",
        "rightLabel": "TTFT (s)",
        "maxTTFT": 1.0,         # 1 second max time-to-first-token
    }


def _find_capacity(measured: list, thresholds: dict) -> int:
    """Find max concurrency that meets quality thresholds."""
    cap = 0
    for m in measured:
        ok = True
        itl = 1000 / m["decode_tps_per_user"] if m["decode_tps_per_user"] > 0 else float("inf")
        if "maxTTFT" in thresholds and m["ttft"] > thresholds["maxTTFT"]:
            ok = False
        if "maxITL" in thresholds and itl > thresholds["maxITL"]:
            ok = False
        if ok:
            cap = m["batch_size"]
        else:
            break
    return cap


def chart_capacity(scenario_name: str, scenario_data: dict, thresholds: dict) -> str:
    """Dual-axis capacity chart for a single scenario.

    Left axis: ITL (ms) — inter-token latency (= 1000 / decode_tps_per_user).
    Right axis: TTFT (s).
    Both go up = worse; threshold lines are maximums.
    """
    measured = scenario_data["measured"]
    concurrencies = [m["batch_size"] for m in measured]
    itl_vals = [
        1000 / m["decode_tps_per_user"] if m["decode_tps_per_user"] > 0 else 0
        for m in measured
    ]
    right_key = thresholds["rightMetric"]
    right_vals = [m[right_key] for m in measured]

    capacity = _find_capacity(measured, thresholds)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(concurrencies))

    # Left axis: ITL (green)
    ax1.plot(
        x,
        itl_vals,
        color="#10b981",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label="ITL (ms)",
        zorder=3,
    )
    ax1.fill_between(x, itl_vals, color="#10b981", alpha=0.08)
    ax1.set_ylabel("Inter-Token Latency (ms)", color="#10b981")
    ax1.tick_params(axis="y", colors="#10b981")
    ax1.set_ylim(bottom=0)

    # Right axis: TTFT (orange)
    ax2.plot(
        x,
        right_vals,
        color="#f59e0b",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label=thresholds["rightLabel"],
        zorder=3,
    )
    ax2.fill_between(x, right_vals, color="#f59e0b", alpha=0.08)
    ax2.set_ylabel(thresholds["rightLabel"], color="#f59e0b")
    ax2.tick_params(axis="y", colors="#f59e0b")
    ax2.set_ylim(bottom=0)

    # Threshold lines — scale both axes so the ITL and TTFT thresholds
    # sit at the same vertical position (single dashed line).
    max_itl_thresh = thresholds["maxITL"]
    right_thresh_val = thresholds["maxTTFT"]

    itl_data_max = max(itl_vals) if itl_vals else 0
    right_data_max = max(right_vals) if right_vals else 0

    # Determine where the threshold should sit as a fraction of the axis.
    # Use the larger of (data_max, thresh) * 1.25 as the axis top, but
    # force both axes to place their threshold at the same fraction.
    left_top = max(itl_data_max, max_itl_thresh) * 1.25
    right_top = max(right_data_max, right_thresh_val) * 1.25

    # Compute the fraction each threshold occupies, pick the larger one,
    # then stretch the other axis so both thresholds align.
    frac_left = max_itl_thresh / left_top
    frac_right = right_thresh_val / right_top
    target_frac = max(frac_left, frac_right)
    # Ensure threshold isn't too close to the top
    target_frac = min(target_frac, 0.65)

    left_top = max_itl_thresh / target_frac
    right_top = right_thresh_val / target_frac

    ax1.set_ylim(0, left_top)
    ax2.set_ylim(0, right_top)

    # Draw a single threshold line (on left axis — it aligns with right too)
    ax1.axhline(
        y=max_itl_thresh,
        color="#eab308",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.text(
        0.5,
        max_itl_thresh * 1.05,
        f"{max_itl_thresh}ms ITL",
        ha="left",
        fontsize=9,
        color="#eab308",
        alpha=0.8,
    )
    ax2.text(
        len(concurrencies) - 0.5,
        right_thresh_val * 1.05,
        f"{right_thresh_val}s TTFT",
        ha="right",
        fontsize=9,
        color="#eab308",
        alpha=0.8,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in concurrencies])
    ax1.set_xlabel("Concurrent Requests")
    ax1.grid(True, alpha=0.3)

    inp = scenario_data["input_len"]
    out = scenario_data["output_len"]
    in_label = f"{inp // 1024}K" if inp >= 1024 else str(inp)
    out_label = f"{out // 1024}K" if out >= 1024 else str(out)
    ax1.set_title(_friendly_name(scenario_name))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(labels1) + len(labels2), 4), framealpha=0.9,
    )

    fig.subplots_adjust(left=0.10, right=0.88, top=0.92, bottom=0.12)
    return fig_to_base64(fig)


# ── Capacity data generation ─────────────────────────────────────────────


def build_scenario_data(df: pd.DataFrame) -> dict:
    """Build per-scenario data dict for the HTML report.

    For each scenario, provides measured data points and power-law fit
    parameters for extrapolation beyond measured range.
    """
    scenarios_data = {}

    for scenario in df["scenario"].unique():
        g = df[df["scenario"] == scenario].sort_values("batch_size")
        input_len = int(g["input_len"].iloc[0])
        output_len = int(g["output_len"].iloc[0])

        measured = []
        for _, row in g.iterrows():
            bs = int(row["batch_size"])
            prefill_pu = float(row["prefill_tps_per_user"])
            decode_pu = float(row["decode_tps_per_user"])
            ttft = input_len / prefill_pu if prefill_pu else float("inf")
            decode_time = output_len / decode_pu if decode_pu else float("inf")
            e2e = ttft + decode_time
            measured.append(
                {
                    "batch_size": bs,
                    "prefill_tps_per_user": round(prefill_pu, 1),
                    "decode_tps_per_user": round(decode_pu, 1),
                    "prefill_tps_total": round(float(row["prefill_tps_total"]), 0),
                    "decode_tps_total": round(float(row["decode_tps_total"]), 0),
                    "ttft": round(ttft, 3),
                    "decode_time": round(decode_time, 3),
                    "e2e": round(e2e, 3),
                }
            )

        scenarios_data[scenario] = {
            "input_len": input_len,
            "output_len": output_len,
            "measured": measured,
        }

    return scenarios_data


# ── HTML report generation ───────────────────────────────────────────────


def generate_html_report(
    df: pd.DataFrame, model_name: str, gpu_name: str, scenarios_data: dict, out_dir: str
):
    """Generate static HTML report with embedded PNG charts."""
    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    if not os.path.exists(template_path):
        print(f"WARNING: HTML template not found at {template_path}", file=sys.stderr)
        return

    with open(template_path) as f:
        html = f.read()

    # ── Compute basic data from first scenario ──────────────────────────
    sorted_ids = _sorted_scenarios(scenarios_data)
    first_scenario = sorted_ids[0]
    sd0 = scenarios_data[first_scenario]
    input_len = sd0["input_len"]
    output_len = sd0["output_len"]
    batch_sizes = [m["batch_size"] for m in sd0["measured"]]
    max_bs = max(batch_sizes)

    decode_per_user = [m["decode_tps_per_user"] for m in sd0["measured"]]
    decode_total = [m["decode_tps_total"] for m in sd0["measured"]]
    prefill_per_user = [m["prefill_tps_per_user"] for m in sd0["measured"]]

    peak_decode = max(decode_per_user)
    max_sys_throughput = max(decode_total)
    max_sys_bs = batch_sizes[decode_total.index(max_sys_throughput)]
    best_ttft = input_len / max(prefill_per_user)
    eff_at_max = (decode_total[-1] / (decode_total[0] * batch_sizes[-1])) * 100

    # ── Generate chart images ───────────────────────────────────────────
    print("  Generating inline charts for HTML report...")
    img_throughput_range = chart_throughput_range(scenarios_data, batch_sizes)
    img_ttft_range = chart_ttft_range(scenarios_data, batch_sizes)
    img_system_throughput = chart_system_throughput(scenarios_data, batch_sizes)
    img_itl = chart_itl(scenarios_data, batch_sizes)
    img_decode_speed = chart_decode_speed(scenarios_data, batch_sizes)
    img_scaling_efficiency = chart_scaling_efficiency(scenarios_data, batch_sizes)

    # ── Explainer text (data-driven) ────────────────────────────────────
    def _find_best(metric_key, mode="max"):
        best_val = -float("inf") if mode == "max" else float("inf")
        best_ctx, best_bs = "", 0
        for sid in sorted_ids:
            for m in scenarios_data[sid]["measured"]:
                v = m.get(metric_key)
                if v is None:
                    continue
                if (mode == "max" and v > best_val) or (mode == "min" and v < best_val):
                    best_val = v
                    best_ctx = _friendly_name(sid)
                    best_bs = m["batch_size"]
        return best_val, best_ctx, best_bs

    # Throughput range explainer
    peak_val, peak_ctx, _ = _find_best("decode_tps_per_user", "max")
    at_max_bs = [_get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", max_bs)]
    flat = [
        v
        for v in _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", max_bs)
        if v
    ]
    min_at_max = min(flat) if flat else 0
    max_at_max = max(flat) if flat else 0
    explainer_throughput = (
        f"Peak per-user decode speed is <strong>{peak_val:.1f} tok/s</strong> "
        f"(single user, {peak_ctx}). At maximum concurrency ({max_bs} users), "
        f"per-user speed ranges from {min_at_max:.1f} to {max_at_max:.1f} tok/s "
        f"across context lengths — the shaded band shows how much individual "
        f"throughput degrades under load."
    )

    # TTFT range explainer
    best_ttft, best_ttft_ctx, best_ttft_bs = _find_best("ttft", "min")
    worst_ttft, worst_ttft_ctx, worst_ttft_bs = _find_best("ttft", "max")
    if best_ttft < 0.1:
        ttft_rating = 'Best-case TTFT is <strong style="color:#10b981">excellent</strong> — under 100ms.'
    elif best_ttft < 0.5:
        ttft_rating = 'Best-case TTFT is <strong style="color:#f59e0b">good</strong> for this context range.'
    else:
        ttft_rating = (
            "TTFT is elevated even in the best case — may indicate scheduling overhead."
        )
    explainer_ttft = (
        f"Best TTFT is <strong>{best_ttft * 1000:.0f}ms</strong> "
        f"({best_ttft_ctx}, {best_ttft_bs} concurrent), worst case "
        f"<strong>{worst_ttft:.2f}s</strong> ({worst_ttft_ctx}, {worst_ttft_bs} concurrent). "
        f"{ttft_rating} Industry references: ~50ms at 1K context, ~2s at 32K, ~18s at 128K on top hardware."
    )

    # System throughput explainer
    peak_sys, peak_sys_ctx, peak_sys_bs = _find_best("decode_tps_total", "max")
    tok_per_hour = round(peak_sys * 3600)
    explainer_sys = (
        f"Peak system throughput is <strong>{peak_sys:.1f} tok/s</strong> "
        f"({peak_sys_ctx}, {peak_sys_bs} concurrent users), equivalent to "
        f"~{tok_per_hour:,} tokens/hour. Higher concurrency increases total throughput "
        f"as the GPU processes more requests in parallel, even though per-user speed decreases."
    )

    # ITL explainer
    best_decode_val, best_decode_ctx, _ = _find_best("decode_tps_per_user", "max")
    worst_decode_val, worst_decode_ctx, worst_decode_bs = _find_best(
        "decode_tps_per_user", "min"
    )
    best_itl = 1000 / best_decode_val
    worst_itl = 1000 / worst_decode_val
    if best_itl < 10:
        itl_rating = 'This is <strong style="color:#10b981">excellent</strong> — well under the 20ms threshold where streaming feels instantaneous.'
    elif best_itl < 20:
        itl_rating = 'This is <strong style="color:#10b981">good</strong> — under the 20ms threshold where streaming feels instantaneous.'
    else:
        itl_rating = (
            "This is above the 20ms threshold — streaming may feel noticeably chunky."
        )
    explainer_itl = (
        f"Best inter-token latency is <strong>{best_itl:.1f}ms</strong> "
        f"({best_decode_ctx}, single user), rising to <strong>{worst_itl:.1f}ms</strong> "
        f"under maximum load ({worst_decode_ctx}, {worst_decode_bs} concurrent). "
        f"{itl_rating} For reference, top-tier H200 SXM setups achieve 4–8ms at short contexts."
    )

    # Decode speed explainer
    if best_decode_val >= 150:
        decode_rating = 'Peak speed is <strong style="color:#10b981">excellent</strong> — matching top-tier single-user performance.'
    elif best_decode_val >= 80:
        decode_rating = 'Peak speed is <strong style="color:#f59e0b">good</strong> — solid for interactive use.'
    else:
        decode_rating = "Peak speed is below typical expectations for interactive use."
    explainer_decode = (
        f"Per-user decode speed ranges from <strong>{best_decode_val:.1f} tok/s</strong> "
        f"({best_decode_ctx}, single user) down to <strong>{worst_decode_val:.1f} tok/s</strong> "
        f"({worst_decode_ctx}, {worst_decode_bs} concurrent). {decode_rating} "
        f"Industry benchmarks show 150+ tok/s at short contexts and 80+ tok/s at 32K on high-end GPUs."
    )

    # Scaling efficiency explainer
    best_eff = 0
    worst_eff = 100
    best_eff_ctx = worst_eff_ctx = ""
    for sid in sorted_ids:
        m1 = next(
            (p for p in scenarios_data[sid]["measured"] if p["batch_size"] == 1), None
        )
        m_max = next(
            (p for p in scenarios_data[sid]["measured"] if p["batch_size"] == max_bs),
            None,
        )
        if not m1 or not m_max:
            continue
        eff = (m_max["decode_tps_total"] / (m1["decode_tps_total"] * max_bs)) * 100
        if eff > best_eff:
            best_eff = eff
            best_eff_ctx = _friendly_name(sid)
        if eff < worst_eff:
            worst_eff = eff
            worst_eff_ctx = _friendly_name(sid)
    if worst_eff >= 90:
        eff_rating = 'Scaling is <strong style="color:#10b981">excellent</strong> across all contexts — near-linear throughput gains with added users.'
    elif worst_eff >= 70:
        eff_rating = 'Scaling is <strong style="color:#f59e0b">good</strong> overall, though longer contexts show more contention.'
    else:
        eff_rating = "Scaling drops significantly at longer contexts — GPU memory bandwidth is the likely bottleneck."
    explainer_eff = (
        f"At {max_bs}x concurrency, scaling efficiency ranges from "
        f"<strong>{best_eff:.0f}%</strong> ({best_eff_ctx}) to "
        f"<strong>{worst_eff:.0f}%</strong> ({worst_eff_ctx}). {eff_rating} "
        f"Values above 90% are considered excellent; below 50% indicates severe resource contention."
    )

    # ── Chart captions ─────────────────────────────────────────────────
    caption_throughput = (
        f"Per-user decode throughput (tok/s) from single-user to {max_bs}-user concurrency "
        f"across {len(sorted_ids)} context-length scenarios."
    )
    caption_ttft = (
        f"Time to first token (seconds) from single-user to {max_bs}-user concurrency "
        f"across context lengths."
    )
    caption_sys = (
        f"Aggregate system decode throughput (tok/s) at each concurrency level "
        f"across {len(sorted_ids)} context-length scenarios."
    )
    caption_itl = (
        f"Average inter-token latency (ms) derived from per-user decode speed. "
        f"The 20ms threshold marks the boundary of perceptually instant streaming."
    )
    caption_decode = (
        f"Per-user decode speed (tok/s) at each concurrency level. "
        f"Higher is better for interactive use."
    )
    caption_eff = (
        f"Scaling efficiency (%) at each concurrency level relative to ideal linear scaling. "
        f"100% means no per-user speed loss when adding concurrent users."
    )

    # ── Summary table (below System Throughput chart) ──────────────────
    summary_rows = []
    # Pick key concurrency levels for the summary
    summary_bs_list = [1]
    if len(batch_sizes) >= 3:
        summary_bs_list.append(batch_sizes[len(batch_sizes) // 2])
    summary_bs_list.append(max_bs)
    # Remove duplicates while preserving order
    seen = set()
    summary_bs_list = [b for b in summary_bs_list if not (b in seen or seen.add(b))]

    for bs in summary_bs_list:
        # Get throughput across all scenarios for this bs
        vals = _get_metric(scenarios_data, sorted_ids, "decode_tps_total", bs)
        peak_val = max(v for v in vals if v is not None) if any(v for v in vals) else 0
        min_val = min(v for v in vals if v is not None) if any(v for v in vals) else 0
        # Per-user decode at this bs (first scenario)
        pu_vals = _get_metric(scenarios_data, sorted_ids, "decode_tps_per_user", bs)
        pu_peak = (
            max(v for v in pu_vals if v is not None) if any(v for v in pu_vals) else 0
        )
        tok_hr = round(peak_val * 3600)
        if bs == 1:
            condition = "Single user"
        elif bs == max_bs:
            condition = f"Max concurrency ({bs} reqs)"
        else:
            condition = f"Mid concurrency ({bs} reqs)"
        summary_rows.append(
            f"<tr><td>{condition}</td><td>{peak_val:,.0f}</td>"
            f"<td>{pu_peak:,.1f}</td><td>{tok_hr:,}</td></tr>"
        )
    summary_table_html = (
        '<table class="summary-table" style="margin-top: 20px;">'
        "<thead><tr>"
        "<th>Condition</th><th>Peak System Throughput (tok/s)</th>"
        "<th>Peak Per-User (tok/s)</th><th>Tokens/Hour</th>"
        "</tr></thead><tbody>" + "\n".join(summary_rows) + "</tbody></table>"
    )

    # ── Narrative paragraph ────────────────────────────────────────────
    tok_per_hour_peak = round(peak_sys * 3600)
    if tok_per_hour_peak >= 1_000_000:
        tok_hr_str = f"<strong>{tok_per_hour_peak / 1_000_000:.1f} million</strong>"
    else:
        tok_hr_str = f"<strong>{tok_per_hour_peak:,}</strong>"
    narrative_sys = (
        f"At peak throughput ({peak_sys_bs} concurrent requests, {peak_sys_ctx}), "
        f"this configuration produces approximately {tok_hr_str} tokens per hour "
        f"with a per-user decode speed of <strong>{worst_decode_val:.1f} tok/s</strong> "
        f"under maximum load. Single-user performance reaches "
        f"<strong>{peak_decode:.1f} tok/s</strong>, and scaling efficiency at "
        f"{max_bs}x concurrency is <strong>{eff_at_max:.0f}%</strong>."
    )

    # ── Specs HTML ──────────────────────────────────────────────────────
    scenario_names = ", ".join(_friendly_name(sid) for sid in sorted_ids)
    specs_items = [
        ("Model", model_name),
        ("GPU", gpu_name),
        ("Scenarios", scenario_names),
    ]
    specs_html = "\n    ".join(
        f'<div class="spec-item"><span class="spec-label">{label}</span>'
        f'<span class="spec-value">{value}</span></div>'
        for label, value in specs_items
    )

    # ── Key metrics HTML ────────────────────────────────────────────────
    metrics_cards = [
        (
            "metric-blue",
            "Peak Decode Speed",
            f"{peak_decode:.1f}",
            "tok/s",
            "Single user (batch size 1)",
        ),
        (
            "metric-green",
            "Max System Throughput",
            f"{int(max_sys_throughput)}",
            "tok/s",
            f"Decode at batch size {max_sys_bs}",
        ),
        (
            "metric-orange",
            "Best TTFT",
            f"{best_ttft:.2f}",
            "s",
            f"{input_len} in, single user",
        ),
        (
            "metric-cyan",
            "Scaling Efficiency",
            f"{eff_at_max:.0f}%",
            f"@ bs={max_bs}",
            "vs ideal linear scaling",
        ),
    ]
    key_metrics_html = "\n    ".join(
        f'<div class="metric-card {cls}"><div class="label">{label}</div>'
        f'<div class="value">{val} <span class="unit">{unit}</span></div>'
        f'<div class="detail">{detail}</div></div>'
        for cls, label, val, unit, detail in metrics_cards
    )

    # ── Data table HTML ─────────────────────────────────────────────────
    table_rows = []
    dc_total_0 = decode_total[0]
    for m in sd0["measured"]:
        bs = m["batch_size"]
        dc_pu = m["decode_tps_per_user"]
        sys_dc = int(m["decode_tps_total"])
        pf_time = input_len / m["prefill_tps_per_user"]
        dc_time = output_len / dc_pu
        e2e = pf_time + dc_time
        eff = (m["decode_tps_total"] / (dc_total_0 * bs)) * 100
        table_rows.append(
            f"<tr><td>{bs}</td><td>{dc_pu:.1f}</td>"
            f"<td>{sys_dc}</td><td>{pf_time:.2f}s</td><td>{dc_time:.2f}s</td>"
            f"<td>{e2e:.2f}s</td><td>{eff:.0f}%</td></tr>"
        )
    table_body_html = "\n        ".join(table_rows)

    # ── Capacity cards & charts HTML ────────────────────────────────────
    card_colors = [
        "metric-green",
        "metric-blue",
        "metric-purple",
        "metric-cyan",
        "metric-orange",
    ]
    cap_cards = []
    cap_charts = []
    scenario_ids = _sorted_scenarios(scenarios_data)

    for i, sid in enumerate(scenario_ids):
        sd = scenarios_data[sid]
        thresholds = _get_scenario_thresholds(sd)
        capacity = _find_capacity(sd["measured"], thresholds)
        cap_label = str(capacity) if capacity > 0 else "&lt;1"

        inp = sd["input_len"]
        out = sd["output_len"]
        in_k = f"{inp // 1024}K" if inp >= 1024 else str(inp)
        out_k = f"{out // 1024}K" if out >= 1024 else str(out)

        cls = card_colors[i % len(card_colors)]
        friendly = _friendly_name(sid)
        cap_cards.append(
            f'<div class="metric-card {cls}">'
            f'<div class="label">{friendly} Capacity</div>'
            f'<div class="value">{cap_label} <span class="unit">users</span></div>'
            f'<div class="detail">{in_k} in / {out_k} out</div></div>'
        )

        # Generate capacity chart image
        img_cap = chart_capacity(sid, sd, thresholds)

        max_tested = max(m["batch_size"] for m in sd["measured"])
        if capacity > 0:
            cap_explainer = (
                f"This scenario supports up to <strong>{capacity} concurrent users</strong> "
                f"within quality thresholds. Beyond that, ITL or TTFT exceeds "
                f"acceptable limits. The system was tested up to {max_tested} concurrent requests."
            )
        else:
            cap_explainer = (
                "Even at minimum concurrency, this scenario does not meet the quality "
                "thresholds. Consider reducing context length or upgrading hardware for this workload."
            )

        cap_charts.append(
            f'<div class="chart-grid" style="margin-top: 24px;">'
            f'<div class="chart-card full-width">'
            f"<h3>{friendly}</h3>"
            f'<div class="chart-desc">Context length: {inp:,} input + {out:,} output tokens.</div>'
            f'<img src="{img_cap}" style="width:100%;border-radius:8px;">'
            f'<div class="threshold-legend">'
            f'<div class="threshold-item"><div class="threshold-dot" style="background: #10b981;"></div> ITL (left axis)</div>'
            f'<div class="threshold-item"><div class="threshold-dot" style="background: #f59e0b;"></div> TTFT (right axis)</div>'
            f'<div class="threshold-item"><div class="threshold-dot" style="background: #eab308;"></div> Quality thresholds</div>'
            f"</div>"
            f'<div class="chart-explainer">{cap_explainer}</div>'
            f"</div></div>"
        )

    capacity_cards_html = "\n    ".join(cap_cards)
    capacity_charts_html = "\n    ".join(cap_charts)

    # ── Template substitution ───────────────────────────────────────────
    replacements = {
        "{{PAGE_TITLE}}": f"{model_name} Inference Benchmark",
        "{{MODEL_NAME}}": model_name,
        "{{GPU_NAME}}": gpu_name,
        "{{SPECS_HTML}}": specs_html,
        "{{KEY_METRICS_HTML}}": key_metrics_html,
        "{{IMG_THROUGHPUT_RANGE}}": img_throughput_range,
        "{{EXPLAINER_THROUGHPUT_RANGE}}": explainer_throughput,
        "{{IMG_TTFT_RANGE}}": img_ttft_range,
        "{{EXPLAINER_TTFT_RANGE}}": explainer_ttft,
        "{{IMG_SYSTEM_THROUGHPUT}}": img_system_throughput,
        "{{EXPLAINER_SYSTEM_THROUGHPUT}}": explainer_sys,
        "{{IMG_ITL}}": img_itl,
        "{{EXPLAINER_ITL}}": explainer_itl,
        "{{IMG_DECODE_SPEED}}": img_decode_speed,
        "{{EXPLAINER_DECODE_SPEED}}": explainer_decode,
        "{{IMG_SCALING_EFFICIENCY}}": img_scaling_efficiency,
        "{{EXPLAINER_SCALING_EFFICIENCY}}": explainer_eff,
        "{{CAPTION_THROUGHPUT_RANGE}}": caption_throughput,
        "{{CAPTION_TTFT_RANGE}}": caption_ttft,
        "{{CAPTION_SYSTEM_THROUGHPUT}}": caption_sys,
        "{{CAPTION_ITL}}": caption_itl,
        "{{CAPTION_DECODE_SPEED}}": caption_decode,
        "{{CAPTION_SCALING_EFFICIENCY}}": caption_eff,
        "{{SUMMARY_TABLE_HTML}}": summary_table_html,
        "{{NARRATIVE_SYSTEM_THROUGHPUT}}": narrative_sys,
        "{{CAPACITY_CARDS_HTML}}": capacity_cards_html,
        "{{CAPACITY_CHARTS_HTML}}": capacity_charts_html,
        "{{TABLE_BODY_HTML}}": table_body_html,
        "{{DATE_STR}}": datetime.now().strftime("%B %d, %Y"),
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  HTML report: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate benchmark report from latency results"
    )
    ap.add_argument(
        "input", help="Input directory (containing summary.csv) or CSV file path"
    )
    ap.add_argument("--out-dir", default="report", help="Output directory for report")
    args = ap.parse_args()

    # Resolve input path — prefer report_data.json over CSV when available
    os.makedirs(args.out_dir, exist_ok=True)
    json_loaded = False

    if os.path.isdir(args.input):
        json_path = os.path.join(args.input, "report_data.json")
        if os.path.exists(json_path):
            df, model_name, gpu_name = load_report_json(json_path)
            json_loaded = True
        else:
            csv_path = os.path.join(args.input, "summary.csv")
            if not os.path.exists(csv_path):
                csvs = [f for f in os.listdir(args.input) if f.endswith(".csv")]
                if csvs:
                    csv_path = os.path.join(args.input, csvs[0])
                else:
                    print(
                        f"ERROR: No CSV or report_data.json found in {args.input}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
    elif args.input.endswith(".json"):
        df, model_name, gpu_name = load_report_json(args.input)
        json_loaded = True
    else:
        csv_path = args.input

    if not json_loaded:
        df = load_csv(csv_path)
        if df.empty:
            print("ERROR: No valid data rows found in CSV", file=sys.stderr)
            sys.exit(1)

        # Detect model and GPU names
        model_name = "Unknown Model"
        gpu_name = "Unknown GPU"

        input_dir = os.path.dirname(csv_path)
        if os.path.basename(input_dir) != "":
            model_name = os.path.basename(input_dir)

        # Try to read GPU info from raw JSON files
        raw_dir = os.path.join(input_dir, "raw")
        if os.path.isdir(raw_dir):
            for f in os.listdir(raw_dir):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(raw_dir, f)) as jf:
                            jdata = json.load(jf)
                        if "gpu" in jdata:
                            gpu_name = jdata["gpu"]
                            break
                    except Exception:
                        pass

        # Try nvidia-smi
        if gpu_name == "Unknown GPU":
            try:
                import subprocess

                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().split("\n")[0].strip()
            except Exception:
                pass

    sd = build_scenario_data(df)
    scenarios = _sorted_scenarios(sd)
    print(f"Model: {model_name}")
    print(f"GPU:   {gpu_name}")
    print(f"Scenarios: {scenarios}")
    print(f"Generating report to {args.out_dir}/")
    print()

    # Generate charts
    fig_throughput_tradeoff(df, model_name, args.out_dir)
    print("  [1/5] Throughput tradeoff")

    fig_system_throughput(df, model_name, args.out_dir)
    print("  [2/5] System throughput")

    fig_per_user_speed(df, model_name, args.out_dir)
    print("  [3/5] Per-user speed")

    fig_scaling_efficiency(df, model_name, args.out_dir)
    print("  [4/5] Scaling efficiency")

    fig_latency(df, model_name, args.out_dir)
    print("  [5/5] Latency breakdown")

    # Generate HTML report (reuse scenario data built earlier for logging)
    generate_html_report(df, model_name, gpu_name, sd, args.out_dir)

    print()
    print(f"Done! Report in {args.out_dir}/")


if __name__ == "__main__":
    main()
