#!/usr/bin/env python3
"""
Convert simple_benchmarking output (benchmark_summary.json) into the
report_data.json format expected by generate_report.py.

Usage:
    python convert_results.py results/benchmark_summary.json -o results/report_data.json
    python convert_results.py results/benchmark_summary.json --provider "Together AI" --gpu "8xH100"
    python convert_results.py results/benchmark_summary.json --input-len 1024 --output-len 256

The converter maps each request_rate in the benchmark to a "batch_size" entry
in the report format, and derives prefill/decode metrics from client-side
TTFT and ITL measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not math.isfinite(b):
        return default
    return a / b


def _convert_results_to_measured(
    results: list,
    input_len: int,
    num_requests: int,
) -> list:
    """Convert a list of per-rate result dicts into report 'measured' entries."""
    measured = []
    for r in results:
        rate = r["request_rate"]
        cm = r.get("client_metrics", {})
        sm = r.get("server_metrics")

        ttft_s = cm.get("ttft_ms", {}).get("avg", 0) / 1000.0
        e2e_s = cm.get("e2e_ms", {}).get("avg", 0) / 1000.0
        itl_ms = cm.get("itl_ms", {}).get("avg", 0)

        decode_tps_per_user = _safe_div(1000.0, itl_ms) if itl_ms > 0 else 0
        prefill_tps_per_user = _safe_div(input_len, ttft_s) if ttft_s > 0 else 0
        decode_time = e2e_s - ttft_s if e2e_s > ttft_s else 0

        if rate == float('inf') or rate > 1e6:
            batch_size = r.get("num_successful", num_requests)
        else:
            batch_size = int(rate) if rate >= 1 else 1

        if sm and sm.get("decode_throughput", {}).get("avg", 0) > 0:
            decode_tps_total = sm["decode_throughput"]["avg"]
            prefill_tps_total = sm.get("prefill_throughput", {}).get("avg", 0)
        else:
            effective_conc = min(batch_size, r.get("num_successful", batch_size))
            decode_tps_total = decode_tps_per_user * effective_conc
            prefill_tps_total = prefill_tps_per_user * effective_conc

        measured.append({
            "batch_size": batch_size,
            "prefill_tps_per_user": round(prefill_tps_per_user, 1),
            "decode_tps_per_user": round(decode_tps_per_user, 1),
            "prefill_tps_total": round(prefill_tps_total, 0),
            "decode_tps_total": round(decode_tps_total, 0),
            "ttft": round(ttft_s, 3),
            "decode_time": round(decode_time, 3),
            "e2e": round(e2e_s, 3),
        })

    measured.sort(key=lambda m: m["batch_size"])
    return measured


def _make_scenario_name(input_len: int, output_len: int) -> str:
    in_label = f"{input_len // 1024}k" if input_len >= 1024 else str(input_len)
    return f"{in_label}-in-{output_len}-out"


def convert(
    summary: dict,
    *,
    provider_name: Optional[str] = None,
    gpu_name: Optional[str] = None,
    input_len_override: Optional[int] = None,
    output_len_override: Optional[int] = None,
    scenario_name: Optional[str] = None,
) -> dict:
    """Convert a benchmark_summary.json dict into report_data.json format.

    Supports both single-scenario (flat results list) and multi-scenario
    (scenarios list with per-input-length results) formats.
    """
    model_name = summary.get("model", "Unknown Model")
    max_tokens = summary.get("max_tokens", 256)
    num_requests = summary.get("num_requests", 100)

    # Multi-scenario format: summary has "scenarios" key
    scenarios_list = summary.get("scenarios")
    if scenarios_list and isinstance(scenarios_list, list):
        all_scenarios = {}
        for sc in scenarios_list:
            sc_input_len = sc.get("input_length", 512)
            sc_results = sc.get("results", [])
            if not sc_results:
                continue

            # Estimate output length from data
            avg_completions = [r["avg_completion_tokens"] for r in sc_results if r.get("avg_completion_tokens", 0) > 0]
            output_len = int(round(sum(avg_completions) / len(avg_completions))) if avg_completions else max_tokens

            sc_name = _make_scenario_name(sc_input_len, output_len)
            measured = _convert_results_to_measured(sc_results, sc_input_len, num_requests)

            all_scenarios[sc_name] = {
                "input_len": sc_input_len,
                "output_len": output_len,
                "measured": measured,
            }

        if not all_scenarios:
            raise ValueError("No valid scenarios found in benchmark summary")

        return {
            "model_name": model_name,
            "gpu_name": gpu_name or provider_name or "Unknown Provider",
            "scenarios": all_scenarios,
        }

    # Single-scenario format: flat results list (backward compatible)
    results = summary.get("results", [])
    if not results:
        raise ValueError("No results found in benchmark summary")

    if input_len_override:
        input_len = input_len_override
    else:
        avg_prompts = [r["avg_prompt_tokens"] for r in results if r.get("avg_prompt_tokens", 0) > 0]
        input_len = int(round(sum(avg_prompts) / len(avg_prompts))) if avg_prompts else 512

    if output_len_override:
        output_len = output_len_override
    else:
        avg_completions = [r["avg_completion_tokens"] for r in results if r.get("avg_completion_tokens", 0) > 0]
        output_len = int(round(sum(avg_completions) / len(avg_completions))) if avg_completions else max_tokens

    if not scenario_name:
        scenario_name = _make_scenario_name(input_len, output_len)

    measured = _convert_results_to_measured(results, input_len, num_requests)

    return {
        "model_name": model_name,
        "gpu_name": gpu_name or provider_name or "Unknown Provider",
        "scenarios": {
            scenario_name: {
                "input_len": input_len,
                "output_len": output_len,
                "measured": measured,
            }
        },
    }


def merge_reports(reports: list[dict]) -> dict:
    """Merge multiple report_data.json dicts (e.g., from different ISL/OSL runs)
    into a single report with multiple scenarios."""
    if not reports:
        raise ValueError("No reports to merge")

    merged = {
        "model_name": reports[0]["model_name"],
        "gpu_name": reports[0]["gpu_name"],
        "scenarios": {},
    }
    for r in reports:
        merged["scenarios"].update(r["scenarios"])

    return merged


def main():
    ap = argparse.ArgumentParser(description="Convert benchmark results to report format")
    ap.add_argument("input", help="Path to benchmark_summary.json")
    ap.add_argument("-o", "--output", default=None, help="Output path (default: report_data.json in same dir)")
    ap.add_argument("--provider", default=None, help="Provider name (shown as GPU in report)")
    ap.add_argument("--gpu", default=None, help="GPU/hardware name")
    ap.add_argument("--input-len", type=int, default=None, help="Override input sequence length")
    ap.add_argument("--output-len", type=int, default=None, help="Override output sequence length")
    ap.add_argument("--scenario", default=None, help="Override scenario name")
    args = ap.parse_args()

    with open(args.input) as f:
        summary = json.load(f)

    report = convert(
        summary,
        provider_name=args.provider,
        gpu_name=args.gpu,
        input_len_override=args.input_len,
        output_len_override=args.output_len,
        scenario_name=args.scenario,
    )

    output_path = args.output
    if not output_path:
        import os
        output_path = os.path.join(os.path.dirname(args.input) or ".", "report_data.json")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Converted: {args.input} -> {output_path}")
    print(f"  Model: {report['model_name']}")
    print(f"  Provider: {report['gpu_name']}")
    scenarios = list(report['scenarios'].keys())
    print(f"  Scenarios: {scenarios}")
    for s in scenarios:
        n = len(report['scenarios'][s]['measured'])
        print(f"    {s}: {n} data points")


if __name__ == "__main__":
    main()
