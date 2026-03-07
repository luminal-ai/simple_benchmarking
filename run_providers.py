#!/usr/bin/env python3
"""
Run benchmarks across multiple providers and generate comparison reports.

Usage:
    python run_providers.py                           # uses providers.yaml
    python run_providers.py --config my_providers.yaml
    python run_providers.py --output-dir ./results
    python run_providers.py --providers local-vllm together-ai  # subset only

Each provider gets:
  results/<provider>/benchmark_summary.json  - raw benchmark data
  results/<provider>/report_data.json        - converted for report generation
  results/<provider>/report/index.html       - full HTML report with charts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import yaml

from convert_results import convert, merge_reports


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_benchmark(provider: dict, defaults: dict, output_dir: str) -> str:
    """Run simple_benchmarking for a single provider. Returns path to benchmark_summary.json."""
    name = provider["name"]
    provider_dir = os.path.join(output_dir, name)
    os.makedirs(provider_dir, exist_ok=True)

    url = provider["url"]
    model = provider["model"]
    num_requests = provider.get("num_requests", defaults.get("num_requests", 100))
    max_tokens = provider.get("max_tokens", defaults.get("max_tokens", 256))
    model_type = provider.get("model_type", defaults.get("model_type", "text"))
    rates = provider.get("rates", defaults.get("rates", "1,10,100"))
    wait = provider.get("inter_run_wait_seconds", defaults.get("inter_run_wait_seconds", 30))
    disable_server = provider.get("disable_server_metrics", False)

    # Resolve API key from environment
    api_key = None
    api_key_env = provider.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(f"  WARNING: {api_key_env} not set, skipping {name}")
            return ""

    cmd = [
        sys.executable, "main.py",
        "--url", url,
        "--model", model,
        "--model-type", model_type,
        "--num-requests", str(num_requests),
        "--max-tokens", str(max_tokens),
        "--run-multi",
        "--rates", rates,
        "--output-dir", provider_dir,
        "--inter-run-wait-seconds", str(wait),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if disable_server:
        cmd.append("--disable-server-metrics")

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Rates: {rates}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  ERROR: Benchmark failed for {name} (exit code {result.returncode})")
        return ""

    summary_path = os.path.join(provider_dir, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        print(f"  ERROR: No benchmark_summary.json produced for {name}")
        return ""

    return summary_path


def convert_and_report(provider: dict, summary_path: str, output_dir: str):
    """Convert benchmark results and generate HTML report for a provider."""
    name = provider["name"]
    provider_dir = os.path.join(output_dir, name)

    with open(summary_path) as f:
        summary = json.load(f)

    report_data = convert(
        summary,
        provider_name=name,
        gpu_name=provider.get("gpu", name),
    )

    report_json = os.path.join(provider_dir, "report_data.json")
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Converted: {report_json}")

    # Generate HTML report
    report_dir = os.path.join(provider_dir, "report")
    cmd = [
        sys.executable, "generate_report.py",
        report_json,
        "--out-dir", report_dir,
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode == 0:
        print(f"  Report: {os.path.join(report_dir, 'index.html')}")
    else:
        print(f"  WARNING: Report generation failed for {name}")


def main():
    ap = argparse.ArgumentParser(description="Run benchmarks across multiple providers")
    ap.add_argument("--config", default="providers.yaml", help="Provider config file")
    ap.add_argument("--output-dir", default="results", help="Output directory")
    ap.add_argument("--providers", nargs="*", help="Run only these providers (by name)")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="Skip benchmarking, only convert + report from existing data")
    args = ap.parse_args()

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    providers = config.get("providers", [])

    if not providers:
        print("No providers configured. Edit providers.yaml and uncomment at least one provider.")
        sys.exit(1)

    # Filter to requested providers
    if args.providers:
        providers = [p for p in providers if p["name"] in args.providers]
        if not providers:
            print(f"No matching providers found for: {args.providers}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Providers to benchmark: {[p['name'] for p in providers]}")
    print(f"Output directory: {args.output_dir}")

    summary_paths = {}

    for provider in providers:
        name = provider["name"]

        if args.skip_benchmark:
            # Look for existing benchmark data
            sp = os.path.join(args.output_dir, name, "benchmark_summary.json")
            if os.path.exists(sp):
                summary_paths[name] = sp
                print(f"  Using existing data for {name}: {sp}")
            else:
                print(f"  No existing data for {name}, skipping")
        else:
            sp = run_benchmark(provider, defaults, args.output_dir)
            if sp:
                summary_paths[name] = sp

    # Convert and generate reports
    print(f"\n{'='*60}")
    print("  Generating reports")
    print(f"{'='*60}")

    for provider in providers:
        name = provider["name"]
        if name in summary_paths:
            convert_and_report(provider, summary_paths[name], args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")
    for provider in providers:
        name = provider["name"]
        report_path = os.path.join(args.output_dir, name, "report", "index.html")
        if os.path.exists(report_path):
            print(f"  {name}: {report_path}")


if __name__ == "__main__":
    main()
