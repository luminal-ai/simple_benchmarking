#!/usr/bin/env python3
"""
DeepSeek-VL2 saturation estimator (SGLang default; vLLM optional)

- Polls /metrics (Prometheus text) and samples:
  * num_running_reqs
  * num_queue_reqs
  * optional token/cache usage

- Recommends a queue length that keeps the engine saturated:
  p95(queue length) during periods where num_running_reqs >= util_threshold*max_running,
  with a small +1 headroom.

Examples:
  python saturation_finder.py --base-url http://127.0.0.1:30000 --model deepseek-vl2
  python saturation_finder.py --backend vllm --base-url http://127.0.0.1:8000 --model deepseek-vl2

Requires: pip install aiohttp
"""

import argparse
import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# ----------------------------
# Data classes
# ----------------------------

@dataclass
class QueueMetricsSample:
    timestamp: float
    num_queue_reqs: int
    avg_queue_latency: float
    num_running_reqs: int
    token_usage: float  # percentage if available, else 0.0

@dataclass
class ServerMetrics:
    num_requests: int
    num_aborted_requests: int
    prompt_tokens: int
    generation_tokens: int
    cached_tokens: int
    prefill_throughput_buckets: Dict[float, float] = field(default_factory=dict)
    decode_throughput_buckets: Dict[float, float] = field(default_factory=dict)
    prefill_throughput_sum: float = 0.0
    decode_throughput_sum: float = 0.0
    avg_batch_size: float = 0.0
    num_queue_reqs: int = 0
    avg_request_queue_latency: float = 0.0
    queue_length_buckets: Dict[float, float] = field(default_factory=dict)
    queue_latency_buckets: Dict[float, float] = field(default_factory=dict)
    queue_length_sum: float = 0.0
    queue_latency_sum: float = 0.0

# ----------------------------
# Prometheus parsing utilities
# ----------------------------

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="(.*?)"')

def _parse_labels(s: str) -> Dict[str, str]:
    return dict(_LABEL_RE.findall(s))

def parse_prometheus_metrics(
    text: str
) -> Tuple[Dict[str, float], Dict[str, Dict[float, float]], Dict[str, Dict[tuple, float]]]:
    """
    Parse Prometheus text into:
      - metrics: flat {name -> value} (labelled metrics are summed by name)
      - histograms: {base_metric_name -> {le -> cumulative_count}}
      - labeled_metrics: {name -> {tuple(sorted(label_items)) -> value}}
    Robust to +Inf and label order.
    """
    metrics: Dict[str, float] = {}
    histograms: Dict[str, Dict[float, float]] = {}
    labeled_metrics: Dict[str, Dict[tuple, float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Histogram bucket of form: <metric>_bucket{... le="<bound>"} <count>
        if "_bucket{" in line:
            m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{(.*)\}\s+([\d.e+-]+)$', line)
            if m:
                base = m.group(1)
                labels_str = m.group(2)
                count = float(m.group(3))
                labels = _parse_labels(labels_str)
                le_s = labels.get("le", None)
                if le_s is not None:
                    if le_s in ["+Inf", r"\+Inf"]:
                        le = float("inf")
                    else:
                        try:
                            le = float(le_s)
                        except ValueError:
                            continue
                    histograms.setdefault(base, {})[le] = count
                    continue

        # Labeled metric: <metric>{k="v",...} <value>
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{(.*)\}\s+([\d.e+-]+)$', line)
        if m:
            name = m.group(1)
            labels_str = m.group(2)
            val = float(m.group(3))
            labels = _parse_labels(labels_str)
            label_key = tuple(sorted(labels.items()))
            labeled_metrics.setdefault(name, {})[label_key] = val
            metrics[name] = metrics.get(name, 0.0) + val
            continue

        # Plain metric: <metric> <value>
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.e+-]+)$', line)
        if m:
            name = m.group(1)
            val = float(m.group(2))
            metrics[name] = val

    return metrics, histograms, labeled_metrics

def _pick(parsed: Dict[str, float], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in parsed:
            return float(parsed[k])
    return default

def _pick_percent(parsed: Dict[str, float], *keys: str) -> float:
    # Returns a percentage 0..100 if possible; otherwise 0.0
    for k in keys:
        if k in parsed:
            v = float(parsed[k])
            if 0.0 <= v <= 1.0001:  # ratio -> percent
                return v * 100.0
            return v
    return 0.0

# ----------------------------
# Polling sampler
# ----------------------------

async def poll_queue_metrics(
    base_url: str,
    start_time: float,
    samples: List[QueueMetricsSample],
    poll_interval: float = 1.0,
    stop_event: Optional[asyncio.Event] = None,
    backend: str = "sglang",
    metrics_path: str = "/metrics",
    bearer_token: Optional[str] = None,
):
    assert stop_event is not None, "stop_event must be provided"

    base = base_url.rstrip("/")
    url = f"{base}{metrics_path if metrics_path.startswith('/') else '/' + metrics_path}"

    timeout = aiohttp.ClientTimeout(total=max(2.5, poll_interval * 2))
    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
        polls = 0
        while not stop_event.is_set():
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        parsed, _, _ = parse_prometheus_metrics(text)

                        if backend == "sglang":
                        num_wait = int(_pick(parsed, "sglang:num_queue_reqs", default=0))
                        num_run  = int(_pick(parsed, "sglang:num_running_reqs", default=0))
                        token_pct = _pick_percent(parsed, "sglang:token_usage")
                    
                        else:  # vllm
                            num_wait = int(_pick(parsed, "vllm:num_requests_waiting", default=0))
                            num_run  = int(_pick(parsed, "vllm:num_requests_running", default=0))
                            token_pct = _pick_percent(parsed, "vllm:gpu_cache_usage_perc")
                        

                        sample = QueueMetricsSample(
                            timestamp=time.perf_counter() - start_time,
                            num_queue_reqs=num_wait,
                            avg_queue_latency=0.0,
                            num_running_reqs=num_run,
                            token_usage=token_pct,
                        )

                        samples.append(sample)
                        polls += 1
                        if polls % 5 == 0:
                            print(f"[saturation_finder] polled /metrics ok | samples={len(samples)} "
                                  f"| running={sample.num_running_reqs} waiting={sample.num_queue_reqs} token%={sample.token_usage:.1f}")
                    else:
                        print(f"[saturation_finder] /metrics HTTP {resp.status} at {url}")
            except Exception as e:
                print(f"[saturation_finder] Warning: Failed to poll {url}: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass

# ----------------------------
# Saturation Finder
# ----------------------------

class SaturationFinder:
    """
    Estimate the queue length needed to achieve/maintain near-full engine saturation.

    Heuristic:
      1) Observe max(num_running_reqs) over the sampling window.
      2) target_running = ceil(util_threshold * max_running).
      3) During times where running >= target, compute p95 of queue length.
      4) Recommend that p95 + 1 as steady-state queue length (>= 1).
    """

    def __init__(
        self,
        base_url: str,
        backend: str = "sglang",
        duration_s: float = 60.0,
        poll_interval_s: float = 1.0,
        util_threshold: float = 0.95,
        min_samples_for_confident: int = 10,
        model_name: str = "deepseek-vl2",
        metrics_path: str = "/metrics",
        bearer_token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.duration_s = duration_s
        self.poll_interval_s = poll_interval_s
        self.util_threshold = util_threshold
        self.min_samples_for_confident = min_samples_for_confident
        self.model_name = model_name
        self.metrics_path = metrics_path
        self.bearer_token = bearer_token
        self.samples: List[QueueMetricsSample] = []

    async def run(self) -> Dict[str, Any]:
        start = time.perf_counter()
        stop_event = asyncio.Event()

        async def _stopper():
            await asyncio.sleep(self.duration_s)
            stop_event.set()

        poll_task = asyncio.create_task(
            poll_queue_metrics(
                base_url=self.base_url,
                start_time=start,
                samples=self.samples,
                poll_interval=self.poll_interval_s,
                stop_event=stop_event,
                backend=self.backend,
                metrics_path=self.metrics_path,
                bearer_token=self.bearer_token,
            )
        )
        stopper_task = asyncio.create_task(_stopper())

        await asyncio.gather(poll_task, stopper_task)
        return self._analyze()

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        vs = sorted(values)
        if len(vs) == 1:
            return float(vs[0])
        k = (len(vs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vs[int(k)])
        return float(vs[f] + (k - f) * (vs[c] - vs[f]))

    def _analyze(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "backend": self.backend,
                "model": self.model_name,
                "recommended_queue_length": 0,
                "confidence": "low",
                "reason": "No samples collected (is /metrics reachable on the same port?).",
                "stats": {},
            }

        max_running = max(s.num_running_reqs for s in self.samples)
        target_running = max(1, math.ceil(self.util_threshold * max_running))

        saturated_samples = [s for s in self.samples if s.num_running_reqs >= target_running]
        overall_q = [s.num_queue_reqs for s in self.samples]
        sat_q = [s.num_queue_reqs for s in saturated_samples]

        overall_p50 = self._percentile(overall_q, 50)
        overall_p95 = self._percentile(overall_q, 95)
        sat_p50 = self._percentile(sat_q, 50) if sat_q else 0.0
        sat_p95 = self._percentile(sat_q, 95) if sat_q else 0.0

        if len(saturated_samples) >= self.min_samples_for_confident:
            rec = max(1, int(round(sat_p95)) + 1)
            confidence = "high"
            basis = "saturated-window p95"
            used_p95 = sat_p95
        else:
            rec = max(1, int(round(overall_p95)) + 1)
            confidence = "medium" if len(self.samples) >= self.min_samples_for_confident else "low"
            basis = "overall p95"
            used_p95 = overall_p95

        token_usages = [s.token_usage for s in self.samples if isinstance(s.token_usage, (int, float))]
        token_usage_p50 = self._percentile(token_usages, 50) if token_usages else None
        token_usage_p95 = self._percentile(token_usages, 95) if token_usages else None

        return {
            "backend": self.backend,
            "model": self.model_name,
            "recommended_queue_length": rec,
            "confidence": confidence,
            "basis": basis,
            "stats": {
                "max_running_reqs": max_running,
                "target_running_reqs": target_running,
                "samples_total": len(self.samples),
                "samples_saturated": len(saturated_samples),
                "queue_len": {
                    "overall_p50": overall_p50,
                    "overall_p95": overall_p95,
                    "saturated_p50": sat_p50,
                    "saturated_p95": sat_p95,
                    "used_p95": used_p95,
                },
                "token_usage_percent": {
                    "p50": token_usage_p50,
                    "p95": token_usage_p95,
                },
            },
        }

# ----------------------------
# CLI
# ----------------------------

async def main_async(args) -> None:
    finder = SaturationFinder(
        base_url=args.base_url,
        backend=args.backend,
        duration_s=args.duration,
        poll_interval_s=args.poll_interval,
        util_threshold=args.util_threshold,
        min_samples_for_confident=args.min_samples,
        model_name=args.model,
        metrics_path=args.metrics_path,
        bearer_token=args.bearer_token,
    )
    result = await finder.run()

    print("\n=== DeepSeek-VL2 Saturation Estimate ===")
    print(f"Backend: {result['backend']} | Model: {result['model']}")
    print(
        f"Recommended queue length: {result['recommended_queue_length']} "
        f"(confidence: {result['confidence']}, basis: {result.get('basis')})"
    )
    stats = result["stats"]
    if stats:
        print(f"Max running reqs: {stats['max_running_reqs']} | Target running: {stats['target_running_reqs']}")
        print(f"Samples: total={stats['samples_total']} saturated={stats['samples_saturated']}")
        q = result["stats"]["queue_len"]
        print(f"Queue length p50/p95 overall: {q['overall_p50']:.2f}/{q['overall_p95']:.2f}")
        print(f"Queue length p50/p95 (saturated): {q['saturated_p50']:.2f}/{q['saturated_p95']:.2f}")
        tu = result["stats"].get("token_usage_percent")
        if tu and (tu['p50'] is not None):
            print(f"Token usage % p50/p95: {tu['p50']:.1f}% / {tu['p95']:.1f}%")
    print("========================================\n")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate queue length to keep DeepSeek-VL2 saturated on SGLang (default) or vLLM."
    )
    p.add_argument("--base-url", required=True, help="Base URL (e.g., http://127.0.0.1:30000 for SGLang)")
    p.add_argument("--backend", choices=["sglang", "vllm"], default="sglang", help="Backend type (default: sglang)")
    p.add_argument("--model", default="deepseek-vl2", help="Model name (labels only)")
    p.add_argument("--duration", type=float, default=60.0, help="Sampling duration in seconds (default: 60)")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds (default: 1)")
    p.add_argument("--util-threshold", type=float, default=0.95, help="Target running reqs fraction (default: 0.95)")
    p.add_argument("--metrics-path", default="/metrics", help="Path to metrics endpoint (default: /metrics)")
    p.add_argument("--bearer-token", default=None, help="Bearer token for /metrics if protected")
    p.add_argument("--min-samples", type=int, default=10, help="Samples needed for high confidence (default: 10)")
    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()

