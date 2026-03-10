"""Shared fixtures for the benchmark test suite."""

import pytest
from client_metrics import RequestInput, RequestOutput


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_request_input(request_id=0, prompt="Hello", prompt_length=5, image_url=None):
    return RequestInput(
        prompt=prompt,
        image_url=image_url,
        request_id=request_id,
        prompt_length=prompt_length,
        image_size_bytes=0,
    )


def make_request_output(
    request_id=0,
    success=True,
    ttft=0.1,
    e2e_latency=1.0,
    output_tokens=50,
    prompt_tokens=100,
    completion_tokens=50,
    itl=None,
    error="",
):
    return RequestOutput(
        request_id=request_id,
        success=success,
        error=error,
        generated_text="x " * output_tokens,
        ttft=ttft,
        e2e_latency=e2e_latency,
        output_tokens=output_tokens,
        itl=itl or [0.02] * max(output_tokens - 1, 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Standard benchmark result bundle (matches main.py output format)
# ---------------------------------------------------------------------------

def make_result_bundle(
    request_rate=1.0,
    num_successful=10,
    avg_prompt_tokens=1024,
    avg_completion_tokens=256,
    ttft_avg_ms=150.0,
    e2e_avg_ms=30000.0,
    itl_avg_ms=25.0,
    toks_avg=40.0,
):
    return {
        "request_rate": request_rate,
        "num_total": num_successful,
        "num_successful": num_successful,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "client_metrics": {
            "ttft_ms": {"p50": ttft_avg_ms, "p95": ttft_avg_ms * 1.5, "p99": ttft_avg_ms * 2, "avg": ttft_avg_ms},
            "e2e_ms": {"p50": e2e_avg_ms, "p95": e2e_avg_ms * 1.2, "p99": e2e_avg_ms * 1.5, "avg": e2e_avg_ms},
            "itl_ms": {"p50": itl_avg_ms, "p95": itl_avg_ms * 1.5, "p99": itl_avg_ms * 2, "avg": itl_avg_ms},
            "toks": {"p50": toks_avg, "p95": toks_avg * 0.8, "p99": toks_avg * 0.6, "avg": toks_avg},
        },
        "server_metrics": None,
    }


# ---------------------------------------------------------------------------
# Scenario data (matches report_data.json structure)
# ---------------------------------------------------------------------------

def make_scenario_measured(batch_sizes=None):
    """Create measured data points for a scenario."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    measured = []
    for bs in batch_sizes:
        decode_per_user = max(5.0, 44.0 - bs * 1.5)
        measured.append({
            "batch_size": bs,
            "prefill_tps_per_user": max(50.0, 6000.0 / bs),
            "decode_tps_per_user": decode_per_user,
            "prefill_tps_total": max(50.0, 6000.0 / bs) * bs,
            "decode_tps_total": decode_per_user * bs * 0.95,
            "ttft": 0.15 + bs * 2.5,
            "decode_time": 30.0 + bs * 1.5,
            "e2e": 30.15 + bs * 4.0,
        })
    return measured


def make_scenarios_data():
    """Create a multi-scenario data dict."""
    return {
        "1k-in-1463-out": {
            "input_len": 1024,
            "output_len": 1463,
            "measured": make_scenario_measured(),
        },
        "4k-in-1302-out": {
            "input_len": 4096,
            "output_len": 1302,
            "measured": make_scenario_measured(),
        },
    }


def make_dual_report_data():
    """Create a full dual-mode report_data.json structure."""
    return {
        "model_name": "test/model",
        "gpu_name": "8xH200",
        "dual_mode": True,
        "peak_performance": {
            "x_label": "Concurrent Users",
            "scenarios": make_scenarios_data(),
        },
        "queue_depth": {
            "x_label": "Queue Depth (Simultaneous Requests)",
            "scenarios": make_scenarios_data(),
        },
    }


def make_single_report_data():
    """Create a single-mode report_data.json structure."""
    return {
        "model_name": "test/model",
        "gpu_name": "8xH200",
        "scenarios": make_scenarios_data(),
    }


# ---------------------------------------------------------------------------
# Benchmark summary (matches benchmark_summary.json from main.py)
# ---------------------------------------------------------------------------

def make_benchmark_summary(
    rates=None,
    input_lengths=None,
    num_requests=32,
    dual_mode=False,
):
    if rates is None:
        rates = [1.0, 2.0, 4.0]
    if input_lengths is None:
        input_lengths = [1024, 4096]

    def _make_scenario_results(x_values):
        scenarios = []
        for il in input_lengths:
            results = [
                make_result_bundle(
                    request_rate=r,
                    avg_prompt_tokens=il,
                    avg_completion_tokens=256,
                )
                for r in x_values
            ]
            scenarios.append({
                "input_length": il,
                "rates": x_values,
                "results": results,
            })
        return scenarios

    base = {
        "url": "http://localhost:3000/v1/chat/completions",
        "model": "test/model",
        "model_type": "text",
        "num_requests": num_requests,
        "max_tokens": 4096,
        "input_lengths": input_lengths,
    }

    if dual_mode:
        concs = [1, 2, 4, 8]
        queue_n = [1, 2, 3, 4, 5]  # Adaptive discovery results
        base["dual_mode"] = True
        base["peak_performance"] = {
            "x_label": "Concurrent Users",
            "x_values": concs,
            "scenarios": _make_scenario_results([float(c) for c in concs]),
        }
        base["queue_depth"] = {
            "x_label": "Queue Depth (Simultaneous Requests)",
            "x_values": [float(n) for n in queue_n],
            "scenarios": _make_scenario_results([float(n) for n in queue_n]),
        }
    else:
        base["rates"] = rates
        base["scenarios"] = _make_scenario_results(rates)

    return base


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def request_outputs():
    """A batch of successful request outputs."""
    return [
        make_request_output(request_id=i, ttft=0.1 + i * 0.01, e2e_latency=1.0 + i * 0.1)
        for i in range(10)
    ]


@pytest.fixture
def request_outputs_with_failures():
    """A batch including some failures."""
    outputs = [
        make_request_output(request_id=i, ttft=0.1 + i * 0.01, e2e_latency=1.0 + i * 0.1)
        for i in range(8)
    ]
    outputs.append(make_request_output(request_id=8, success=False, error="timeout", ttft=0, e2e_latency=0))
    outputs.append(make_request_output(request_id=9, success=False, error="500", ttft=0, e2e_latency=0))
    return outputs


@pytest.fixture
def scenarios_data():
    return make_scenarios_data()


@pytest.fixture
def dual_report_data():
    return make_dual_report_data()


@pytest.fixture
def single_report_data():
    return make_single_report_data()


@pytest.fixture
def dual_summary():
    return make_benchmark_summary(dual_mode=True)


@pytest.fixture
def single_summary():
    return make_benchmark_summary(dual_mode=False)
