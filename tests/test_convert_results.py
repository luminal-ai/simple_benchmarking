"""Tests for convert_results.py — the data transformation layer."""

import math
import pytest
from convert_results import (
    _safe_div,
    _make_scenario_name,
    _convert_results_to_measured,
    _convert_dual_mode,
    convert,
    merge_reports,
)
from tests.conftest import make_result_bundle, make_benchmark_summary


# ── _safe_div ─────────────────────────────────────────────────────────────

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 2) == 5.0

    def test_division_by_zero(self):
        assert _safe_div(10, 0) == 0.0

    def test_division_by_zero_custom_default(self):
        assert _safe_div(10, 0, default=-1.0) == -1.0

    def test_division_by_inf(self):
        assert _safe_div(10, float("inf")) == 0.0

    def test_division_by_nan(self):
        assert _safe_div(10, float("nan")) == 0.0

    def test_zero_numerator(self):
        assert _safe_div(0, 5) == 0.0


# ── _make_scenario_name ──────────────────────────────────────────────────

class TestMakeScenarioName:
    def test_1k_input(self):
        assert _make_scenario_name(1024, 256) == "1k-in-256-out"

    def test_4k_input(self):
        assert _make_scenario_name(4096, 1302) == "4k-in-1302-out"

    def test_32k_input(self):
        assert _make_scenario_name(32768, 1085) == "32k-in-1085-out"

    def test_sub_1k_input(self):
        assert _make_scenario_name(512, 100) == "512-in-100-out"

    def test_16k_input(self):
        assert _make_scenario_name(16384, 500) == "16k-in-500-out"


# ── _convert_results_to_measured ─────────────────────────────────────────

class TestConvertResultsToMeasured:
    def test_basic_conversion(self):
        results = [make_result_bundle(request_rate=1.0, avg_prompt_tokens=1024)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert len(measured) == 1
        m = measured[0]
        assert m["batch_size"] == 1
        assert "prefill_tps_per_user" in m
        assert "decode_tps_per_user" in m
        assert "ttft" in m
        assert "e2e" in m

    def test_multiple_rates(self):
        results = [
            make_result_bundle(request_rate=1.0),
            make_result_bundle(request_rate=4.0),
            make_result_bundle(request_rate=8.0),
        ]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert len(measured) == 3
        # Should be sorted by batch_size
        assert measured[0]["batch_size"] <= measured[1]["batch_size"] <= measured[2]["batch_size"]

    def test_inf_rate_uses_num_successful(self):
        results = [make_result_bundle(request_rate=float("inf"), num_successful=50)]
        measured = _convert_results_to_measured(results, 1024, 100)
        assert measured[0]["batch_size"] == 50

    def test_ttft_derived_from_client_metrics(self):
        results = [make_result_bundle(request_rate=1.0, ttft_avg_ms=200.0)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert measured[0]["ttft"] == pytest.approx(0.2, abs=0.001)

    def test_decode_tps_derived_from_itl(self):
        results = [make_result_bundle(request_rate=1.0, itl_avg_ms=25.0)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert measured[0]["decode_tps_per_user"] == pytest.approx(40.0, abs=0.1)

    def test_decode_time_is_e2e_minus_ttft(self):
        results = [make_result_bundle(request_rate=1.0, ttft_avg_ms=500.0, e2e_avg_ms=10500.0)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert measured[0]["decode_time"] == pytest.approx(10.0, abs=0.01)

    def test_zero_itl_gives_zero_decode_tps(self):
        results = [make_result_bundle(request_rate=1.0, itl_avg_ms=0)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert measured[0]["decode_tps_per_user"] == 0

    def test_zero_ttft_gives_zero_prefill_tps(self):
        results = [make_result_bundle(request_rate=1.0, ttft_avg_ms=0)]
        measured = _convert_results_to_measured(results, 1024, 32)
        assert measured[0]["prefill_tps_per_user"] == 0


# ── convert (single scenario) ───────────────────────────────────────────

class TestConvertSingleScenario:
    def test_flat_results(self):
        summary = {
            "model": "test-model",
            "max_tokens": 256,
            "num_requests": 32,
            "results": [
                make_result_bundle(request_rate=1.0, avg_prompt_tokens=1024, avg_completion_tokens=256),
                make_result_bundle(request_rate=4.0, avg_prompt_tokens=1024, avg_completion_tokens=256),
            ],
        }
        report = convert(summary, gpu_name="TestGPU")
        assert report["model_name"] == "test-model"
        assert report["gpu_name"] == "TestGPU"
        assert "scenarios" in report
        assert len(report["scenarios"]) == 1
        scenario = list(report["scenarios"].values())[0]
        assert len(scenario["measured"]) == 2

    def test_provider_name_fallback(self):
        summary = {
            "model": "m",
            "results": [make_result_bundle()],
        }
        report = convert(summary, provider_name="TogetherAI")
        assert report["gpu_name"] == "TogetherAI"

    def test_no_results_raises(self):
        with pytest.raises(ValueError, match="No results"):
            convert({"model": "m", "results": []})

    def test_input_len_override(self):
        summary = {
            "model": "m",
            "results": [make_result_bundle(avg_prompt_tokens=512)],
        }
        report = convert(summary, input_len_override=2048)
        scenario = list(report["scenarios"].values())[0]
        assert scenario["input_len"] == 2048

    def test_output_len_override(self):
        summary = {
            "model": "m",
            "results": [make_result_bundle(avg_completion_tokens=300)],
        }
        report = convert(summary, output_len_override=500)
        scenario = list(report["scenarios"].values())[0]
        assert scenario["output_len"] == 500

    def test_scenario_name_override(self):
        summary = {
            "model": "m",
            "results": [make_result_bundle()],
        }
        report = convert(summary, scenario_name="custom-scenario")
        assert "custom-scenario" in report["scenarios"]


# ── convert (multi-scenario) ─────────────────────────────────────────────

class TestConvertMultiScenario:
    def test_multi_scenario_format(self, single_summary):
        report = convert(single_summary, gpu_name="GPU")
        assert report["model_name"] == "test/model"
        assert len(report["scenarios"]) == 2
        for name, scenario in report["scenarios"].items():
            assert "input_len" in scenario
            assert "output_len" in scenario
            assert "measured" in scenario
            assert len(scenario["measured"]) == 3  # 3 rates

    def test_scenario_names_include_context_size(self, single_summary):
        report = convert(single_summary, gpu_name="GPU")
        names = list(report["scenarios"].keys())
        assert any("1k" in n for n in names)
        assert any("4k" in n for n in names)

    def test_empty_scenarios_raises(self):
        summary = {
            "model": "m",
            "scenarios": [],
        }
        with pytest.raises(ValueError, match="No (valid scenarios|results)"):
            convert(summary)


# ── convert (dual mode) ──────────────────────────────────────────────────

class TestConvertDualMode:
    def test_dual_mode_detected(self, dual_summary):
        report = convert(dual_summary, gpu_name="GPU")
        assert report.get("dual_mode") is True

    def test_has_both_modes(self, dual_summary):
        report = convert(dual_summary, gpu_name="GPU")
        assert "peak_performance" in report
        assert "queue_depth" in report

    def test_each_mode_has_scenarios(self, dual_summary):
        report = convert(dual_summary, gpu_name="GPU")
        for mode_key in ("peak_performance", "queue_depth"):
            mode = report[mode_key]
            assert "x_label" in mode
            assert "scenarios" in mode
            assert len(mode["scenarios"]) == 2

    def test_x_labels_preserved(self, dual_summary):
        report = convert(dual_summary, gpu_name="GPU")
        assert report["peak_performance"]["x_label"] == "Concurrent Users"
        assert report["queue_depth"]["x_label"] == "Queue Depth (Simultaneous Requests)"

    def test_measured_points_present(self, dual_summary):
        report = convert(dual_summary, gpu_name="GPU")
        for mode_key in ("peak_performance", "queue_depth"):
            for scenario in report[mode_key]["scenarios"].values():
                assert len(scenario["measured"]) > 0
                m = scenario["measured"][0]
                assert "batch_size" in m
                assert "decode_tps_per_user" in m
                assert "ttft" in m

    def test_gpu_name_passthrough(self, dual_summary):
        report = convert(dual_summary, gpu_name="8xH200")
        assert report["gpu_name"] == "8xH200"


# ── _convert_dual_mode directly ──────────────────────────────────────────

class TestConvertDualModeDirect:
    def test_missing_mode_graceful(self):
        summary = {
            "model": "m",
            "dual_mode": True,
            "peak_performance": {
                "x_label": "Concurrent Users",
                "scenarios": [{
                    "input_length": 1024,
                    "results": [make_result_bundle(request_rate=1.0, avg_prompt_tokens=1024)],
                }],
            },
        }
        report = _convert_dual_mode(summary, gpu_name="G")
        assert "peak_performance" in report
        # queue_depth should not be in report since it was missing from summary

    def test_empty_scenario_results_skipped(self):
        summary = {
            "model": "m",
            "dual_mode": True,
            "peak_performance": {
                "x_label": "Concurrent Users",
                "scenarios": [{"input_length": 1024, "results": []}],
            },
            "queue_depth": {
                "x_label": "Queue Depth (Simultaneous Requests)",
                "scenarios": [{"input_length": 1024, "results": [make_result_bundle()]}],
            },
        }
        report = _convert_dual_mode(summary, gpu_name="G")
        assert len(report["peak_performance"]["scenarios"]) == 0
        assert len(report["queue_depth"]["scenarios"]) == 1


# ── merge_reports ─────────────────────────────────────────────────────────

class TestMergeReports:
    def test_merge_two_reports(self):
        r1 = {
            "model_name": "m",
            "gpu_name": "g",
            "scenarios": {"1k-in-256-out": {"input_len": 1024, "output_len": 256, "measured": []}},
        }
        r2 = {
            "model_name": "m",
            "gpu_name": "g",
            "scenarios": {"4k-in-256-out": {"input_len": 4096, "output_len": 256, "measured": []}},
        }
        merged = merge_reports([r1, r2])
        assert len(merged["scenarios"]) == 2
        assert "1k-in-256-out" in merged["scenarios"]
        assert "4k-in-256-out" in merged["scenarios"]

    def test_merge_preserves_model_from_first(self):
        r1 = {"model_name": "first", "gpu_name": "g", "scenarios": {"a": {}}}
        r2 = {"model_name": "second", "gpu_name": "g", "scenarios": {"b": {}}}
        merged = merge_reports([r1, r2])
        assert merged["model_name"] == "first"

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="No reports"):
            merge_reports([])

    def test_merge_overlapping_scenarios_last_wins(self):
        r1 = {"model_name": "m", "gpu_name": "g", "scenarios": {"same": {"v": 1}}}
        r2 = {"model_name": "m", "gpu_name": "g", "scenarios": {"same": {"v": 2}}}
        merged = merge_reports([r1, r2])
        assert merged["scenarios"]["same"]["v"] == 2
