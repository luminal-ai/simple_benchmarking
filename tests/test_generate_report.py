"""Tests for generate_report.py — chart generation and HTML report building."""

import json
import os
import tempfile

import pytest
import pandas as pd

from generate_report import (
    load_report_json,
    build_scenario_data,
    _friendly_name,
    _sorted_scenarios,
    _distinct_colors,
    _get_metric,
    _get_scenario_thresholds,
    _find_capacity,
    chart_throughput_range,
    chart_ttft_range,
    chart_itl,
    chart_decode_speed,
    chart_scaling_efficiency,
    chart_capacity,
    generate_html_report,
    fig_to_base64,
)
from tests.conftest import (
    make_scenarios_data,
    make_single_report_data,
    make_scenario_measured,
)


# ── Helper functions ─────────────────────────────────────────────────────

class TestFriendlyName:
    def test_known_prefixes(self):
        assert _friendly_name("1k") == "Chatbot"
        assert _friendly_name("4k") == "RAG / QA"
        assert _friendly_name("16k") == "Agentic"
        assert _friendly_name("32k") == "Tool Calling Agentic"

    def test_scenario_with_suffix(self):
        assert _friendly_name("1k-in-256-out") == "Chatbot"
        assert _friendly_name("4k-in-1302-out") == "RAG / QA"

    def test_unknown_scenario(self):
        assert _friendly_name("unknown-scenario") == "unknown-scenario"

    def test_numeric_prefix(self):
        assert _friendly_name("8k-in-500-out") == "8k-in-500-out"


class TestSortedScenarios:
    def test_sorts_by_input_len(self):
        data = {
            "32k-in-100-out": {"input_len": 32768},
            "1k-in-100-out": {"input_len": 1024},
            "4k-in-100-out": {"input_len": 4096},
        }
        result = _sorted_scenarios(data)
        assert result == ["1k-in-100-out", "4k-in-100-out", "32k-in-100-out"]

    def test_single_scenario(self):
        data = {"only": {"input_len": 1024}}
        assert _sorted_scenarios(data) == ["only"]


class TestDistinctColors:
    def test_returns_n_colors(self):
        for n in [1, 3, 5, 12]:
            colors = _distinct_colors(n)
            assert len(colors) == n

    def test_all_are_hex_strings(self):
        colors = _distinct_colors(5)
        for c in colors:
            assert c.startswith("#")

    def test_more_than_base_colors(self):
        colors = _distinct_colors(20)
        assert len(colors) == 20
        assert len(set(colors)) == 20  # All unique


class TestGetMetric:
    def test_extracts_metric(self):
        data = make_scenarios_data()
        sorted_ids = _sorted_scenarios(data)
        values = _get_metric(data, sorted_ids, "decode_tps_per_user", 1)
        assert len(values) == len(sorted_ids)
        assert all(v is not None and v > 0 for v in values)

    def test_missing_batch_size_returns_zero(self):
        data = make_scenarios_data()
        sorted_ids = _sorted_scenarios(data)
        values = _get_metric(data, sorted_ids, "decode_tps_per_user", 9999)
        assert all(v == 0.0 for v in values)


# ── load_report_json ─────────────────────────────────────────────────────

class TestLoadReportJson:
    def test_loads_single_mode(self, tmp_path):
        data = make_single_report_data()
        path = str(tmp_path / "report_data.json")
        with open(path, "w") as f:
            json.dump(data, f)

        df, model_name, gpu_name = load_report_json(path)
        assert isinstance(df, pd.DataFrame)
        assert model_name == "test/model"
        assert gpu_name == "8xH200"
        assert "scenario" in df.columns
        assert "batch_size" in df.columns
        assert len(df) > 0

    def test_all_scenarios_present(self, tmp_path):
        data = make_single_report_data()
        path = str(tmp_path / "report_data.json")
        with open(path, "w") as f:
            json.dump(data, f)

        df, _, _ = load_report_json(path)
        scenarios = df["scenario"].unique()
        assert len(scenarios) == 2

    def test_columns_are_numeric(self, tmp_path):
        data = make_single_report_data()
        path = str(tmp_path / "report_data.json")
        with open(path, "w") as f:
            json.dump(data, f)

        df, _, _ = load_report_json(path)
        for col in ["batch_size", "prefill_tps_per_user", "decode_tps_per_user"]:
            assert df[col].dtype in ["float64", "int64"]


# ── build_scenario_data ──────────────────────────────────────────────────

class TestBuildScenarioData:
    def test_builds_from_dataframe(self, tmp_path):
        data = make_single_report_data()
        path = str(tmp_path / "report_data.json")
        with open(path, "w") as f:
            json.dump(data, f)

        df, _, _ = load_report_json(path)
        sd = build_scenario_data(df)
        assert len(sd) == 2
        for scenario in sd.values():
            assert "input_len" in scenario
            assert "output_len" in scenario
            assert "measured" in scenario
            assert len(scenario["measured"]) > 0

    def test_measured_has_expected_fields(self, tmp_path):
        data = make_single_report_data()
        path = str(tmp_path / "report_data.json")
        with open(path, "w") as f:
            json.dump(data, f)

        df, _, _ = load_report_json(path)
        sd = build_scenario_data(df)
        for scenario in sd.values():
            for m in scenario["measured"]:
                assert "batch_size" in m
                assert "prefill_tps_per_user" in m
                assert "decode_tps_per_user" in m
                assert "ttft" in m
                assert "e2e" in m


# ── Capacity analysis functions ──────────────────────────────────────────

class TestGetScenarioThresholds:
    def test_returns_expected_keys(self):
        scenario = {"input_len": 1024, "measured": make_scenario_measured()}
        thresholds = _get_scenario_thresholds(scenario)
        assert "maxITL" in thresholds
        assert "rightMetric" in thresholds
        assert "maxTTFT" in thresholds
        assert thresholds["maxITL"] == 20

    def test_returns_consistent_thresholds(self):
        """Thresholds are constant across all scenarios."""
        short = _get_scenario_thresholds({"input_len": 1024, "measured": make_scenario_measured()})
        long = _get_scenario_thresholds({"input_len": 32768, "measured": make_scenario_measured()})
        assert short["maxTTFT"] == long["maxTTFT"]
        assert short["maxITL"] == long["maxITL"]


class TestFindCapacity:
    def test_all_within_threshold(self):
        measured = [
            {"batch_size": 1, "decode_tps_per_user": 100, "ttft": 0.15},
            {"batch_size": 2, "decode_tps_per_user": 80, "ttft": 0.5},
            {"batch_size": 4, "decode_tps_per_user": 60, "ttft": 0.8},
        ]
        thresholds = {"maxITL": 20, "maxTTFT": 1.0}
        cap = _find_capacity(measured, thresholds)
        assert cap == 4

    def test_exceeds_itl_threshold(self):
        measured = [
            {"batch_size": 1, "decode_tps_per_user": 100, "ttft": 0.15},
            {"batch_size": 2, "decode_tps_per_user": 10, "ttft": 0.5},  # ITL = 100ms > 20ms
        ]
        thresholds = {"maxITL": 20, "maxTTFT": 1.0}
        cap = _find_capacity(measured, thresholds)
        assert cap == 1

    def test_exceeds_ttft_threshold(self):
        measured = [
            {"batch_size": 1, "decode_tps_per_user": 100, "ttft": 0.15},
            {"batch_size": 2, "decode_tps_per_user": 80, "ttft": 5.0},  # > maxTTFT
        ]
        thresholds = {"maxITL": 20, "maxTTFT": 1.0}
        cap = _find_capacity(measured, thresholds)
        assert cap == 1

    def test_none_within_threshold(self):
        measured = [
            {"batch_size": 1, "decode_tps_per_user": 5, "ttft": 5.0},  # Both bad
        ]
        thresholds = {"maxITL": 20, "maxTTFT": 1.0}
        cap = _find_capacity(measured, thresholds)
        assert cap == 0

    def test_empty_measured(self):
        assert _find_capacity([], {"maxITL": 20, "maxTTFT": 1.0}) == 0


# ── Chart generation (smoke tests) ──────────────────────────────────────

class TestChartGeneration:
    """Smoke tests: verify chart functions return base64 image strings."""

    @pytest.fixture
    def chart_data(self):
        return make_scenarios_data(), [1, 2, 4, 8]

    def test_chart_throughput_range(self, chart_data):
        sd, bs = chart_data
        result = chart_throughput_range(sd, bs)
        assert result.startswith("data:image/png;base64,")

    def test_chart_ttft_range(self, chart_data):
        sd, bs = chart_data
        result = chart_ttft_range(sd, bs)
        assert result.startswith("data:image/png;base64,")

    def test_chart_itl(self, chart_data):
        sd, bs = chart_data
        result = chart_itl(sd, bs)
        assert result.startswith("data:image/png;base64,")

    def test_chart_decode_speed(self, chart_data):
        sd, bs = chart_data
        result = chart_decode_speed(sd, bs)
        assert result.startswith("data:image/png;base64,")

    def test_chart_scaling_efficiency(self, chart_data):
        sd, bs = chart_data
        result = chart_scaling_efficiency(sd, bs)
        assert result.startswith("data:image/png;base64,")

    def test_chart_capacity(self, chart_data):
        sd, bs = chart_data
        sid = list(sd.keys())[0]
        thresholds = _get_scenario_thresholds(sd[sid])
        result = chart_capacity(sid, sd[sid], thresholds)
        assert result.startswith("data:image/png;base64,")


# ── fig_to_base64 ────────────────────────────────────────────────────────

class TestFigToBase64:
    def test_returns_data_uri(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = fig_to_base64(fig)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 100  # Should have substantial data


# ── HTML report generation ───────────────────────────────────────────────

class TestGenerateHtmlReport:
    def test_single_mode_html(self, tmp_path):
        data = make_single_report_data()
        json_path = str(tmp_path / "report_data.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        df, model_name, gpu_name = load_report_json(json_path)
        sd = build_scenario_data(df)

        out_dir = str(tmp_path / "report")
        os.makedirs(out_dir)
        generate_html_report(df, model_name, gpu_name, sd, out_dir)

        html_path = os.path.join(out_dir, "index.html")
        assert os.path.exists(html_path)
        with open(html_path) as f:
            html = f.read()
        assert "test/model" in html
        assert "8xH200" in html
        assert "data:image/png;base64," in html
        # No unreplaced placeholders
        assert "{{" not in html
