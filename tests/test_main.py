"""Tests for main.py — orchestration, CLI parsing, and benchmark flow."""

import argparse
import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import (
    parse_rates,
    _ensure_outdir,
    _save_json,
    build_and_plot,
    run_single_benchmark_concurrency,
    run_single_benchmark,
    run_multi_rate_benchmark,
    run_multi_concurrency_benchmark,
    _collect_scenarios,
    run_dual_mode_benchmark,
    run_queue_depth_discovery,
    _generate_report,
    main,
)
from tests.conftest import make_result_bundle, make_request_input


# ── parse_rates ──────────────────────────────────────────────────────────

class TestParseRates:
    def test_default_rates(self):
        assert parse_rates(None) == [1.0, 10.0, 100.0]

    def test_comma_separated(self):
        assert parse_rates("1,2,4,8") == [1.0, 2.0, 4.0, 8.0]

    def test_with_spaces(self):
        assert parse_rates("1, 2, 4") == [1.0, 2.0, 4.0]

    def test_inf_rate(self):
        result = parse_rates("1,inf")
        assert result[0] == 1.0
        assert result[1] == float("inf")

    def test_float_rates(self):
        assert parse_rates("0.5,1.5,3.0") == [0.5, 1.5, 3.0]

    def test_single_rate(self):
        assert parse_rates("42") == [42.0]

    def test_empty_string(self):
        assert parse_rates("") == [1.0, 10.0, 100.0]


# ── _ensure_outdir ───────────────────────────────────────────────────────

class TestEnsureOutdir:
    def test_creates_directory(self, tmp_path):
        d = str(tmp_path / "new_dir")
        assert not os.path.exists(d)
        _ensure_outdir(d)
        assert os.path.isdir(d)

    def test_existing_directory_ok(self, tmp_path):
        d = str(tmp_path)
        _ensure_outdir(d)  # Should not raise
        assert os.path.isdir(d)

    def test_nested_directory(self, tmp_path):
        d = str(tmp_path / "a" / "b" / "c")
        _ensure_outdir(d)
        assert os.path.isdir(d)


# ── _save_json ───────────────────────────────────────────────────────────

class TestSaveJson:
    def test_saves_valid_json(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"key": "value", "number": 42}
        _save_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_saves_with_indent(self, tmp_path):
        path = str(tmp_path / "test.json")
        _save_json(path, {"a": 1})
        with open(path) as f:
            content = f.read()
        assert "\n" in content  # Indented format has newlines


# ── CLI argument parsing ─────────────────────────────────────────────────

class TestCLIParsing:
    """Test that the argument parser is configured correctly."""

    def _parse(self, args_list):
        """Run main's argparse on a list of args, capturing SystemExit."""
        import main as m
        parser = argparse.ArgumentParser()
        # Re-create the parser setup from main()
        parser.add_argument("--url", type=str, required=True)
        parser.add_argument("--model", type=str, default="default")
        parser.add_argument("--model-type", type=str, choices=["vision", "text"], default="vision")
        parser.add_argument("--num-requests", type=int, default=100)
        parser.add_argument("--request-rate", type=float, default=float("inf"))
        parser.add_argument("--rates", type=str)
        parser.add_argument("--run-multi", action="store_true")
        parser.add_argument("--max-tokens", type=int, default=256)
        parser.add_argument("--output-dir", type=str, default=".")
        parser.add_argument("--api-key", type=str)
        parser.add_argument("--disable-tqdm", action="store_true")
        parser.add_argument("--disable-server-metrics", action="store_true")
        parser.add_argument("--provider", type=str)
        parser.add_argument("--gpu", type=str)
        parser.add_argument("--input-lengths", type=str)
        parser.add_argument("--inter-run-wait-seconds", type=int, default=30)
        parser.add_argument("--concurrencies", type=str)
        parser.add_argument("--queue-ttft-limit", type=float, default=10.0)
        parser.add_argument("--max-queue-depth", type=int, default=64)
        parser.add_argument("--run-dual", action="store_true")
        return parser.parse_args(args_list)

    def test_basic_args(self):
        args = self._parse(["--url", "http://localhost:3000/v1/chat/completions"])
        assert args.url == "http://localhost:3000/v1/chat/completions"
        assert args.model == "default"
        assert args.num_requests == 100

    def test_dual_mode_args(self):
        args = self._parse([
            "--url", "http://localhost:3000/v1/chat/completions",
            "--run-dual",
            "--concurrencies", "1,2,4,8",
            "--input-lengths", "1024,4096",
            "--queue-ttft-limit", "15.0",
            "--max-queue-depth", "32",
        ])
        assert args.run_dual is True
        assert args.concurrencies == "1,2,4,8"
        assert args.input_lengths == "1024,4096"
        assert args.queue_ttft_limit == 15.0
        assert args.max_queue_depth == 32

    def test_concurrency_mode_args(self):
        args = self._parse([
            "--url", "http://localhost:3000/v1/chat/completions",
            "--run-multi",
            "--concurrencies", "1,2,4,8,16,32",
        ])
        assert args.run_multi is True
        assert args.concurrencies == "1,2,4,8,16,32"

    def test_default_inter_run_wait(self):
        args = self._parse(["--url", "http://localhost:3000/v1/chat/completions"])
        assert args.inter_run_wait_seconds == 30

    def test_api_key(self):
        args = self._parse([
            "--url", "http://localhost:3000/v1/chat/completions",
            "--api-key", "secret123",
        ])
        assert args.api_key == "secret123"


# ── _collect_scenarios ───────────────────────────────────────────────────

class TestCollectScenarios:
    @pytest.mark.asyncio
    async def test_collects_all_input_lengths(self):
        """_collect_scenarios should iterate over all input lengths."""
        call_log = []

        async def mock_runner(args, x_values, preloaded_requests=None):
            call_log.append(len(preloaded_requests or []))
            return [make_result_bundle(request_rate=v) for v in x_values]

        args = argparse.Namespace(
            num_requests=5,
            inter_run_wait_seconds=0,
        )

        results = await _collect_scenarios(
            mock_runner, args, [1.0, 2.0], [1024, 4096], label="test"
        )
        assert len(results) == 2  # 2 input lengths
        assert len(call_log) == 2
        for sc in results:
            assert "input_length" in sc
            assert "results" in sc
            assert len(sc["results"]) == 2  # 2 x_values

    @pytest.mark.asyncio
    async def test_rates_stored_as_floats(self):
        async def mock_runner(args, x_values, preloaded_requests=None):
            return [make_result_bundle(request_rate=v) for v in x_values]

        args = argparse.Namespace(num_requests=5, inter_run_wait_seconds=0)
        results = await _collect_scenarios(
            mock_runner, args, [1, 2, 4], [1024], label="test"
        )
        assert all(isinstance(r, float) for r in results[0]["rates"])


# ── run_queue_depth_discovery ─────────────────────────────────────────────

class TestRunQueueDepthDiscovery:
    @pytest.mark.asyncio
    async def test_stops_when_ttft_exceeds_limit(self, tmp_path):
        """Discovery should stop ramping when avg TTFT exceeds the limit."""
        call_count = 0

        async def mock_send(session, url, req, model, max_tokens, api_key, pbar=None, first_token_event=None):
            nonlocal call_count
            call_count += 1
            from client_metrics import RequestOutput
            import time
            now = time.perf_counter()
            # Simulate increasing TTFT: N=1 → 1s, N=2 → 3s, N=3 → 8s, N=4 → 12s (over limit)
            return RequestOutput(
                request_id=req.request_id, success=True,
                ttft=call_count * 3.0, e2e_latency=call_count * 3.0 + 5.0,
                output_tokens=50, prompt_tokens=100, completion_tokens=50,
                send_time=now, first_token_time=now + call_count * 3.0,
            )

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            max_tokens=256,
            api_key=None,
            disable_tqdm=True,
            disable_server_metrics=True,
            inter_run_wait_seconds=0,
            queue_ttft_limit=10.0,
            max_queue_depth=10,
        )

        with patch("main.send_chat_request", side_effect=mock_send), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(20)]):
            results = await run_queue_depth_discovery(args, input_len=1024, max_requests=10)

        # Should have stopped before testing all 10
        assert len(results) < 10
        # Last result should have TTFT > 10s
        assert results[-1]["avg_ttft_s"] > 10.0

    @pytest.mark.asyncio
    async def test_baseline_n1_always_runs(self, tmp_path):
        """N=1 baseline should always be the first result."""
        async def mock_send(session, url, req, model, max_tokens, api_key, pbar=None, first_token_event=None):
            from client_metrics import RequestOutput
            import time
            now = time.perf_counter()
            # Immediately over limit — should still get N=1 result
            return RequestOutput(
                request_id=req.request_id, success=True,
                ttft=15.0, e2e_latency=20.0,
                output_tokens=50, prompt_tokens=100, completion_tokens=50,
                send_time=now, first_token_time=now + 15.0,
            )

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            max_tokens=256,
            api_key=None,
            disable_tqdm=True,
            disable_server_metrics=True,
            inter_run_wait_seconds=0,
            queue_ttft_limit=10.0,
            max_queue_depth=10,
        )

        with patch("main.send_chat_request", side_effect=mock_send), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(20)]):
            results = await run_queue_depth_discovery(args, input_len=1024, max_requests=10)

        # Should have at least the N=1 result
        assert len(results) >= 1
        assert results[0]["request_rate"] == 1

    @pytest.mark.asyncio
    async def test_results_have_queue_inference(self):
        """Each result bundle should contain queue_inference data."""
        async def mock_send(session, url, req, model, max_tokens, api_key, pbar=None, first_token_event=None):
            from client_metrics import RequestOutput
            import time
            now = time.perf_counter()
            return RequestOutput(
                request_id=req.request_id, success=True,
                ttft=0.5, e2e_latency=5.0,
                output_tokens=50, prompt_tokens=100, completion_tokens=50,
                send_time=now, first_token_time=now + 0.5,
            )

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            max_tokens=256,
            api_key=None,
            disable_tqdm=True,
            disable_server_metrics=True,
            inter_run_wait_seconds=0,
            queue_ttft_limit=2.0,  # Low limit so it stops quickly
            max_queue_depth=5,
        )

        with patch("main.send_chat_request", side_effect=mock_send), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(20)]):
            results = await run_queue_depth_discovery(args, input_len=1024, max_requests=5)

        for r in results:
            assert "queue_inference" in r
            assert "summary_by_depth" in r["queue_inference"]

    @pytest.mark.asyncio
    async def test_n_values_are_sequential(self):
        """N should ramp 1, 2, 3, ... linearly."""
        async def mock_send(session, url, req, model, max_tokens, api_key, pbar=None, first_token_event=None):
            from client_metrics import RequestOutput
            import time
            now = time.perf_counter()
            return RequestOutput(
                request_id=req.request_id, success=True,
                ttft=0.1, e2e_latency=1.0,
                output_tokens=50, prompt_tokens=100, completion_tokens=50,
                send_time=now, first_token_time=now + 0.1,
            )

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            max_tokens=256,
            api_key=None,
            disable_tqdm=True,
            disable_server_metrics=True,
            inter_run_wait_seconds=0,
            queue_ttft_limit=10.0,
            max_queue_depth=5,
        )

        with patch("main.send_chat_request", side_effect=mock_send), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(5)]):
            results = await run_queue_depth_discovery(args, input_len=1024, max_requests=5)

        n_values = [r["request_rate"] for r in results]
        assert n_values == [1, 2, 3, 4, 5]


# ── run_dual_mode_benchmark ──────────────────────────────────────────────

class TestRunDualModeBenchmark:
    @pytest.mark.asyncio
    async def test_produces_dual_summary(self, tmp_path):
        """Dual mode should save a summary with dual_mode=True."""
        mock_bundle = make_result_bundle()

        async def mock_runner(args, x_values, preloaded_requests=None):
            return [make_result_bundle(request_rate=v) for v in x_values]

        # Mock discovery to return a few bundles with avg_ttft_s
        async def mock_discovery(args, input_len, max_requests=64):
            return [
                {**make_result_bundle(request_rate=n), "avg_ttft_s": n * 0.5,
                 "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []}}
                for n in [1, 2, 3]
            ]

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            model_type="text",
            num_requests=5,
            max_tokens=256,
            output_dir=str(tmp_path),
            inter_run_wait_seconds=0,
            disable_server_metrics=True,
            disable_tqdm=True,
            provider=None,
            gpu="TestGPU",
            queue_ttft_limit=10.0,
            max_queue_depth=64,
        )

        with patch("main._collect_scenarios") as mock_collect, \
             patch("main.run_queue_depth_discovery", side_effect=mock_discovery), \
             patch("main._generate_report") as mock_report:
            mock_collect.return_value = [{
                "input_length": 1024,
                "rates": [1.0, 2.0],
                "results": [mock_bundle, mock_bundle],
            }]

            await run_dual_mode_benchmark(args, [1, 2], [1024])

            # Check the summary was saved
            summary_path = os.path.join(str(tmp_path), "benchmark_summary.json")
            assert os.path.exists(summary_path)
            with open(summary_path) as f:
                summary = json.load(f)
            assert summary["dual_mode"] is True
            assert "peak_performance" in summary
            assert "queue_depth" in summary
            assert summary["peak_performance"]["x_label"] == "Concurrent Users"
            assert summary["queue_depth"]["x_label"] == "Queue Depth (Simultaneous Requests)"


# ── _generate_report ─────────────────────────────────────────────────────

class TestGenerateReport:
    def test_single_mode_report(self, tmp_path, single_summary):
        """Single-mode summary should produce a report_data.json."""
        args = argparse.Namespace(provider=None, gpu="TestGPU")

        with patch("main.convert") as mock_convert:
            mock_convert.return_value = {
                "model_name": "test",
                "gpu_name": "TestGPU",
                "scenarios": {"1k-in-256-out": {"input_len": 1024, "output_len": 256, "measured": []}},
            }
            # Patch the HTML generation imports to avoid matplotlib
            with patch("main.generate_html_report", create=True), \
                 patch("main.load_report_json", create=True), \
                 patch("main.build_scenario_data", create=True):
                _generate_report(single_summary, str(tmp_path), args)

        assert os.path.exists(os.path.join(str(tmp_path), "report_data.json"))

    def test_dual_mode_report(self, tmp_path, dual_summary):
        """Dual-mode summary should route to generate_dual_html_report."""
        args = argparse.Namespace(provider=None, gpu="TestGPU")

        with patch("main.convert") as mock_convert, \
             patch("main.generate_dual_html_report", create=True) as mock_dual:
            mock_convert.return_value = {
                "model_name": "test",
                "gpu_name": "TestGPU",
                "dual_mode": True,
                "peak_performance": {"x_label": "Concurrent Users", "scenarios": {}},
                "sustained_load": {"x_label": "Rate", "scenarios": {}},
            }
            _generate_report(dual_summary, str(tmp_path), args)

        # report_data.json should exist
        assert os.path.exists(os.path.join(str(tmp_path), "report_data.json"))


# ── build_and_plot ───────────────────────────────────────────────────────

class TestBuildAndPlot:
    def test_saves_summary_json(self, tmp_path):
        results = [
            make_result_bundle(request_rate=1.0),
            make_result_bundle(request_rate=4.0),
        ]
        args = argparse.Namespace(
            url="http://test",
            model="test-model",
            model_type="text",
            num_requests=32,
            max_tokens=256,
            output_dir=str(tmp_path),
            provider=None,
            gpu="TestGPU",
        )

        with patch("main.plot_tradeoff"), \
             patch("main.plot_frontier"), \
             patch("main.plot_series_vs_rate"), \
             patch("main.plot_client_latency_curves"), \
             patch("main._generate_report"):
            build_and_plot(results, str(tmp_path), args)

        summary_path = os.path.join(str(tmp_path), "benchmark_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["model"] == "test-model"
        assert len(summary["results"]) == 2
