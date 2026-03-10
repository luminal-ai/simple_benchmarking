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
    run_queue_test,
    merge_phase1_and_phase2,
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
        parser.add_argument("--max-queue-depth", type=int, default=256)
        parser.add_argument("--queue-confirm-runs", type=int, default=3)
        parser.add_argument("--run-queue-test", action="store_true")
        parser.add_argument("--run-dual", action="store_true")
        parser.add_argument("--merge-results", nargs=2, metavar=("PEAK_JSON", "QUEUE_JSON"))
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

    def test_merge_results_args(self):
        args = self._parse([
            "--url", "http://localhost:3000/v1/chat/completions",
            "--merge-results", "peak.json", "queue.json",
        ])
        assert args.merge_results == ["peak.json", "queue.json"]


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


# ── run_queue_depth_discovery (3-phase: scout → narrow → confirm) ─────────

def _make_ttft_mock(crossover_n, baseline_ttft=0.1):
    """Create a mock send_chat_request where TTFT spikes above 10s around crossover_n.

    Below crossover_n: TTFT is low (baseline + small growth).
    At/above crossover_n: TTFT jumps above 10s.
    """
    # Track how many requests are in each "step" (N) to give consistent TTFT per step
    step_call_counts = {}

    async def mock_send(session, url, req, model, max_tokens, api_key, pbar=None, first_token_event=None):
        from client_metrics import RequestOutput
        import time
        now = time.perf_counter()
        # Use request_id to determine which N-step this belongs to
        # We'll rely on the n_requests tracker instead
        rid = req.request_id

        # Estimate N from how many requests are in this batch
        # We use a simpler approach: TTFT based on request_id proximity
        # The mock is called with all N requests for a step — we track via nonlocal
        n = mock_send._current_n
        if n < crossover_n:
            ttft = baseline_ttft + n * 0.01  # Slowly increasing
        else:
            ttft = 10.0 + (n - crossover_n) * 2.0  # Spikes above 10s

        return RequestOutput(
            request_id=rid, success=True,
            ttft=ttft, e2e_latency=ttft + 5.0,
            output_tokens=50, prompt_tokens=100, completion_tokens=50,
            send_time=now, first_token_time=now + ttft,
        )

    mock_send._current_n = 1
    return mock_send


def _queue_test_args(**overrides):
    """Standard args namespace for queue discovery tests."""
    defaults = dict(
        url="http://test:3000/v1/chat/completions",
        model="test-model",
        max_tokens=256,
        api_key=None,
        disable_tqdm=True,
        disable_server_metrics=True,
        inter_run_wait_seconds=0,
        queue_ttft_limit=10.0,
        max_queue_depth=256,
        queue_confirm_runs=3,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunQueueDepthDiscovery:
    """Tests for the 3-phase queue depth discovery: scout → narrow → confirm."""

    @pytest.mark.asyncio
    async def test_scout_uses_exponential_steps(self):
        """Scout phase should double N: 1, 2, 4, 8, 16, ..."""
        tested_ns = []

        async def mock_step(session, args, n, requests):
            tested_ns.append(n)
            mock_step._current_n = n
            ttft = 0.1 if n < 20 else 12.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args(max_queue_depth=128)
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # Scout should have tested exponential sequence up to crossing
        scout_ns = [r["request_rate"] for r in result["scout_results"]]
        assert scout_ns[0] == 1
        # Each step should roughly double
        for i in range(1, len(scout_ns)):
            assert scout_ns[i] >= scout_ns[i - 1] * 2 or scout_ns[i] > args.queue_ttft_limit

    @pytest.mark.asyncio
    async def test_scout_finds_upper_bound(self):
        """Scout should stop once it finds an N where TTFT > limit."""
        async def mock_step(session, args, n, requests):
            ttft = 0.5 if n <= 8 else 15.0  # Crosses at N=16
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # Scout should have found that N=16 is bad
        scout_ns = [r["request_rate"] for r in result["scout_results"]]
        last_scout = result["scout_results"][-1]
        assert last_scout["avg_ttft_s"] > 10.0

    @pytest.mark.asyncio
    async def test_narrow_binary_searches(self):
        """Narrow phase should binary search between last_good and first_bad."""
        async def mock_step(session, args, n, requests):
            # Crossover at N=12: N<=11 is OK, N>=12 is over limit
            ttft = 0.5 + n * 0.3 if n <= 11 else 11.0 + (n - 11) * 1.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # Narrow phase should have run
        assert len(result["narrow_results"]) > 0
        # Should have found max_safe_n close to 11
        assert result["max_safe_n"] in range(10, 13)

    @pytest.mark.asyncio
    async def test_confirm_phase_runs(self):
        """Confirm phase should run multiple times at the found max_safe_n."""
        async def mock_step(session, args, n, requests):
            ttft = 0.5 if n <= 10 else 12.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args(queue_confirm_runs=3)
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # Confirm phase should have 3 results
        assert len(result["confirm_results"]) == 3
        # All confirm runs should be at max_safe_n
        for r in result["confirm_results"]:
            assert r["request_rate"] == result["max_safe_n"]

    @pytest.mark.asyncio
    async def test_n1_over_limit_no_narrow_or_confirm(self):
        """If N=1 is already over limit, skip narrow/confirm and report max_safe_n=0."""
        async def mock_step(session, args, n, requests):
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": 15.0,  # Always over limit
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        assert result["max_safe_n"] == 0
        # N=1 tested once + 2 retests (confirm_runs=3) = 3 scout results
        assert len(result["scout_results"]) == 3
        assert len(result["narrow_results"]) == 0
        assert len(result["confirm_results"]) == 0

    @pytest.mark.asyncio
    async def test_never_exceeds_limit_reports_max(self):
        """If TTFT never exceeds limit, report max_safe_n = max tested."""
        async def mock_step(session, args, n, requests):
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": 0.5,  # Always under limit
            }

        args = _queue_test_args(max_queue_depth=32)
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # Should have scouted up to max_queue_depth and reported that as safe
        assert result["max_safe_n"] == 32
        assert len(result["narrow_results"]) == 0
        # Should still confirm at max
        assert len(result["confirm_results"]) == 3

    @pytest.mark.asyncio
    async def test_all_results_sorted_by_n(self):
        """all_results should contain every data point sorted by N."""
        async def mock_step(session, args, n, requests):
            ttft = 0.5 if n <= 6 else 12.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        all_ns = [r["request_rate"] for r in result["all_results"]]
        assert all_ns == sorted(all_ns)
        # Should include scout + narrow + confirm data
        total = len(result["scout_results"]) + len(result["narrow_results"]) + len(result["confirm_results"])
        assert len(result["all_results"]) == total

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self):
        """Discovery result should have all expected top-level fields."""
        async def mock_step(session, args, n, requests):
            ttft = 0.5 if n <= 5 else 12.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        assert "input_length" in result
        assert "max_safe_n" in result
        assert "ttft_limit" in result
        assert "scout_results" in result
        assert "narrow_results" in result
        assert "confirm_results" in result
        assert "all_results" in result
        assert result["input_length"] == 1024
        assert result["ttft_limit"] == 10.0

    @pytest.mark.asyncio
    async def test_confirm_failure_drops_max_safe_n(self):
        """If confirmation runs show TTFT over limit, max_safe_n should drop."""
        confirm_call = 0

        async def mock_step(session, args, n, requests):
            nonlocal confirm_call
            # Crossover at N=8, but confirmation is flaky — 2 of 3 fail
            if n < 8:
                ttft = 0.5
            elif n > 8:
                ttft = 12.0
            else:
                # N=8: first call OK, subsequent calls (confirm) sometimes fail
                confirm_call += 1
                ttft = 9.0 if confirm_call <= 1 else 11.0
            return {
                "request_rate": n, "num_total": n, "num_successful": n,
                "avg_prompt_tokens": 100, "avg_completion_tokens": 50,
                "client_metrics": make_result_bundle(request_rate=n)["client_metrics"],
                "server_metrics": None,
                "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []},
                "avg_ttft_s": ttft,
            }

        args = _queue_test_args()
        with patch("main._run_queue_depth_step", side_effect=mock_step), \
             patch("main.load_text_dataset_with_length", return_value=[make_request_input(i) for i in range(256)]):
            result = await run_queue_depth_discovery(args, input_len=1024)

        # max_safe_n should be less than 8 since confirmation failed
        assert result["max_safe_n"] < 8


# ── run_queue_test (standalone --run-queue-test) ─────────────────────────

class TestRunQueueTest:
    @pytest.mark.asyncio
    async def test_saves_queue_summary(self, tmp_path):
        """Standalone queue test should save queue_summary.json."""
        async def mock_discovery(args, input_len, max_requests=256):
            return {
                "input_length": input_len,
                "max_safe_n": 10,
                "ttft_limit": 10.0,
                "scout_results": [make_result_bundle(request_rate=1)],
                "narrow_results": [],
                "confirm_results": [make_result_bundle(request_rate=10)] * 3,
                "all_results": [make_result_bundle(request_rate=1), make_result_bundle(request_rate=10)],
            }

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            model_type="text",
            num_requests=32,
            max_tokens=256,
            output_dir=str(tmp_path),
            inter_run_wait_seconds=0,
            disable_server_metrics=True,
            disable_tqdm=True,
            provider=None,
            gpu="TestGPU",
            queue_ttft_limit=10.0,
            max_queue_depth=256,
            queue_confirm_runs=3,
        )

        with patch("main.run_queue_depth_discovery", side_effect=mock_discovery), \
             patch("main._generate_report") as mock_report:
            from main import run_queue_test
            await run_queue_test(args, [1024, 4096])

        summary_path = os.path.join(str(tmp_path), "queue_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert len(summary["scenarios"]) == 2
        assert summary["scenarios"][0]["max_safe_n"] == 10

        # Verify _generate_report was called
        mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_queue_summary_compatible_with_convert(self, tmp_path):
        """queue_summary.json should be convertible via the multi-scenario path."""
        from convert_results import convert

        async def mock_discovery(args, input_len, max_requests=256):
            return {
                "input_length": input_len,
                "max_safe_n": 8,
                "ttft_limit": 10.0,
                "scout_results": [],
                "narrow_results": [],
                "confirm_results": [],
                "all_results": [
                    make_result_bundle(request_rate=n) for n in [1, 2, 4, 8]
                ],
            }

        args = argparse.Namespace(
            url="http://test:3000/v1/chat/completions",
            model="test-model",
            model_type="text",
            num_requests=32,
            max_tokens=256,
            output_dir=str(tmp_path),
            inter_run_wait_seconds=0,
            disable_server_metrics=True,
            disable_tqdm=True,
            provider=None,
            gpu="TestGPU",
            queue_ttft_limit=10.0,
            max_queue_depth=256,
            queue_confirm_runs=3,
        )

        with patch("main.run_queue_depth_discovery", side_effect=mock_discovery), \
             patch("main._generate_report"):
            await run_queue_test(args, [1024])

        # Load the saved queue_summary.json and convert it
        with open(os.path.join(str(tmp_path), "queue_summary.json")) as f:
            summary = json.load(f)

        report_data = convert(summary, gpu_name="TestGPU")
        assert "scenarios" in report_data
        assert report_data["model_name"] == "test-model"
        # Should have a scenario for the 1024 input length
        assert len(report_data["scenarios"]) == 1


# ── merge_phase1_and_phase2 ──────────────────────────────────────────────

class TestMergePhase1AndPhase2:
    def test_merge_produces_dual_mode(self, tmp_path):
        """Merging Phase 1 + Phase 2 JSONs should produce a dual-mode summary."""
        peak = {
            "url": "http://test:3000/v1/chat/completions",
            "model": "test-model",
            "model_type": "text",
            "num_requests": 32,
            "max_tokens": 256,
            "input_lengths": [1024],
            "scenarios": [
                {
                    "input_length": 1024,
                    "rates": [1.0, 2.0, 4.0],
                    "results": [make_result_bundle(request_rate=r) for r in [1, 2, 4]],
                }
            ],
        }
        queue = {
            "url": "http://test:3000/v1/chat/completions",
            "model": "test-model",
            "model_type": "text",
            "max_tokens": 256,
            "ttft_limit": 10.0,
            "scenarios": [
                {
                    "input_length": 1024,
                    "max_safe_n": 8,
                    "ttft_limit": 10.0,
                    "rates": [1.0, 2.0, 4.0, 8.0],
                    "results": [make_result_bundle(request_rate=r) for r in [1, 2, 4, 8]],
                }
            ],
        }

        peak_path = str(tmp_path / "benchmark_summary.json")
        queue_path = str(tmp_path / "queue_summary.json")
        with open(peak_path, "w") as f:
            json.dump(peak, f)
        with open(queue_path, "w") as f:
            json.dump(queue, f)

        args = argparse.Namespace(
            output_dir=str(tmp_path / "merged"),
            provider=None,
            gpu="TestGPU",
        )

        with patch("main._generate_report"):
            merge_phase1_and_phase2(peak_path, queue_path, args)

        merged_path = str(tmp_path / "merged" / "benchmark_summary.json")
        assert os.path.exists(merged_path)
        with open(merged_path) as f:
            merged = json.load(f)
        assert merged["dual_mode"] is True
        assert "peak_performance" in merged
        assert "queue_depth" in merged
        assert merged["peak_performance"]["x_label"] == "Concurrent Users"
        assert merged["queue_depth"]["x_label"] == "Queue Depth (Simultaneous Requests)"
        assert len(merged["queue_depth"]["scenarios"]) == 1
        assert len(merged["peak_performance"]["scenarios"]) == 1

    def test_merge_calls_generate_report(self, tmp_path):
        """Merge should call _generate_report with the merged data."""
        peak = {
            "url": "http://test", "model": "m", "model_type": "text",
            "num_requests": 10, "max_tokens": 256,
            "scenarios": [{"input_length": 1024, "rates": [1.0],
                           "results": [make_result_bundle(request_rate=1)]}],
        }
        queue = {
            "url": "http://test", "model": "m", "model_type": "text",
            "max_tokens": 256, "ttft_limit": 10.0,
            "scenarios": [{"input_length": 1024, "max_safe_n": 4, "ttft_limit": 10.0,
                           "rates": [1.0], "results": [make_result_bundle(request_rate=1)]}],
        }

        peak_path = str(tmp_path / "peak.json")
        queue_path = str(tmp_path / "queue.json")
        with open(peak_path, "w") as f:
            json.dump(peak, f)
        with open(queue_path, "w") as f:
            json.dump(queue, f)

        args = argparse.Namespace(output_dir=str(tmp_path), provider=None, gpu="TestGPU")

        with patch("main._generate_report") as mock_report:
            merge_phase1_and_phase2(peak_path, queue_path, args)
            mock_report.assert_called_once()
            call_args = mock_report.call_args[0]
            assert call_args[0]["dual_mode"] is True


# ── run_dual_mode_benchmark ──────────────────────────────────────────────

class TestRunDualModeBenchmark:
    @pytest.mark.asyncio
    async def test_produces_dual_summary(self, tmp_path):
        """Dual mode should save a summary with dual_mode=True."""
        mock_bundle = make_result_bundle()

        async def mock_runner(args, x_values, preloaded_requests=None):
            return [make_result_bundle(request_rate=v) for v in x_values]

        # Mock discovery to return the new dict format
        async def mock_discovery(args, input_len, max_requests=256):
            return {
                "input_length": input_len,
                "max_safe_n": 10,
                "ttft_limit": 10.0,
                "scout_results": [],
                "narrow_results": [],
                "confirm_results": [],
                "all_results": [
                    {**make_result_bundle(request_rate=n), "avg_ttft_s": n * 0.5,
                     "queue_inference": {"n_requests": n, "inferred": [], "summary_by_depth": []}}
                    for n in [1, 2, 4, 8]
                ],
            }

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
            max_queue_depth=256,
            queue_confirm_runs=3,
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
