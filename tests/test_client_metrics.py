"""Tests for client_metrics.py — metrics calculation and request handling."""

import asyncio
import json
import pytest
import numpy as np

from client_metrics import (
    RequestInput,
    RequestOutput,
    BenchmarkMetrics,
    calculate_metrics,
    generate_requests,
    count_tokens_starmie,
    load_text_dataset_with_length,
    infer_queue_depths,
    summarize_by_queue_depth,
)
from tests.conftest import make_request_output, make_request_input


# ── RequestInput / RequestOutput dataclasses ─────────────────────────────

class TestDataclasses:
    def test_request_input_defaults(self):
        ri = RequestInput(prompt="Hi", image_url=None, request_id=0, prompt_length=2)
        assert ri.image_size_bytes == 0
        assert ri.image_url is None

    def test_request_output_defaults(self):
        ro = RequestOutput(request_id=0)
        assert ro.success is False
        assert ro.ttft == 0.0
        assert ro.itl == []
        assert ro.prompt_tokens == 0

    def test_request_output_with_itl(self):
        ro = RequestOutput(request_id=1, itl=[0.02, 0.03, 0.025])
        assert len(ro.itl) == 3


# ── calculate_metrics ────────────────────────────────────────────────────

class TestCalculateMetrics:
    def test_basic_metrics(self, request_outputs):
        metrics = calculate_metrics(request_outputs, duration_s=10.0)
        assert isinstance(metrics, BenchmarkMetrics)
        assert metrics.ttft_avg_ms > 0
        assert metrics.e2e_avg_ms > 0
        assert metrics.itl_avg_ms > 0

    def test_ttft_percentiles_ordered(self, request_outputs):
        metrics = calculate_metrics(request_outputs, duration_s=10.0)
        assert metrics.ttft_p50_ms <= metrics.ttft_p95_ms <= metrics.ttft_p99_ms

    def test_e2e_percentiles_ordered(self, request_outputs):
        metrics = calculate_metrics(request_outputs, duration_s=10.0)
        assert metrics.e2e_p50_ms <= metrics.e2e_p95_ms <= metrics.e2e_p99_ms

    def test_itl_in_milliseconds(self, request_outputs):
        metrics = calculate_metrics(request_outputs, duration_s=10.0)
        # ITL should be in ms (our fixtures use 0.02s = 20ms)
        assert metrics.itl_avg_ms == pytest.approx(20.0, abs=1.0)

    def test_ttft_in_milliseconds(self):
        outputs = [make_request_output(ttft=0.5)]  # 500ms
        metrics = calculate_metrics(outputs, duration_s=1.0)
        assert metrics.ttft_avg_ms == pytest.approx(500.0, abs=1.0)

    def test_no_successful_raises(self):
        outputs = [make_request_output(success=False, error="fail")]
        with pytest.raises(ValueError, match="No successful"):
            calculate_metrics(outputs, duration_s=1.0)

    def test_filters_out_failures(self, request_outputs_with_failures):
        metrics = calculate_metrics(request_outputs_with_failures, duration_s=10.0)
        # Should still work — failures are filtered out
        assert metrics.ttft_avg_ms > 0

    def test_toks_per_sec_calculation(self):
        output = make_request_output(
            prompt_tokens=100,
            completion_tokens=50,
            e2e_latency=1.5,  # 1.5 seconds
        )
        metrics = calculate_metrics([output], duration_s=2.0)
        expected_tps = (100 + 50) / 1.5
        assert metrics.toks_avg == pytest.approx(expected_tps, rel=0.01)

    def test_single_request(self):
        output = make_request_output(ttft=0.2, e2e_latency=5.0)
        metrics = calculate_metrics([output], duration_s=5.0)
        assert metrics.ttft_p50_ms == pytest.approx(200.0, abs=1.0)
        assert metrics.ttft_p95_ms == pytest.approx(200.0, abs=1.0)
        assert metrics.ttft_p99_ms == pytest.approx(200.0, abs=1.0)

    def test_zero_ttft_excluded(self):
        outputs = [
            make_request_output(ttft=0.0, e2e_latency=1.0),  # zero TTFT
            make_request_output(request_id=1, ttft=0.5, e2e_latency=2.0),
        ]
        metrics = calculate_metrics(outputs, duration_s=5.0)
        # Only the 0.5s TTFT should be counted
        assert metrics.ttft_avg_ms == pytest.approx(500.0, abs=1.0)

    def test_empty_itl_gives_zero(self):
        output = make_request_output(itl=[], output_tokens=1)
        metrics = calculate_metrics([output], duration_s=1.0)
        assert metrics.itl_avg_ms == 0


# ── generate_requests ────────────────────────────────────────────────────

class TestGenerateRequests:
    def test_yields_all_requests(self):
        requests = [make_request_input(request_id=i) for i in range(5)]

        async def _collect():
            collected = []
            async for r in generate_requests(requests, float("inf")):
                collected.append(r)
            return collected

        result = asyncio.run(_collect())
        assert len(result) == 5

    def test_inf_rate_no_delay(self):
        """At infinite rate, should yield immediately."""
        requests = [make_request_input(request_id=i) for i in range(3)]

        async def _collect():
            import time
            start = time.perf_counter()
            async for _ in generate_requests(requests, float("inf")):
                pass
            return time.perf_counter() - start

        elapsed = asyncio.run(_collect())
        assert elapsed < 0.1  # Should be near-instant

    def test_preserves_request_ids(self):
        requests = [make_request_input(request_id=i) for i in range(5)]

        async def _collect():
            return [r.request_id async for r in generate_requests(requests, float("inf"))]

        ids = asyncio.run(_collect())
        assert ids == [0, 1, 2, 3, 4]


# ── load_text_dataset_with_length ────────────────────────────────────────

class TestLoadTextDatasetWithLength:
    def test_returns_correct_count(self):
        requests = load_text_dataset_with_length(10, 1024)
        assert len(requests) == 10

    def test_returns_request_input_objects(self):
        requests = load_text_dataset_with_length(3, 512)
        for r in requests:
            assert isinstance(r, RequestInput)
            assert r.image_url is None
            assert r.prompt_length > 0

    def test_prompts_approximate_target_length(self):
        target = 2048
        requests = load_text_dataset_with_length(5, target)
        for r in requests:
            # Approximate: 1 token ~ 4 chars, allow 50% tolerance
            char_len = len(r.prompt)
            expected_chars = target * 4
            assert char_len >= expected_chars * 0.5
            assert char_len <= expected_chars * 1.5

    def test_different_lengths_produce_different_prompts(self):
        r1 = load_text_dataset_with_length(1, 512)
        r2 = load_text_dataset_with_length(1, 8192)
        assert len(r1[0].prompt) < len(r2[0].prompt)

    def test_unique_request_ids(self):
        requests = load_text_dataset_with_length(20, 1024)
        ids = [r.request_id for r in requests]
        assert len(set(ids)) == len(ids)


# ── count_tokens_starmie ─────────────────────────────────────────────────

class TestCountTokens:
    def test_returns_positive_int(self):
        count = count_tokens_starmie("Hello world, this is a test")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string(self):
        count = count_tokens_starmie("")
        assert count == 0

    def test_longer_text_more_tokens(self):
        short = count_tokens_starmie("Hello")
        long = count_tokens_starmie("Hello " * 100)
        assert long > short


# ── Queue depth inference ─────────────────────────────────────────────────

def _make_output(request_id, send_time, first_token_time, e2e_latency):
    """Helper: build a RequestOutput with absolute timestamps."""
    return RequestOutput(
        request_id=request_id,
        success=True,
        ttft=first_token_time - send_time,
        e2e_latency=e2e_latency,
        send_time=send_time,
        first_token_time=first_token_time,
        output_tokens=50,
        prompt_tokens=100,
        completion_tokens=50,
    )


class TestInferQueueDepths:
    """Tests for infer_queue_depths — deriving queue depth from timestamps."""

    def test_single_request_zero_queue(self):
        """A single request should see queue depth 0."""
        outputs = [_make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=10.0)]
        result = infer_queue_depths(outputs)
        assert len(result) == 1
        assert result[0]["queue_depth_at_send"] == 0
        assert result[0]["decoding_at_send"] == 0

    def test_two_sequential_requests(self):
        """Two requests that don't overlap: second sees queue 0."""
        outputs = [
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=5.0),
            _make_output(1, send_time=110.0, first_token_time=110.5, e2e_latency=5.0),
        ]
        result = infer_queue_depths(outputs)
        assert result[0]["queue_depth_at_send"] == 0
        assert result[1]["queue_depth_at_send"] == 0
        assert result[1]["decoding_at_send"] == 0  # first already finished

    def test_two_concurrent_requests_both_prefilling(self):
        """Two requests sent at the same time, both still prefilling."""
        # Both sent at t=100, first tokens at t=101 and t=102
        outputs = [
            _make_output(0, send_time=100.0, first_token_time=101.0, e2e_latency=10.0),
            _make_output(1, send_time=100.0, first_token_time=102.0, e2e_latency=10.0),
        ]
        result = infer_queue_depths(outputs)
        # First request (by send_time, ties broken by order): sees nobody ahead
        assert result[0]["queue_depth_at_send"] == 0
        # Second request: first request's first_token_time (101) > second's send_time (100)
        # so first request is still queued/prefilling when second arrives
        assert result[1]["queue_depth_at_send"] == 1

    def test_second_arrives_while_first_decoding(self):
        """Second request arrives while first is actively decoding."""
        outputs = [
            # First: sent at t=100, first token at t=100.5, finishes at t=110
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=10.0),
            # Second: sent at t=105 (while first is decoding)
            _make_output(1, send_time=105.0, first_token_time=105.5, e2e_latency=10.0),
        ]
        result = infer_queue_depths(outputs)
        assert result[1]["queue_depth_at_send"] == 0  # first already got TTFT
        assert result[1]["decoding_at_send"] == 1      # first is still decoding

    def test_mixed_queue_and_decode(self):
        """Three concurrent requests: some queued, some decoding."""
        outputs = [
            # Req 0: sent t=100, TTFT at t=100.5, finishes t=115
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=15.0),
            # Req 1: sent t=100, TTFT at t=103 (waited behind req 0), finishes t=115
            _make_output(1, send_time=100.0, first_token_time=103.0, e2e_latency=15.0),
            # Req 2: sent t=102 (while req 0 is decoding, req 1 still prefilling)
            _make_output(2, send_time=102.0, first_token_time=104.0, e2e_latency=12.0),
        ]
        result = infer_queue_depths(outputs)
        # Req 2 at send_time=102:
        #   Req 0: first_token_time=100.5 < 102, e2e ends at 115 > 102 → decoding
        #   Req 1: first_token_time=103 > 102 → still queued
        r2 = next(r for r in result if r["request_id"] == 2)
        assert r2["queue_depth_at_send"] == 1   # req 1 still prefilling
        assert r2["decoding_at_send"] == 1      # req 0 is decoding

    def test_many_simultaneous_requests(self):
        """N requests sent at the same time — request i should see i-1 queued."""
        N = 8
        outputs = [
            _make_output(
                i,
                send_time=100.0,
                first_token_time=100.5 + i * 0.5,  # staggered TTFT
                e2e_latency=20.0,
            )
            for i in range(N)
        ]
        result = infer_queue_depths(outputs)
        # Sort by first_token_time order (same as send order since all sent at t=100)
        for i, entry in enumerate(result):
            # For request i, all requests j>i have first_token_time > send_time(i)=100
            # But only requests j<i are "earlier" in the loop, so:
            # queue_depth = number of j<i whose first_token_time > 100.0
            # Since all requests have first_token_time > 100.0, all j<i are queued
            assert entry["queue_depth_at_send"] == i

    def test_empty_outputs(self):
        assert infer_queue_depths([]) == []

    def test_failures_excluded(self):
        """Failed requests should be excluded from inference."""
        outputs = [
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=10.0),
            RequestOutput(request_id=1, success=False, send_time=100.0),
        ]
        result = infer_queue_depths(outputs)
        assert len(result) == 1
        assert result[0]["request_id"] == 0

    def test_zero_timestamps_excluded(self):
        """Requests with zero send_time or first_token_time should be excluded."""
        outputs = [
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=10.0),
            RequestOutput(request_id=1, success=True, send_time=0.0, first_token_time=0.0),
        ]
        result = infer_queue_depths(outputs)
        assert len(result) == 1

    def test_ttft_matches_computed(self):
        """The ttft field in results should match first_token_time - send_time."""
        outputs = [_make_output(0, send_time=100.0, first_token_time=100.3, e2e_latency=10.0)]
        result = infer_queue_depths(outputs)
        assert result[0]["ttft"] == pytest.approx(0.3, abs=0.001)

    def test_sorted_by_send_time(self):
        """Results should be sorted by send_time regardless of input order."""
        outputs = [
            _make_output(1, send_time=105.0, first_token_time=105.5, e2e_latency=10.0),
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=10.0),
        ]
        result = infer_queue_depths(outputs)
        assert result[0]["request_id"] == 0
        assert result[1]["request_id"] == 1

    def test_request_finished_before_next_sent(self):
        """If request 0 finishes entirely before request 1 is sent, decoding=0."""
        outputs = [
            _make_output(0, send_time=100.0, first_token_time=100.5, e2e_latency=2.0),  # ends at 102
            _make_output(1, send_time=103.0, first_token_time=103.5, e2e_latency=2.0),  # sent after 102
        ]
        result = infer_queue_depths(outputs)
        r1 = result[1]
        assert r1["queue_depth_at_send"] == 0
        assert r1["decoding_at_send"] == 0

    def test_probe_under_load_scenario(self):
        """Simulate probe-under-load: 4 backgrounds decoding, 1 probe arrives."""
        outputs = []
        # 4 background requests: sent at t=1, TTFT at t=1.5+, decoding until t=31
        for i in range(4):
            outputs.append(_make_output(
                i, send_time=1.0, first_token_time=1.5 + i * 0.1, e2e_latency=30.0
            ))
        # Probe arrives at t=5 (all backgrounds are decoding)
        outputs.append(_make_output(
            4, send_time=5.0, first_token_time=5.8, e2e_latency=10.0
        ))

        result = infer_queue_depths(outputs)
        probe = next(r for r in result if r["request_id"] == 4)
        assert probe["queue_depth_at_send"] == 0   # all backgrounds already got TTFT
        assert probe["decoding_at_send"] == 4       # all 4 are actively decoding


class TestSummarizeByQueueDepth:
    """Tests for summarize_by_queue_depth — grouping inferred data into buckets."""

    def test_empty_input(self):
        assert summarize_by_queue_depth([]) == []

    def test_single_bucket(self):
        """All requests at same in-flight count → one bucket."""
        inferred = [
            {"request_id": 0, "send_time": 100, "first_token_time": 101,
             "ttft": 1.0, "queue_depth_at_send": 0, "decoding_at_send": 3},
            {"request_id": 1, "send_time": 100, "first_token_time": 101.5,
             "ttft": 1.5, "queue_depth_at_send": 1, "decoding_at_send": 2},
        ]
        result = summarize_by_queue_depth(inferred)
        assert len(result) == 1
        assert result[0]["in_flight"] == 3
        assert result[0]["count"] == 2
        assert result[0]["avg_ttft"] == pytest.approx(1.25)

    def test_multiple_buckets(self):
        """Requests at different in-flight counts → separate buckets."""
        inferred = [
            {"request_id": 0, "send_time": 100, "first_token_time": 100.5,
             "ttft": 0.5, "queue_depth_at_send": 0, "decoding_at_send": 0},
            {"request_id": 1, "send_time": 100, "first_token_time": 101,
             "ttft": 1.0, "queue_depth_at_send": 0, "decoding_at_send": 4},
            {"request_id": 2, "send_time": 100, "first_token_time": 102,
             "ttft": 2.0, "queue_depth_at_send": 0, "decoding_at_send": 8},
        ]
        result = summarize_by_queue_depth(inferred)
        assert len(result) == 3
        # Sorted by in_flight
        assert result[0]["in_flight"] == 0
        assert result[1]["in_flight"] == 4
        assert result[2]["in_flight"] == 8
        assert result[0]["avg_ttft"] == pytest.approx(0.5)
        assert result[2]["avg_ttft"] == pytest.approx(2.0)

    def test_buckets_sorted_by_in_flight(self):
        """Output should be sorted by in_flight count ascending."""
        inferred = [
            {"request_id": 0, "ttft": 2.0, "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 5, "decoding_at_send": 5},
            {"request_id": 1, "ttft": 0.5, "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 0, "decoding_at_send": 0},
            {"request_id": 2, "ttft": 1.0, "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 1, "decoding_at_send": 2},
        ]
        result = summarize_by_queue_depth(inferred)
        in_flights = [r["in_flight"] for r in result]
        assert in_flights == sorted(in_flights)

    def test_avg_queued_and_decoding_tracked(self):
        """Verify avg_queued and avg_decoding are computed per bucket."""
        inferred = [
            {"request_id": 0, "ttft": 1.0, "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 2, "decoding_at_send": 6},
            {"request_id": 1, "ttft": 1.2, "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 4, "decoding_at_send": 4},
        ]
        result = summarize_by_queue_depth(inferred)
        assert len(result) == 1  # both have in_flight=8
        assert result[0]["avg_queued"] == pytest.approx(3.0)
        assert result[0]["avg_decoding"] == pytest.approx(5.0)

    def test_higher_in_flight_has_higher_ttft(self):
        """Sanity: busier system should correlate with higher TTFT."""
        inferred = [
            {"request_id": i, "ttft": 0.5 + i * 0.3,
             "send_time": 0, "first_token_time": 0,
             "queue_depth_at_send": 0, "decoding_at_send": i * 4}
            for i in range(5)
        ]
        result = summarize_by_queue_depth(inferred)
        ttfts = [r["avg_ttft"] for r in result]
        assert ttfts == sorted(ttfts)  # monotonically increasing

    def test_integration_with_infer(self):
        """End-to-end: infer_queue_depths → summarize_by_queue_depth."""
        # 8 simultaneous requests, staggered TTFT
        outputs = [
            _make_output(
                i,
                send_time=100.0,
                first_token_time=100.5 + i * 0.3,
                e2e_latency=20.0,
            )
            for i in range(8)
        ]
        inferred = infer_queue_depths(outputs)
        summary = summarize_by_queue_depth(inferred)

        assert len(summary) > 0
        # First request sees 0 in-flight, last sees 7
        assert summary[0]["in_flight"] == 0
        assert summary[-1]["in_flight"] == 7
        # TTFT should increase with in_flight
        ttfts = [s["avg_ttft"] for s in summary]
        assert ttfts == sorted(ttfts)
