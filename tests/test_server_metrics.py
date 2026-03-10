"""Tests for server_metrics.py — Prometheus parsing and metric calculation."""

import pytest
from server_metrics import (
    parse_prometheus_metrics,
    get_labeled_metric_value,
    calculate_percentiles_from_histogram,
    total_from_cumhist_delta,
)


# ── parse_prometheus_metrics ─────────────────────────────────────────────

class TestParsePrometheusMetrics:
    def test_simple_metric(self):
        text = "my_metric 42.5\n"
        metrics, histograms, labeled = parse_prometheus_metrics(text)
        assert metrics["my_metric"] == 42.5
        assert len(histograms) == 0

    def test_multiple_metrics(self):
        text = "metric_a 10\nmetric_b 20\nmetric_c 30\n"
        metrics, _, _ = parse_prometheus_metrics(text)
        assert metrics["metric_a"] == 10
        assert metrics["metric_b"] == 20
        assert metrics["metric_c"] == 30

    def test_ignores_comments(self):
        text = "# HELP my_metric Description\n# TYPE my_metric gauge\nmy_metric 5\n"
        metrics, _, _ = parse_prometheus_metrics(text)
        assert metrics["my_metric"] == 5
        assert len(metrics) == 1

    def test_ignores_empty_lines(self):
        text = "\n\nmy_metric 5\n\n"
        metrics, _, _ = parse_prometheus_metrics(text)
        assert metrics["my_metric"] == 5

    def test_histogram_buckets(self):
        text = (
            'my_hist_bucket{le="0.1"} 5\n'
            'my_hist_bucket{le="0.5"} 15\n'
            'my_hist_bucket{le="1.0"} 25\n'
        )
        metrics, histograms, _ = parse_prometheus_metrics(text)
        assert "my_hist" in histograms
        buckets = histograms["my_hist"]
        assert buckets[0.1] == 5
        assert buckets[0.5] == 15
        assert buckets[1.0] == 25

    def test_labeled_metrics(self):
        text = 'http_requests{method="GET",code="200"} 100\nhttp_requests{method="POST",code="200"} 50\n'
        metrics, _, labeled = parse_prometheus_metrics(text)
        assert "http_requests" in labeled
        # Total should be summed
        assert metrics["http_requests"] == 150

    def test_scientific_notation(self):
        text = "my_metric 1.5e+03\n"
        metrics, _, _ = parse_prometheus_metrics(text)
        assert metrics["my_metric"] == 1500.0

    def test_inf_bucket_not_captured(self):
        """The +Inf bucket is not currently captured by the histogram parser."""
        text = 'my_hist_bucket{le="+Inf"} 100\n'
        _, histograms, _ = parse_prometheus_metrics(text)
        # +Inf doesn't match the bucket regex — falls through to labeled metrics
        assert "my_hist" not in histograms

    def test_empty_input(self):
        metrics, histograms, labeled = parse_prometheus_metrics("")
        assert len(metrics) == 0
        assert len(histograms) == 0
        assert len(labeled) == 0

    def test_vllm_metrics_format(self):
        text = (
            '# HELP vllm:request_success_total Total requests\n'
            '# TYPE vllm:request_success_total counter\n'
            'vllm:request_success_total{finish_reason="stop"} 42\n'
            'vllm:request_success_total{finish_reason="abort"} 3\n'
            'vllm:prompt_tokens_total 10000\n'
            'vllm:generation_tokens_total 5000\n'
        )
        metrics, _, labeled = parse_prometheus_metrics(text)
        assert metrics["vllm:prompt_tokens_total"] == 10000
        assert metrics["vllm:generation_tokens_total"] == 5000
        assert metrics["vllm:request_success_total"] == 45  # sum of labels


# ── get_labeled_metric_value ─────────────────────────────────────────────

class TestGetLabeledMetricValue:
    def test_filter_by_label(self):
        labeled = {
            "http_requests": {
                (("code", "200"), ("method", "GET")): 100,
                (("code", "200"), ("method", "POST")): 50,
                (("code", "500"), ("method", "GET")): 5,
            }
        }
        assert get_labeled_metric_value(labeled, "http_requests", {"code": "200"}) == 150
        assert get_labeled_metric_value(labeled, "http_requests", {"method": "GET"}) == 105
        assert get_labeled_metric_value(labeled, "http_requests", {"code": "500", "method": "GET"}) == 5

    def test_missing_metric_returns_zero(self):
        assert get_labeled_metric_value({}, "nonexistent", {"k": "v"}) == 0.0

    def test_no_matching_labels(self):
        labeled = {
            "m": {(("a", "1"),): 10},
        }
        assert get_labeled_metric_value(labeled, "m", {"a": "2"}) == 0.0


# ── calculate_percentiles_from_histogram ────────────────────────────────

class TestCalculatePercentilesFromHistogram:
    def test_basic_percentiles(self):
        buckets = {0.1: 10, 0.5: 50, 1.0: 90, float("inf"): 100}
        result = calculate_percentiles_from_histogram(buckets, [50, 95, 99])
        assert 50 in result
        assert 95 in result
        assert 99 in result
        assert result[50] < result[95] <= result[99]

    def test_p50_in_expected_range(self):
        buckets = {0.1: 10, 0.5: 50, 1.0: 90, float("inf"): 100}
        result = calculate_percentiles_from_histogram(buckets, [50])
        # p50 should be around 0.5 (50th percentile)
        assert 0.1 < result[50] < 1.0

    def test_empty_buckets(self):
        result = calculate_percentiles_from_histogram({}, [50])
        assert result[50] == 0.0

    def test_zero_total(self):
        buckets = {0.1: 0, 0.5: 0, float("inf"): 0}
        result = calculate_percentiles_from_histogram(buckets, [50])
        assert result[50] == 0.0

    def test_single_bucket(self):
        buckets = {1.0: 100}
        result = calculate_percentiles_from_histogram(buckets, [50])
        assert result[50] >= 0

    def test_custom_first_lower_bound(self):
        buckets = {0.5: 50, 1.0: 100}
        result = calculate_percentiles_from_histogram(buckets, [0], first_lower_bound=0.0)
        assert result[0] == 0.0

    def test_p100(self):
        buckets = {0.1: 10, 0.5: 50, 1.0: 100}
        result = calculate_percentiles_from_histogram(buckets, [100])
        assert result[100] == 1.0

    def test_non_decreasing_violation_raises(self):
        buckets = {0.1: 50, 0.5: 30}  # decreasing
        with pytest.raises(ValueError, match="non-decreasing"):
            calculate_percentiles_from_histogram(buckets, [50])


# ── total_from_cumhist_delta ─────────────────────────────────────────────

class TestTotalFromCumhistDelta:
    def test_cumulative_delta(self):
        # Cumulative histogram: each bucket is total up to that point
        delta = {0.1: 5, 0.5: 15, 1.0: 25, float("inf"): 30}
        total = total_from_cumhist_delta(delta)
        assert total == 30  # last cumulative value

    def test_empty_returns_zero(self):
        assert total_from_cumhist_delta({}) == 0.0

    def test_non_cumulative_sums_all(self):
        # If counts decrease (non-cumulative), falls back to sum
        delta = {0.1: 10, 0.5: 5, 1.0: 3}
        total = total_from_cumhist_delta(delta)
        assert total == 18  # sum of all
