"""Unit tests for calibration tracking and Brier score utilities."""
from __future__ import annotations

import pytest

from src.alpha.calibration import (
    CalibrationTracker,
    brier_score,
    expected_calibration_error,
    get_calibration_tracker,
    reliability_diagram_buckets,
)
from src.core.constants import CALIBRATION_MIN_SAMPLES, CALIBRATION_SHRINKAGE_MIN
from src.core.models import AlphaType, Side


def _resolve_all(tracker, market_id, winning_side, prefix="sig"):
    """Helper: resolve all predictions for a market."""
    return tracker.record_outcome(market_id, winning_side)


class TestBrierScore:
    def test_perfect_score(self):
        preds = [1.0, 1.0, 0.0, 0.0]
        outcomes = [True, True, False, False]
        assert brier_score(preds, outcomes) == 0.0

    def test_worst_score(self):
        preds = [0.0, 1.0]
        outcomes = [True, False]
        assert abs(brier_score(preds, outcomes) - 1.0) < 1e-9

    def test_uninformative(self):
        preds = [0.5] * 100
        outcomes = [True] * 50 + [False] * 50
        bs = brier_score(preds, outcomes)
        assert abs(bs - 0.25) < 0.01

    def test_empty_returns_uninformative(self):
        assert brier_score([], []) == 0.25

    def test_skilled_forecaster(self):
        # 70% predictions that come true 70% of the time
        preds = [0.7] * 70 + [0.3] * 30
        outcomes = [True] * 70 + [False] * 30
        bs = brier_score(preds, outcomes)
        assert bs < 0.25  # better than random


class TestReliabilityDiagram:
    def test_returns_buckets(self):
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [False, False, True, True, True]
        buckets = reliability_diagram_buckets(preds, outcomes, n_buckets=5)
        assert len(buckets) > 0
        for b in buckets:
            assert "bucket_lo" in b
            assert "actual_win_rate" in b
            assert 0.0 <= b["actual_win_rate"] <= 1.0

    def test_empty_returns_no_buckets(self):
        assert reliability_diagram_buckets([], []) == []


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_zero_ece(self):
        # Each bucket: predicted = actual win rate
        preds = [0.1, 0.1, 0.9, 0.9]
        outcomes = [False, False, True, True]
        ece = expected_calibration_error(preds, outcomes, n_buckets=10)
        assert ece < 0.15

    def test_overconfident_nonzero_ece(self):
        # Always predict 90% but only win 50% of the time
        preds = [0.9] * 50
        outcomes = [True] * 25 + [False] * 25
        ece = expected_calibration_error(preds, outcomes)
        assert ece > 0.0

    def test_empty(self):
        assert expected_calibration_error([], []) == 0.0


class TestCalibrationTracker:
    def setup_method(self):
        self.tracker = CalibrationTracker()

    def test_record_and_resolve(self):
        self.tracker.record_prediction(
            "sig1", "market1", AlphaType.EV_DIRECTIONAL, Side.YES, 0.70
        )
        updated = self.tracker.record_outcome("market1", Side.YES)
        assert updated == 1

    def test_no_calibration_before_min_samples(self):
        # Only 3 predictions — below CALIBRATION_MIN_SAMPLES
        for i in range(3):
            self.tracker.record_prediction(
                f"sig{i}", "m1", AlphaType.EV_DIRECTIONAL, Side.YES, 0.80
            )
            self.tracker.record_outcome("m1", Side.YES)

        # Should return raw probability unchanged
        assert self.tracker.calibrate(AlphaType.EV_DIRECTIONAL, 0.80) == 0.80

    def test_calibration_applied_after_min_samples(self):
        # Record enough wrong predictions to force calibration
        n = CALIBRATION_MIN_SAMPLES + 2
        for i in range(n):
            mid = f"market_{i}"
            self.tracker.record_prediction(
                f"sig_{i}", mid, AlphaType.EV_DIRECTIONAL, Side.YES, 0.90
            )
            # Model predicted 90% but loses half the time → overconfident
            winning = Side.YES if i % 2 == 0 else Side.NO
            self.tracker.record_outcome(mid, winning)

        # Calibrated probability should be shrunk toward 0.5
        raw = 0.90
        calibrated = self.tracker.calibrate(AlphaType.EV_DIRECTIONAL, raw)
        assert calibrated < raw  # must be shrunk
        assert calibrated >= 0.03

    def test_calibration_weight_high_brier_low_weight(self):
        # Force high Brier score: always predict 0.9, always wrong
        n = CALIBRATION_MIN_SAMPLES + 2
        for i in range(n):
            mid = f"m_{i}"
            self.tracker.record_prediction(
                f"s_{i}", mid, AlphaType.MEAN_REVERSION, Side.YES, 0.90
            )
            # Always lose
            self.tracker.record_outcome(mid, Side.NO)

        stats = self.tracker.get_stats(AlphaType.MEAN_REVERSION)
        assert stats.brier_score > 0.25
        assert stats.calibration_weight <= 1.0
        assert stats.calibration_weight >= CALIBRATION_SHRINKAGE_MIN

    def test_calibration_weight_perfect_forecaster(self):
        # Always predict p=0.8 and always win → low Brier → high weight
        n = CALIBRATION_MIN_SAMPLES + 2
        for i in range(n):
            mid = f"perf_{i}"
            self.tracker.record_prediction(
                f"ps_{i}", mid, AlphaType.TIME_DECAY, Side.YES, 0.80
            )
            self.tracker.record_outcome(mid, Side.YES)

        stats = self.tracker.get_stats(AlphaType.TIME_DECAY)
        # Brier = (0.8-1)^2 per win = 0.04 → weight should be near 1.0
        assert stats.brier_score < 0.20
        assert stats.calibration_weight > 0.80

    def test_unresolved_predictions_not_counted(self):
        self.tracker.record_prediction(
            "sig_open", "market_open", AlphaType.ORDER_FLOW, Side.YES, 0.65
        )
        stats = self.tracker.get_stats(AlphaType.ORDER_FLOW)
        assert stats.n_resolved == 0
        # No calibration without resolved predictions
        assert self.tracker.calibrate(AlphaType.ORDER_FLOW, 0.65) == 0.65

    def test_record_outcome_returns_count(self):
        for i in range(3):
            self.tracker.record_prediction(
                f"s{i}", "shared_market", AlphaType.EV_DIRECTIONAL, Side.YES, 0.70
            )
        count = self.tracker.record_outcome("shared_market", Side.YES)
        assert count == 3

    def test_duplicate_resolution_ignored(self):
        self.tracker.record_prediction(
            "s1", "dup_market", AlphaType.EV_DIRECTIONAL, Side.YES, 0.70
        )
        self.tracker.record_outcome("dup_market", Side.YES)
        # Second resolution should not double-count
        count = self.tracker.record_outcome("dup_market", Side.YES)
        assert count == 0

    def test_summary_structure(self):
        summary = self.tracker.summary()
        assert "total_predictions" in summary
        assert "total_resolved" in summary
        assert "by_strategy" in summary

    def test_calibrate_edge_reduces_with_low_weight(self):
        # Set up low-weight model
        n = CALIBRATION_MIN_SAMPLES + 2
        for i in range(n):
            mid = f"em_{i}"
            self.tracker.record_prediction(
                f"es_{i}", mid, AlphaType.CROSS_EXCHANGE_ARB, Side.YES, 0.90
            )
            self.tracker.record_outcome(mid, Side.NO)  # always wrong

        raw_edge = 0.10
        adjusted = self.tracker.calibrate_edge(AlphaType.CROSS_EXCHANGE_ARB, raw_edge, 0.50)
        assert adjusted <= raw_edge


class TestGlobalTracker:
    def test_singleton(self):
        t1 = get_calibration_tracker()
        t2 = get_calibration_tracker()
        assert t1 is t2
