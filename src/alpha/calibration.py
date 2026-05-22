"""
Model Calibration — Brier Score Tracking and Probability Shrinkage.

Tracks prediction accuracy across all signal types. Applies calibration
correction so the system doesn't bet full-Kelly on an overconfident model.

Brier Score: BS = (1/N) * Σ(p_i - o_i)²
    - 0.00 = perfect (predicts 1.0 for all winners, 0.0 for all losers)
    - 0.25 = uninformative (always predict 0.50)
    - 0.50 = maximally wrong

A calibrated forecaster with BS = 0.20 is doing well.
An overconfident model saying "90%" when it wins only 70% has BS = 0.04 * 30 + 0.64 * 70 >> uninformative.

Calibration adjustment (Beta shrinkage toward 0.5):
    calibrated_p = weight * raw_p + (1 - weight) * 0.5
    weight = max(CALIBRATION_SHRINKAGE_MIN, 1 - brier_excess)
    brier_excess = max(0, BS - 0.25)   # how much worse than random

Usage:
    tracker = CalibrationTracker()
    # When signal generated:
    tracker.record_prediction(signal_id, alpha_type, predicted_prob, side)
    # When market resolves:
    tracker.record_outcome(market_id, side, won=True)
    # Before placing a trade:
    calibrated = tracker.calibrate(alpha_type, raw_probability)
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.constants import CALIBRATION_MIN_SAMPLES, CALIBRATION_SHRINKAGE_MIN
from src.core.models import AlphaType, Side

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    signal_id: str
    market_id: str
    alpha_type: AlphaType
    side: Side
    predicted_prob: float
    generated_at: datetime
    resolved: bool = False
    outcome: Optional[bool] = None     # True = won, False = lost
    resolved_at: Optional[datetime] = None


@dataclass
class CalibrationStats:
    alpha_type: AlphaType
    n_total: int = 0
    n_resolved: int = 0
    brier_score: float = 0.25       # start uninformative until we have data
    calibration_weight: float = 1.0  # starts at full confidence
    win_rate: float = 0.50
    mean_predicted: float = 0.50


class CalibrationTracker:
    """
    In-memory calibration store. Persists through the session; cleared on restart.
    For production, back this with the SQLite database for cross-session learning.
    """

    def __init__(self) -> None:
        # signal_id -> PredictionRecord
        self._predictions: dict[str, PredictionRecord] = {}
        # market_id -> (side, outcome) — index for fast lookup when market resolves
        self._market_index: dict[str, list[str]] = defaultdict(list)
        # alpha_type -> cached CalibrationStats
        self._stats_cache: dict[AlphaType, CalibrationStats] = {}
        self._dirty = True  # recompute stats on next access

    # ---------------------------------------------------------------- Recording

    def record_prediction(
        self,
        signal_id: str,
        market_id: str,
        alpha_type: AlphaType,
        side: Side,
        predicted_prob: float,
    ) -> None:
        """Record a new prediction when a signal is generated."""
        rec = PredictionRecord(
            signal_id=signal_id,
            market_id=market_id,
            alpha_type=alpha_type,
            side=side,
            predicted_prob=predicted_prob,
            generated_at=datetime.now(timezone.utc),
        )
        self._predictions[signal_id] = rec
        self._market_index[market_id].append(signal_id)
        self._dirty = True
        logger.debug(
            "Calibration: recorded prediction %s p=%.2f for %s/%s",
            signal_id[:8], predicted_prob, alpha_type.value, side.value,
        )

    def record_outcome(
        self,
        market_id: str,
        winning_side: Side,
    ) -> int:
        """
        Record a market resolution. Returns the number of predictions updated.
        Call this when a market resolves (win/loss determined).
        """
        signal_ids = self._market_index.get(market_id, [])
        updated = 0
        for sid in signal_ids:
            rec = self._predictions.get(sid)
            if rec and not rec.resolved:
                rec.resolved = True
                rec.outcome = rec.side == winning_side
                rec.resolved_at = datetime.now(timezone.utc)
                updated += 1
        if updated:
            self._dirty = True
            logger.info(
                "Calibration: resolved %d predictions for market %s (winner=%s)",
                updated, market_id[:16], winning_side.value,
            )
        return updated

    # ---------------------------------------------------------------- Calibration

    def calibrate(
        self,
        alpha_type: AlphaType,
        raw_probability: float,
    ) -> float:
        """
        Apply calibration adjustment to a raw model probability.

        If we don't have enough resolved data for this strategy type,
        returns the raw probability unchanged.
        """
        stats = self.get_stats(alpha_type)
        if stats.n_resolved < CALIBRATION_MIN_SAMPLES:
            return raw_probability  # not enough data to calibrate

        weight = stats.calibration_weight
        calibrated = weight * raw_probability + (1.0 - weight) * 0.5
        return max(0.03, min(0.97, calibrated))

    def calibrate_edge(
        self,
        alpha_type: AlphaType,
        raw_edge: float,
        implied_prob: float,
    ) -> float:
        """
        Adjust edge after calibration. Shrinks the edge in proportion to
        calibration weight — a poorly-calibrated model's edges are smaller.
        """
        stats = self.get_stats(alpha_type)
        if stats.n_resolved < CALIBRATION_MIN_SAMPLES:
            return raw_edge
        return raw_edge * stats.calibration_weight

    # ---------------------------------------------------------------- Stats

    def get_stats(self, alpha_type: AlphaType) -> CalibrationStats:
        """Return (possibly cached) calibration stats for a strategy type."""
        if self._dirty or alpha_type not in self._stats_cache:
            self._recompute_stats()
        return self._stats_cache.get(alpha_type, CalibrationStats(alpha_type=alpha_type))

    def get_all_stats(self) -> list[CalibrationStats]:
        """Return calibration stats for all strategy types with data."""
        if self._dirty:
            self._recompute_stats()
        return list(self._stats_cache.values())

    def summary(self) -> dict[str, object]:
        """Return a summary dict for dashboard display."""
        stats = self.get_all_stats()
        return {
            "total_predictions": len(self._predictions),
            "total_resolved": sum(1 for p in self._predictions.values() if p.resolved),
            "by_strategy": {
                s.alpha_type.value: {
                    "n": s.n_resolved,
                    "brier": round(s.brier_score, 4),
                    "win_rate": round(s.win_rate, 3),
                    "weight": round(s.calibration_weight, 3),
                }
                for s in stats
                if s.n_resolved >= CALIBRATION_MIN_SAMPLES
            },
        }

    # ---------------------------------------------------------------- Internals

    def _recompute_stats(self) -> None:
        """Recompute Brier scores and calibration weights for all strategy types."""
        # Group resolved predictions by alpha_type
        by_type: dict[AlphaType, list[PredictionRecord]] = defaultdict(list)
        for rec in self._predictions.values():
            if rec.resolved and rec.outcome is not None:
                by_type[rec.alpha_type].append(rec)

        new_cache: dict[AlphaType, CalibrationStats] = {}
        for alpha_type, records in by_type.items():
            stats = self._compute_single_stats(alpha_type, records)
            new_cache[alpha_type] = stats

        self._stats_cache = new_cache
        self._dirty = False

    @staticmethod
    def _compute_single_stats(
        alpha_type: AlphaType,
        records: list[PredictionRecord],
    ) -> CalibrationStats:
        n = len(records)
        if n == 0:
            return CalibrationStats(alpha_type=alpha_type)

        brier_sum = 0.0
        wins = 0
        prob_sum = 0.0
        for rec in records:
            outcome_val = 1.0 if rec.outcome else 0.0
            brier_sum += (rec.predicted_prob - outcome_val) ** 2
            if rec.outcome:
                wins += 1
            prob_sum += rec.predicted_prob

        brier = brier_sum / n
        win_rate = wins / n
        mean_pred = prob_sum / n

        # Calibration weight:
        # If BS > 0.25 (worse than random), shrink heavily
        # If BS = 0.10 (skilled), apply small shrinkage
        brier_excess = max(0.0, brier - 0.20)  # excess over 0.20 "good" threshold
        weight = max(CALIBRATION_SHRINKAGE_MIN, 1.0 - brier_excess * 2.0)

        return CalibrationStats(
            alpha_type=alpha_type,
            n_total=n,
            n_resolved=n,
            brier_score=brier,
            calibration_weight=weight,
            win_rate=win_rate,
            mean_predicted=mean_pred,
        )


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all strategy engines
# ---------------------------------------------------------------------------

_global_tracker: Optional[CalibrationTracker] = None


def get_calibration_tracker() -> CalibrationTracker:
    """Return (or create) the global calibration tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CalibrationTracker()
    return _global_tracker


# ---------------------------------------------------------------------------
# Standalone Brier score utilities (for testing / analysis)
# ---------------------------------------------------------------------------

def brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """
    Compute Brier score for a list of probability predictions vs binary outcomes.
    Lower = better. 0.25 = random.
    """
    if not predictions or len(predictions) != len(outcomes):
        return 0.25
    return sum(
        (p - (1.0 if o else 0.0)) ** 2
        for p, o in zip(predictions, outcomes)
    ) / len(predictions)


def reliability_diagram_buckets(
    predictions: list[float],
    outcomes: list[bool],
    n_buckets: int = 10,
) -> list[dict[str, float]]:
    """
    Compute data for a reliability/calibration diagram.
    Each bucket shows the mean predicted probability vs actual win rate.

    A perfectly calibrated model has all points on the diagonal.
    """
    buckets: list[dict[str, float]] = []
    bucket_size = 1.0 / n_buckets
    for i in range(n_buckets):
        lo = i * bucket_size
        hi = (i + 1) * bucket_size
        in_bucket = [
            (p, o)
            for p, o in zip(predictions, outcomes)
            if lo <= p < hi
        ]
        if not in_bucket:
            continue
        probs, outs = zip(*in_bucket)
        buckets.append({
            "bucket_lo": round(lo, 2),
            "bucket_hi": round(hi, 2),
            "n": len(in_bucket),
            "mean_predicted": sum(probs) / len(probs),
            "actual_win_rate": sum(1 for o in outs if o) / len(outs),
        })
    return buckets


def expected_calibration_error(
    predictions: list[float],
    outcomes: list[bool],
    n_buckets: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — weighted average deviation
    of predicted vs actual probability across bins.
    Lower = better calibrated.
    """
    buckets = reliability_diagram_buckets(predictions, outcomes, n_buckets)
    n_total = len(predictions)
    if n_total == 0:
        return 0.0
    ece = sum(
        (b["n"] / n_total) * abs(b["mean_predicted"] - b["actual_win_rate"])
        for b in buckets
    )
    return ece
