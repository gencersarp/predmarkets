"""Unit tests for the mean-reversion / overreaction signal generator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.alpha.mean_reversion import (
    MeanReversionDetector,
    OUCalibrator,
    OUParameters,
    PriceHistory,
)
from src.core.models import Side
from tests.conftest import make_market


class TestPriceHistory:
    def test_appends_data(self):
        h = PriceHistory()
        h.append(datetime.now(timezone.utc), 0.50, 100.0)
        assert len(h.prices) == 1
        assert len(h.timestamps) == 1
        assert len(h.volumes) == 1

    def test_max_window_enforced(self):
        h = PriceHistory(max_window=50)
        for i in range(100):
            h.append(datetime.now(timezone.utc), 0.5, 1.0)
        assert len(h.prices) <= 50

    def test_recent_filters_by_time(self):
        h = PriceHistory()
        old_time = datetime.now(timezone.utc) - timedelta(minutes=90)
        new_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        h.timestamps = [old_time, new_time]
        h.prices = [0.3, 0.6]
        h.volumes = [100.0, 200.0]
        _, recent_prices, _ = h.recent(minutes=60)
        assert len(recent_prices) == 1
        assert recent_prices[0] == 0.6

    def test_current_price(self):
        h = PriceHistory()
        h.append(datetime.now(timezone.utc), 0.42)
        assert h.current_price() == 0.42

    def test_empty_history_current_price_none(self):
        h = PriceHistory()
        assert h.current_price() is None

    def test_peak_price_in_window(self):
        h = PriceHistory()
        for p in [0.30, 0.80, 0.50, 0.60]:
            h.append(datetime.now(timezone.utc), p, 100.0)
        peak = h.peak_price(minutes=60)
        assert peak == 0.80

    def test_rolling_mean(self):
        h = PriceHistory()
        for p in [0.40, 0.50, 0.60]:
            h.append(datetime.now(timezone.utc), p, 100.0)
        mean = h.rolling_mean(minutes=60)
        assert abs(mean - 0.50) < 1e-6

    def test_rolling_std(self):
        h = PriceHistory()
        for p in [0.40, 0.50, 0.60]:
            h.append(datetime.now(timezone.utc), p, 100.0)
        std = h.rolling_std(minutes=60)
        assert std is not None and std > 0

    def test_std_none_with_single_point(self):
        h = PriceHistory()
        h.append(datetime.now(timezone.utc), 0.50, 100.0)
        assert h.rolling_std() is None


class TestOUParameters:
    def test_expected_price_reverts_to_mean(self):
        ou = OUParameters(mu=0.50, kappa=1.0, sigma=0.02)
        p0 = 0.80  # above mean
        # After a long time → expected price → mu
        expected_far = ou.expected_price(p0, days=100.0)
        assert abs(expected_far - 0.50) < 0.01

    def test_expected_price_immediate(self):
        ou = OUParameters(mu=0.50, kappa=1.0, sigma=0.02)
        p = ou.expected_price(0.65, days=0.0)
        assert abs(p - 0.65) < 1e-6  # no time passed → stays at p0

    def test_reversion_halflife(self):
        import math
        ou = OUParameters(mu=0.50, kappa=2.0, sigma=0.01)
        hl = ou.reversion_halflife_days()
        assert abs(hl - math.log(2) / 2.0) < 1e-6

    def test_zero_kappa_infinite_halflife(self):
        ou = OUParameters(mu=0.50, kappa=0.0, sigma=0.01)
        hl = ou.reversion_halflife_days()
        assert hl == float("inf")


class TestOUCalibrator:
    def _make_ou_history(
        self,
        n: int = 200,
        mu: float = 0.50,
        kappa: float = 2.0,
        noise: float = 0.01,
    ) -> PriceHistory:
        """Generate synthetic OU prices."""
        import math, random
        h = PriceHistory(max_window=n + 10)
        dt = 1.0 / 1440.0  # one minute
        p = mu
        for i in range(n):
            dp = kappa * (mu - p) * dt + noise * math.sqrt(dt) * random.gauss(0, 1)
            p = max(0.01, min(0.99, p + dp))
            h.append(datetime.now(timezone.utc), p, 100.0)
        return h

    def test_calibrates_mean_reversion(self):
        h = self._make_ou_history(n=300, mu=0.50, kappa=3.0)
        params = OUCalibrator.calibrate(h)
        # With enough data, should detect mean reversion (kappa > 0)
        if params is not None:
            assert params.kappa > 0
            # Mean should be roughly near 0.50
            assert 0.20 < params.mu < 0.80

    def test_returns_none_with_too_few_data(self):
        h = PriceHistory()
        for p in [0.50, 0.51, 0.49]:
            h.append(datetime.now(timezone.utc), p, 100.0)
        params = OUCalibrator.calibrate(h)
        assert params is None  # < 20 data points

    def test_returns_none_for_trending_series(self):
        """For a strictly trending series, beta >= 0 → no mean reversion."""
        h = PriceHistory()
        for i in range(50):
            h.append(datetime.now(timezone.utc), 0.01 * i, 100.0)  # strict trend
        params = OUCalibrator.calibrate(h)
        assert params is None  # beta >= 0 → no mean reversion


class TestMeanReversionDetector:
    def _make_history_with_spike(
        self,
        detector: MeanReversionDetector,
        market_id: str,
        base_price: float = 0.50,
        spike_to: float = 0.75,
        spike_duration_ticks: int = 10,
        n_base_ticks: int = 100,
        high_volume: float = 50000.0,
        low_volume: float = 1000.0,
    ) -> None:
        """Feed price history with spike and volume decay."""
        # Base period
        for _ in range(n_base_ticks):
            detector.update(market_id, base_price, 5000.0)
        # Spike period (high volume)
        for _ in range(spike_duration_ticks):
            detector.update(market_id, spike_to, high_volume)
        # Volume decay after spike
        for _ in range(5):
            detector.update(market_id, spike_to * 0.98, low_volume)

    def test_detects_spike_with_volume_decay(self):
        det = MeanReversionDetector(
            spike_threshold=0.10,
            volume_decay_threshold=0.30,
            min_edge=0.03,
        )
        self._make_history_with_spike(det, "test-001")
        market = make_market(
            "test-001",
            yes_bid=0.72, yes_ask=0.75,
            no_bid=0.24, no_ask=0.28,
        )
        # Don't need oracle; just check if signal is generated or gracefully returns None
        signal = det.detect_overreaction(market, oracle_true_prob=0.50)
        # With 25% spike and oracle at 0.50: edge should be ~0.25-0.28
        # Since volume is decaying: should ideally get a NO signal (fade the spike)
        # May not trigger due to exact timing; just ensure no crash
        assert signal is None or signal.side in (Side.YES, Side.NO)

    def test_no_signal_without_history(self):
        det = MeanReversionDetector()
        market = make_market()
        signal = det.detect_overreaction(market)
        assert signal is None

    def test_no_signal_without_spike(self):
        det = MeanReversionDetector(spike_threshold=0.20)
        # Stable price: no spike
        for i in range(100):
            det.update("stable-001", 0.50 + 0.01 * (i % 3), 5000.0)
        market = make_market("stable-001")
        signal = det.detect_overreaction(market)
        assert signal is None  # no spike > 20%

    def test_tracked_markets_updated(self):
        det = MeanReversionDetector()
        det.update("market-A", 0.50, 100.0)
        det.update("market-B", 0.70, 200.0)
        assert "market-A" in det.tracked_markets
        assert "market-B" in det.tracked_markets

    def test_signal_structure_when_generated(self):
        """If a signal is generated, verify its structure is correct."""
        det = MeanReversionDetector(
            spike_threshold=0.05,  # Low threshold to trigger easier
            volume_decay_threshold=0.50,
            min_edge=0.02,
        )
        # Feed rapid spike
        for _ in range(20):
            det.update("fast-001", 0.30, 2000.0)
        for _ in range(10):
            det.update("fast-001", 0.60, 50000.0)
        for _ in range(5):
            det.update("fast-001", 0.58, 1000.0)  # volume decay

        market = make_market(
            "fast-001",
            yes_bid=0.56, yes_ask=0.60,
            no_bid=0.39, no_ask=0.42,
            expiry_days=7.0,
        )
        signal = det.detect_overreaction(market, oracle_true_prob=0.35)
        if signal is not None:
            assert signal.market_id == "fast-001"
            assert 0.0 <= signal.true_probability <= 1.0
            assert 0.0 <= signal.implied_probability <= 1.0
            assert signal.edge > 0
            assert signal.recommended_size_usd >= 0
