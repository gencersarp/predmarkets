"""Unit tests for the TrendFollowingDetector."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from src.alpha.trend_following import TrendFollowingDetector
from tests.conftest import make_market

@pytest.fixture
def detector():
    return TrendFollowingDetector()

def test_trend_uptrend(detector):
    market = make_market(market_id="m1")
    # Feed a clear uptrend with increasing volume
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    prices = [0.50] * 20 + [0.60, 0.70, 0.85, 0.95]
    vols = [1000] * 20 + [2000, 3000, 4000, 5000]
    
    for i, (p, v) in enumerate(zip(prices, vols)):
        detector.update("m1", p, v)
        detector._price_histories["m1"].timestamps[i] = base_time + timedelta(minutes=i)

    signal = detector.detect_trend(market)
    assert signal is not None
    assert signal.side.value == "yes"
    assert signal.edge > 0

def test_trend_no_signal_sideways(detector):
    market = make_market(market_id="m2")
    # Feed sideways price
    for i in range(30):
        detector.update("m2", 0.50, 1000)

    signal = detector.detect_trend(market)
    assert signal is None

def test_trend_downtrend(detector):
    market = make_market(market_id="m3")
    # Feed a clear downtrend
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    prices = [0.50] * 20 + [0.40, 0.30, 0.15, 0.05]
    vols = [1000] * 20 + [2000, 3000, 4000, 5000]
    
    for i, (p, v) in enumerate(zip(prices, vols)):
        detector.update("m3", p, v)
        detector._price_histories["m3"].timestamps[i] = base_time + timedelta(minutes=i)

    signal = detector.detect_trend(market)
    assert signal is not None
    assert signal.side.value == "no"
