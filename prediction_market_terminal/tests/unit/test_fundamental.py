"""Unit tests for the EV/fundamental directional signal engine."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.alpha.fundamental import EVEngine, compute_edge, compute_ev, decimal_odds_from_price
from src.core.models import AlphaType, Exchange, OracleEstimate, RiskFlag, Side
from tests.conftest import make_market


def _make_estimate(
    source: str,
    prob: float,
    ci_low: float = 0.0,
    ci_high: float = 1.0,
) -> OracleEstimate:
    return OracleEstimate(
        source=source,
        market_id="test-001",
        true_probability=prob,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
        model_name=source,
    )


class TestComputeEV:
    def test_positive_ev(self):
        # p_true=0.70, market_price=0.50 → b=(1/0.5)-1=1.0
        # EV = 0.70*1.0 - 0.30 = 0.40
        ev = compute_ev(true_prob=0.70, market_price=0.50)
        assert abs(ev - 0.40) < 1e-6

    def test_zero_ev_at_fair_price(self):
        # p=0.50, price=0.50 → EV = 0.50*(2-1)-0.50 = 0
        ev = compute_ev(true_prob=0.50, market_price=0.50)
        assert abs(ev) < 1e-6

    def test_negative_ev_when_overpriced(self):
        # p=0.30, price=0.60 → buy YES at price that's too high
        ev = compute_ev(true_prob=0.30, market_price=0.60)
        assert ev < 0

    def test_zero_price_returns_zero(self):
        assert compute_ev(0.6, 0.0) == 0.0

    def test_one_price_returns_zero(self):
        assert compute_ev(0.6, 1.0) == 0.0


class TestComputeEdge:
    def test_positive_edge(self):
        edge = compute_edge(true_prob=0.65, implied_prob=0.50)
        assert abs(edge - 0.15) < 1e-6

    def test_negative_edge_overpriced(self):
        edge = compute_edge(true_prob=0.30, implied_prob=0.60)
        assert edge < 0

    def test_zero_edge_at_fair(self):
        edge = compute_edge(true_prob=0.50, implied_prob=0.50)
        assert edge == 0.0


class TestDecimalOdds:
    def test_fifty_fifty_is_two(self):
        assert abs(decimal_odds_from_price(0.50) - 2.0) < 1e-6

    def test_twenty_five_pct_is_four(self):
        assert abs(decimal_odds_from_price(0.25) - 4.0) < 1e-6

    def test_zero_price_is_infinity(self):
        odds = decimal_odds_from_price(0.0)
        assert odds == float("inf")


class TestEVEngine:
    def test_evaluates_positive_edge(self):
        market = make_market(yes_ask=0.40, no_ask=0.62, volume_24h=100000)
        estimates = [_make_estimate("model_a", 0.65, ci_low=0.58, ci_high=0.72)]
        engine = EVEngine(min_edge_pct=0.05)
        signal = engine.evaluate_market(market, estimates, bankroll_usd=1000.0)
        assert signal is not None
        assert signal.side == Side.YES
        assert signal.edge > 0
        assert signal.expected_value_usd > 0

    def test_no_signal_on_zero_edge(self):
        # Market priced at true probability → no edge
        market = make_market(yes_ask=0.60, no_ask=0.42)
        estimates = [_make_estimate("model_a", 0.60, ci_low=0.55, ci_high=0.65)]
        engine = EVEngine(min_edge_pct=0.05)
        signal = engine.evaluate_market(market, estimates, bankroll_usd=1000.0)
        # edge = 0.60 - 0.60 = 0 < 0.05 → no signal
        assert signal is None

    def test_no_signal_without_oracle(self):
        market = make_market()
        engine = EVEngine()
        signal = engine.evaluate_market(market, [], bankroll_usd=1000.0)
        assert signal is None

    def test_filters_low_confidence_oracle(self):
        # Wide CI → low confidence → filtered out
        market = make_market(yes_ask=0.40)
        estimates = [
            _make_estimate("model_a", 0.70, ci_low=0.10, ci_high=0.90)  # CI width = 0.80
        ]
        engine = EVEngine(min_edge_pct=0.05, min_oracle_confidence=0.50)
        signal = engine.evaluate_market(market, estimates)
        assert signal is None  # CI width 0.80 > 0.50 → filtered

    def test_accepts_narrow_ci_oracle(self):
        market = make_market(yes_ask=0.40, no_ask=0.62, volume_24h=100000)
        estimates = [
            _make_estimate("model_a", 0.65, ci_low=0.60, ci_high=0.70)  # CI width = 0.10
        ]
        engine = EVEngine(min_edge_pct=0.05, min_oracle_confidence=0.50)
        signal = engine.evaluate_market(market, estimates)
        assert signal is not None

    def test_chooses_best_direction(self):
        """If NO has better EV than YES, signal should be for NO."""
        # Market YES at 0.80 (overpriced), NO at 0.15 (cheap)
        market = make_market(
            yes_ask=0.80, no_ask=0.15, yes_bid=0.78, no_bid=0.13, volume_24h=100000
        )
        # True prob YES = 0.50 → NO is the better bet
        estimates = [_make_estimate("model", 0.50, ci_low=0.44, ci_high=0.56)]
        engine = EVEngine(min_edge_pct=0.05)
        signal = engine.evaluate_market(market, estimates)
        if signal is not None:
            assert signal.side == Side.NO

    def test_evaluate_universe(self):
        markets = [
            make_market("m1", yes_ask=0.40, no_ask=0.62, volume_24h=100000),
            make_market("m2", yes_ask=0.52, no_ask=0.50),  # near fair
        ]
        oracle_map = {
            "m1": [_make_estimate("model", 0.65, ci_low=0.60, ci_high=0.70)],
            "m2": [_make_estimate("model", 0.52, ci_low=0.47, ci_high=0.57)],
        }
        engine = EVEngine(min_edge_pct=0.05)
        signals = engine.evaluate_universe(markets, oracle_map, bankroll_usd=1000.0)
        assert isinstance(signals, list)
        # Verify sorted by EV descending
        for i in range(len(signals) - 1):
            assert signals[i].expected_value_usd >= signals[i + 1].expected_value_usd

    def test_signal_kelly_sizing_within_bounds(self):
        market = make_market(yes_ask=0.35, no_ask=0.67, volume_24h=100000)
        estimates = [_make_estimate("model", 0.70, ci_low=0.64, ci_high=0.76)]
        engine = EVEngine(min_edge_pct=0.05)
        signal = engine.evaluate_market(market, estimates, bankroll_usd=1000.0)
        if signal is not None:
            assert 0 <= signal.kelly_fraction_suggested <= 0.25
            assert signal.recommended_size_usd <= 1000.0

    def test_consensus_probability_inverse_ci_weighting(self):
        """Narrower CI → higher weight in consensus."""
        engine = EVEngine()
        estimates = [
            _make_estimate("narrow", 0.80, ci_low=0.78, ci_high=0.82),  # width=0.04, w=25
            _make_estimate("wide", 0.20, ci_low=0.00, ci_high=0.60),   # width=0.60, w=1.67
        ]
        prob = engine._consensus_probability(estimates)
        # Narrow estimate should dominate: consensus closer to 0.80
        assert prob > 0.60

    def test_confidence_increases_with_more_sources(self):
        engine = EVEngine()
        one = [_make_estimate("a", 0.60, 0.55, 0.65)]
        two = [_make_estimate("a", 0.60, 0.55, 0.65), _make_estimate("b", 0.58, 0.53, 0.63)]
        c1 = engine._compute_confidence(one)
        c2 = engine._compute_confidence(two)
        assert c2 > c1


class TestDirectionalSignalIsActionable:
    def test_actionable_with_good_signal(self):
        from tests.conftest import make_directional_signal
        sig = make_directional_signal(edge=0.08, ev_usd=25.0, aroc=1.2)
        assert sig.is_actionable

    def test_not_actionable_with_zero_ev(self):
        from tests.conftest import make_directional_signal
        sig = make_directional_signal(ev_usd=-1.0)
        assert not sig.is_actionable

    def test_not_actionable_with_fee_excessive_flag(self):
        from tests.conftest import make_directional_signal
        sig = make_directional_signal(risk_flags=[RiskFlag.FEE_EXCESSIVE])
        assert not sig.is_actionable

    def test_not_actionable_with_aroc_below_min(self):
        from tests.conftest import make_directional_signal
        sig = make_directional_signal(risk_flags=[RiskFlag.AROC_BELOW_MIN])
        assert not sig.is_actionable

    def test_not_actionable_with_low_edge(self):
        from tests.conftest import make_directional_signal
        sig = make_directional_signal(edge=0.015)  # below 0.02 threshold
        assert not sig.is_actionable
