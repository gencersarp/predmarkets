"""Unit tests for Kelly Criterion sizing module."""
from __future__ import annotations

import math
import pytest

from src.risk.kelly import (
    breakeven_probability,
    confidence_adjusted_kelly,
    drawdown_adjusted_kelly,
    expected_value_per_unit,
    kelly_fraction,
    kelly_growth_rate,
    kelly_position_size_usd,
    portfolio_kelly_adjustment,
    required_edge_after_fees,
    ruin_probability_approximation,
    sizing_report,
)
from src.core.constants import KELLY_MAX_FRACTION


class TestKellyFraction:
    def test_positive_edge(self):
        # p=0.6, b=1.0 (even money odds), fraction=0.25
        # Full Kelly = (1.0*0.6 - 0.4) / 1.0 = 0.2
        # Quarter Kelly = 0.05
        f = kelly_fraction(p_win=0.6, b=1.0, fraction=0.25)
        assert abs(f - 0.05) < 1e-6

    def test_zero_edge_breaks_even(self):
        # p=0.5, b=1.0 → full Kelly = 0 → no bet
        f = kelly_fraction(p_win=0.5, b=1.0, fraction=0.25)
        assert f == 0.0

    def test_negative_edge_no_bet(self):
        # p=0.3, b=1.0 → full Kelly = (0.3-0.7)/1 = -0.4 → 0
        f = kelly_fraction(p_win=0.3, b=1.0, fraction=0.25)
        assert f == 0.0

    def test_zero_probability(self):
        assert kelly_fraction(p_win=0.0, b=2.0, fraction=0.25) == 0.0

    def test_zero_b(self):
        assert kelly_fraction(p_win=0.8, b=0.0, fraction=0.25) == 0.0

    def test_certainty_caps_at_max_fraction(self):
        f = kelly_fraction(p_win=1.0, b=3.0, fraction=0.25)
        assert f == KELLY_MAX_FRACTION

    def test_cap_at_kelly_max(self):
        # Very high edge: Kelly might suggest >25%, should be capped
        f = kelly_fraction(p_win=0.99, b=10.0, fraction=0.25)
        assert f <= KELLY_MAX_FRACTION

    def test_full_kelly_vs_quarter_kelly(self):
        # Use small edge (p=0.52, b=1.0) where full Kelly (0.04) is well below cap
        full = kelly_fraction(p_win=0.52, b=1.0, fraction=1.0)
        quarter = kelly_fraction(p_win=0.52, b=1.0, fraction=0.25)
        assert abs(full - quarter * 4) < 1e-6

    def test_different_fractions(self):
        half = kelly_fraction(p_win=0.6, b=1.5, fraction=0.5)
        quarter = kelly_fraction(p_win=0.6, b=1.5, fraction=0.25)
        assert abs(half - quarter * 2) < 1e-6


class TestKellyPositionSize:
    def test_basic_sizing(self):
        size = kelly_position_size_usd(
            p_win=0.60,
            market_price=0.50,
            bankroll_usd=1000.0,
            fraction=0.25,
        )
        assert size > 0

    def test_scales_with_bankroll(self):
        s1 = kelly_position_size_usd(0.65, 0.50, 1000.0, 0.25)
        s2 = kelly_position_size_usd(0.65, 0.50, 2000.0, 0.25)
        assert abs(s2 / s1 - 2.0) < 0.01

    def test_max_position_cap(self):
        size = kelly_position_size_usd(
            p_win=0.99,
            market_price=0.50,
            bankroll_usd=10000.0,
            fraction=1.0,
            max_position_usd=150.0,
        )
        assert size <= 150.0

    def test_invalid_price_returns_zero(self):
        assert kelly_position_size_usd(0.6, 0.0, 1000.0) == 0.0
        assert kelly_position_size_usd(0.6, 1.0, 1000.0) == 0.0

    def test_no_edge_returns_zero(self):
        # p=0.5, price=0.5 → no edge
        size = kelly_position_size_usd(0.50, 0.50, 1000.0, 0.25)
        assert size == 0.0


class TestPortfolioKellyAdjustment:
    def test_no_existing_exposure(self):
        # No existing correlated bets → no adjustment
        adj = portfolio_kelly_adjustment(
            base_kelly=0.10,
            existing_correlated_exposure=0.0,
            bankroll_usd=1000.0,
            correlation=0.8,
            max_factor_exposure=0.40,
        )
        assert adj == 0.10

    def test_full_factor_exposure_caps_to_zero(self):
        # Already at max exposure (0.4 of bankroll) → kelly should be near 0
        adj = portfolio_kelly_adjustment(
            base_kelly=0.10,
            existing_correlated_exposure=400.0,  # 40% of 1000
            bankroll_usd=1000.0,
            correlation=1.0,
            max_factor_exposure=0.40,
        )
        assert adj == 0.0

    def test_partial_exposure_reduces_kelly(self):
        # 20% of max exposure → reduce by 50%
        adj = portfolio_kelly_adjustment(
            base_kelly=0.10,
            existing_correlated_exposure=200.0,  # 20% of 1000
            bankroll_usd=1000.0,
            correlation=1.0,
            max_factor_exposure=0.40,
        )
        # Reduction = (0.2/0.4)*1.0 = 0.5 → adj = 0.10 * (1-0.5) = 0.05
        assert abs(adj - 0.05) < 1e-6

    def test_low_correlation_minimal_reduction(self):
        # Low correlation = small reduction even at high exposure
        adj = portfolio_kelly_adjustment(
            base_kelly=0.10,
            existing_correlated_exposure=300.0,
            bankroll_usd=1000.0,
            correlation=0.1,
            max_factor_exposure=0.40,
        )
        assert adj > 0.08  # minimal reduction


class TestExpectedValue:
    def test_positive_ev(self):
        # p=0.6, b=1.0: EV = 0.6*1.0 - 0.4*1.0 = 0.2
        ev = expected_value_per_unit(p_win=0.6, b=1.0)
        assert abs(ev - 0.2) < 1e-6

    def test_zero_ev(self):
        ev = expected_value_per_unit(p_win=0.5, b=1.0)
        assert ev == 0.0

    def test_negative_ev(self):
        ev = expected_value_per_unit(p_win=0.3, b=1.0)
        assert ev < 0


class TestBreakevenAndEdge:
    def test_breakeven_equals_price(self):
        # Breakeven probability = market price (by definition for binary markets)
        price = 0.35
        assert abs(breakeven_probability(price) - price) < 1e-6

    def test_required_edge_with_fee(self):
        # price=0.50, fee=0.10: required = 0.50/(1-0.10) = 0.556
        req = required_edge_after_fees(market_price=0.50, taker_fee=0.10)
        assert abs(req - 0.50 / 0.90) < 1e-6

    def test_zero_fee_returns_price(self):
        req = required_edge_after_fees(market_price=0.45, taker_fee=0.0)
        assert abs(req - 0.45) < 1e-6


class TestKellyGrowthRate:
    def test_positive_edge_positive_growth(self):
        g = kelly_growth_rate(p_win=0.6, b=1.0, kelly_f=0.1)
        assert g > 0

    def test_zero_fraction_zero_growth(self):
        g = kelly_growth_rate(p_win=0.6, b=1.0, kelly_f=0.0)
        assert g == 0.0

    def test_full_loss_bet_zero_growth(self):
        # kelly_f >= 1 → zero growth (catastrophic)
        g = kelly_growth_rate(p_win=0.6, b=1.0, kelly_f=1.0)
        assert g == 0.0


class TestRuinProbability:
    def test_high_edge_low_ruin(self):
        p_ruin = ruin_probability_approximation(
            p_win=0.75, b=2.0, fraction=0.15, num_bets=100
        )
        assert p_ruin < 0.10

    def test_zero_edge_high_ruin(self):
        p_ruin = ruin_probability_approximation(
            p_win=0.5, b=1.0, fraction=0.25, num_bets=100
        )
        assert p_ruin > 0.3

    def test_returns_probability_bounds(self):
        p = ruin_probability_approximation(0.6, 1.5, 0.10, 50)
        assert 0.0 <= p <= 1.0


class TestDrawdownAdjustedKelly:
    def test_no_drawdown_unchanged(self):
        assert drawdown_adjusted_kelly(0.10, 0.0, 0.20) == 0.10

    def test_full_drawdown_zero(self):
        assert drawdown_adjusted_kelly(0.10, 0.20, 0.20) == 0.0

    def test_half_drawdown_halved(self):
        result = drawdown_adjusted_kelly(0.10, 0.10, 0.20)
        assert abs(result - 0.05) < 1e-9

    def test_beyond_max_drawdown_zero(self):
        assert drawdown_adjusted_kelly(0.10, 0.25, 0.20) == 0.0

    def test_zero_max_drawdown_zero(self):
        assert drawdown_adjusted_kelly(0.10, 0.05, 0.0) == 0.0

    def test_small_drawdown_minimal_reduction(self):
        result = drawdown_adjusted_kelly(0.10, 0.02, 0.20)
        assert result > 0.08  # less than 20% reduction for 10% of limit


class TestConfidenceAdjustedKelly:
    def test_full_confidence_unchanged(self):
        result = confidence_adjusted_kelly(0.10, 1.0, min_confidence=0.50)
        assert abs(result - 0.10) < 1e-9

    def test_at_min_confidence_zero(self):
        result = confidence_adjusted_kelly(0.10, 0.50, min_confidence=0.50)
        assert result == 0.0

    def test_below_min_confidence_zero(self):
        result = confidence_adjusted_kelly(0.10, 0.30, min_confidence=0.50)
        assert result == 0.0

    def test_mid_confidence_scaled(self):
        # confidence = 0.75, min = 0.50 → scale = (0.75-0.50)/(1.0-0.50) = 0.5
        result = confidence_adjusted_kelly(0.10, 0.75, min_confidence=0.50)
        assert abs(result - 0.05) < 1e-9

    def test_high_confidence_near_full(self):
        result = confidence_adjusted_kelly(0.10, 0.95, min_confidence=0.50)
        assert result > 0.08


class TestSizingReport:
    def test_report_structure(self):
        report = sizing_report(
            p_win=0.62,
            market_price=0.50,
            bankroll_usd=1000.0,
            fraction=0.25,
            taker_fee=0.02,
        )
        required_keys = [
            "p_win", "market_price", "decimal_odds", "edge", "ev_per_unit",
            "full_kelly_fraction", "scaled_kelly_fraction", "recommended_size_usd",
            "log_growth_rate", "ruin_prob_100bets",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_positive_edge_report(self):
        report = sizing_report(0.65, 0.50, 1000.0)
        assert report["edge"] > 0
        assert report["ev_per_unit"] > 0
        assert report["recommended_size_usd"] > 0
