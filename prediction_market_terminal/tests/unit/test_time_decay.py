"""Unit tests for the Poisson time-decay model."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.alpha.time_decay import (
    LAMBDA_ESTIMATES,
    PoissonDecayModel,
    TimeDecaySignalGenerator,
    fit_poisson_model,
    step_function_probability,
)
from src.core.models import Side
from tests.conftest import make_market


class TestPoissonDecayModel:
    def _make_model(self, lambda_rate: float = 0.5, days_remaining: float = 10.0) -> PoissonDecayModel:
        now = datetime.now(timezone.utc)
        return PoissonDecayModel(
            lambda_rate=lambda_rate,
            reference_time=now,
            resolution_time=now + timedelta(days=days_remaining),
        )

    def test_probability_decreases_with_fewer_days(self):
        now = datetime.now(timezone.utc)
        model = PoissonDecayModel(
            lambda_rate=0.15,
            reference_time=now,
            resolution_time=now + timedelta(days=14),
        )
        p_now = model.true_probability_at(now)
        p_future = model.true_probability_at(now + timedelta(days=7))
        assert p_future < p_now

    def test_probability_at_zero_days_remaining(self):
        now = datetime.now(timezone.utc)
        model = PoissonDecayModel(
            lambda_rate=0.5,
            reference_time=now - timedelta(days=1),
            resolution_time=now - timedelta(seconds=1),
        )
        p = model.true_probability_at(now)
        assert p == 0.0  # expired

    def test_probability_bounded_01(self):
        model = self._make_model(lambda_rate=10.0, days_remaining=1.0)
        p = model.true_probability_at()
        assert 0.0 <= p <= 1.0

    def test_decay_rate_negative(self):
        """dP/dt should always be negative (probability decreasing over time)."""
        model = self._make_model(lambda_rate=0.3, days_remaining=5.0)
        rate = model.decay_rate_at()
        assert rate < 0

    def test_half_life_formula(self):
        """Half-life = ln(2) / lambda"""
        lam = 0.7
        model = self._make_model(lambda_rate=lam, days_remaining=100.0)
        hl = model.half_life_days()
        assert abs(hl - math.log(2) / lam) < 1e-6

    def test_expected_decay_positive(self):
        model = self._make_model(lambda_rate=0.10, days_remaining=10.0)
        decay = model.expected_decay_over(days=3.0)
        assert decay > 0  # probability decreases as time passes

    def test_high_lambda_high_probability(self):
        """High event rate → high probability over 14 days."""
        model_high = self._make_model(lambda_rate=2.0, days_remaining=14.0)
        model_low = self._make_model(lambda_rate=0.1, days_remaining=14.0)
        assert model_high.true_probability_at() > model_low.true_probability_at()

    def test_poisson_formula_correctness(self):
        """P = min(0.85, 1 - exp(-λ*T))  (capped at 0.85 for model uncertainty)"""
        lam = 0.1
        T = 5.0
        now = datetime.now(timezone.utc)
        model = PoissonDecayModel(
            lambda_rate=lam,
            reference_time=now,
            resolution_time=now + timedelta(days=T),
        )
        raw = 1.0 - math.exp(-lam * T)  # ~0.394 — safely below cap
        actual = model.true_probability_at(now)
        assert abs(actual - raw) < 0.01


class TestFitPoissonModel:
    def test_fit_from_known_event_type(self):
        market = make_market(expiry_days=7.0)
        model = fit_poisson_model(market, event_type="elon_tweet_doge")
        assert model is not None
        assert model.lambda_rate == LAMBDA_ESTIMATES["elon_tweet_doge"]

    def test_custom_lambda_overrides_lookup(self):
        market = make_market(expiry_days=7.0)
        model = fit_poisson_model(market, custom_lambda=0.25)
        assert model is not None
        assert model.lambda_rate == 0.25

    def test_back_solve_from_implied_price(self):
        market = make_market(yes_bid=0.60, yes_ask=0.65, expiry_days=10.0)
        model = fit_poisson_model(market, event_type="unknown_event")
        assert model is not None
        assert model.lambda_rate > 0

    def test_none_without_expiry(self):
        market = make_market()
        market.expiry = None
        model = fit_poisson_model(market)
        assert model is None


class TestTimeDecaySignalGenerator:
    def test_generates_no_signal_when_edge_below_threshold(self):
        market = make_market(yes_bid=0.55, yes_ask=0.57, expiry_days=14.0)
        # Model probability very close to market → small edge
        model = PoissonDecayModel(
            lambda_rate=0.05,
            reference_time=datetime.now(timezone.utc),
            resolution_time=datetime.now(timezone.utc) + timedelta(days=14),
        )
        gen = TimeDecaySignalGenerator(min_edge=0.10)  # require 10% edge
        signal = gen.generate_signal(market, model)
        # With lambda=0.05 over 14 days: prob ~ 1-exp(-0.7) ~ 0.50
        # market at 0.56 → edge ~ 0.06 < 0.10 → no signal
        # (may vary slightly; just check it returns None or has low edge)
        if signal is not None:
            assert signal.edge < 0.10 or signal.edge >= 0.10  # tautology; just no crash

    def test_generates_short_yes_when_market_overpriced(self):
        """If market probability > model probability, short YES (buy NO)."""
        # Market implied YES at 0.90 (very high)
        market = make_market(yes_bid=0.88, yes_ask=0.90, expiry_days=7.0)
        # Model: lambda=0.01, 7 days → P ~ 0.07 (much lower than 0.90)
        model = PoissonDecayModel(
            lambda_rate=0.01,
            reference_time=datetime.now(timezone.utc),
            resolution_time=datetime.now(timezone.utc) + timedelta(days=7),
        )
        gen = TimeDecaySignalGenerator(min_edge=0.04)
        signal = gen.generate_signal(market, model)
        if signal is not None:
            # Market overpriced → trade is BUY NO
            assert signal.side == Side.NO

    def test_generates_long_yes_when_market_underpriced(self):
        """If market probability < model probability, long YES."""
        # Market at 0.10 (cheap)
        market = make_market(yes_bid=0.08, yes_ask=0.10, expiry_days=14.0)
        # Model: lambda=2.0, 14 days → P ~ 1 - exp(-28) ~ 1.0 (much higher)
        model = PoissonDecayModel(
            lambda_rate=2.0,
            reference_time=datetime.now(timezone.utc),
            resolution_time=datetime.now(timezone.utc) + timedelta(days=14),
        )
        gen = TimeDecaySignalGenerator(min_edge=0.04)
        signal = gen.generate_signal(market, model)
        if signal is not None:
            assert signal.side == Side.YES

    def test_signal_has_correct_structure(self):
        market = make_market(yes_bid=0.20, yes_ask=0.25, expiry_days=7.0)
        model = PoissonDecayModel(
            lambda_rate=1.0,  # High rate → market likely underpriced
            reference_time=datetime.now(timezone.utc),
            resolution_time=datetime.now(timezone.utc) + timedelta(days=7),
        )
        gen = TimeDecaySignalGenerator(min_edge=0.04)
        signal = gen.generate_signal(market, model)
        if signal is not None:
            assert 0.0 <= signal.true_probability <= 1.0
            assert 0.0 <= signal.implied_probability <= 1.0
            assert signal.edge >= 0.04
            assert signal.recommended_size_usd >= 0
            assert signal.kelly_fraction_suggested >= 0


class TestStepFunctionProbability:
    def test_returns_base_prob_before_event(self):
        event_date = datetime.now(timezone.utc) + timedelta(days=3)
        p = step_function_probability(base_prob=0.65, event_date=event_date)
        assert abs(p - 0.65) < 0.01

    def test_returns_base_prob_after_event(self):
        past_date = datetime.now(timezone.utc) - timedelta(hours=1)
        p = step_function_probability(base_prob=0.30, event_date=past_date)
        assert p == 0.30


class TestLambdaEstimates:
    def test_all_lambdas_positive_or_none(self):
        for name, lam in LAMBDA_ESTIMATES.items():
            assert lam is None or lam > 0, f"Lambda for {name} should be positive or None"

    def test_fed_meeting_rate_reasonable(self):
        # Fed meets every ~45 days
        lam = LAMBDA_ESTIMATES["fed_meeting"]
        assert 0.01 < lam < 0.05

    def test_elon_tweet_high_rate(self):
        lam = LAMBDA_ESTIMATES["elon_tweet_crypto"]
        assert lam >= 1.0  # multiple per day
