"""
Time-Decay (Theta) Harvesting — Poisson Process Modelling.

Theory:
  For events governed by a Poisson process (e.g., "Will X tweet about Y this week?"),
  the probability the event has NOT yet occurred decays as:
      P(no event by time t) = exp(-λt)
  Therefore:
      P(event occurs by time T | currently at time t) = 1 - exp(-λ(T - t))

If the market's implied probability decays SLOWER than this formula, the market
is overpriced for YES. The strategy: short YES (buy NO).

Conversely, if the event has a known trigger (e.g., a scheduled Fed meeting),
model the probability SPIKE at the event time.

Additional models:
  - Geometric decay: for probabilities that follow a random walk
  - Step-function: for discrete decision events (e.g., "Fed cuts at Jan meeting?")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.core.models import (
    AlphaType,
    DirectionalSignal,
    Exchange,
    Market,
    RiskFlag,
    Side,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Poisson Time-Decay Model
# ---------------------------------------------------------------------------

@dataclass
class PoissonDecayModel:
    """
    Parameters for a Poisson decay model on a binary market.

    lambda_rate: Expected events per day (e.g., 0.5 = once every 2 days on avg)
    reference_time: The time at which the market was created / rate was estimated
    resolution_time: The market expiry time
    """
    lambda_rate: float         # events per day
    reference_time: datetime
    resolution_time: datetime

    def true_probability_at(self, query_time: Optional[datetime] = None) -> float:
        """
        P(event occurs before resolution | not yet occurred at query_time)

        Uses: P = 1 - exp(-λ * Δt_remaining)
        where Δt_remaining = resolution_time - query_time (in days)
        """
        t = query_time or datetime.now(timezone.utc)
        if t >= self.resolution_time:
            return 0.0  # Expired without event

        remaining_days = (self.resolution_time - t).total_seconds() / 86400
        prob = 1.0 - math.exp(-self.lambda_rate * remaining_days)
        return max(0.01, min(0.99, prob))

    def decay_rate_at(self, query_time: Optional[datetime] = None) -> float:
        """
        Instantaneous rate of change of probability per day (dP/dt).
        This is the "theta" of the position.

        dP/dt = -λ * exp(-λ * Δt)
        """
        t = query_time or datetime.now(timezone.utc)
        remaining_days = max(0.0, (self.resolution_time - t).total_seconds() / 86400)
        return -self.lambda_rate * math.exp(-self.lambda_rate * remaining_days)

    def half_life_days(self) -> float:
        """Number of days until probability decays to 50% (from current value)."""
        if self.lambda_rate <= 0:
            return float("inf")
        return math.log(2) / self.lambda_rate

    def expected_decay_over(self, days: float) -> float:
        """How much probability decays over the next N days."""
        t = datetime.now(timezone.utc)
        p_now = self.true_probability_at(t)
        # Simulate forward by `days`
        from datetime import timedelta
        t_future = datetime.now(timezone.utc) + timedelta(days=days)
        if t_future >= self.resolution_time:
            p_future = 0.0
        else:
            remaining_future = (self.resolution_time - t_future).total_seconds() / 86400
            p_future = 1.0 - math.exp(-self.lambda_rate * remaining_future)
        return p_now - p_future


# ---------------------------------------------------------------------------
# Common Poisson Rate Estimates
# ---------------------------------------------------------------------------

# Event type → estimated daily rate (lambda)
LAMBDA_ESTIMATES: dict[str, float] = {
    "elon_tweet_crypto": 3.0,          # ~3 times/day
    "elon_tweet_doge": 0.5,            # ~every 2 days
    "trump_truthsocial_post": 10.0,    # multiple times/day
    "fed_meeting": 1 / 45.0,           # every ~45 days (scheduled)
    "nba_game": 2.5 / 7.0,             # ~2.5 games/week per team
    "cpi_release": 1 / 30.0,           # monthly
    "breaking_news_major": 1 / 7.0,   # rough estimate
    "generic_weekly": 1.0,             # 1 expected event per day over a week
}


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class TimeDecaySignalGenerator:
    """
    Compares market-implied probability against Poisson model probability.
    If market is overpriced (too slow decay), generates a SHORT YES signal.
    If market is underpriced (too fast decay), generates a LONG YES signal.
    """

    def __init__(self, min_edge: float = 0.04) -> None:
        self._min_edge = min_edge  # Minimum edge to generate a signal (4%)

    def generate_signal(
        self,
        market: Market,
        model: PoissonDecayModel,
        oracle_lambda: Optional[float] = None,
    ) -> Optional[DirectionalSignal]:
        """
        Generate a theta-harvesting signal for a market.

        Args:
            market: The market to analyse
            model: Pre-fitted Poisson model
            oracle_lambda: Override the model lambda if external data is available
        """
        if oracle_lambda is not None:
            model = PoissonDecayModel(
                lambda_rate=oracle_lambda,
                reference_time=model.reference_time,
                resolution_time=model.resolution_time,
            )

        yes_outcome = market.yes_outcome
        if not yes_outcome:
            return None

        if market.expiry is None:
            return None

        model_prob = model.true_probability_at()

        ask = yes_outcome.implied_prob_ask
        bid = yes_outcome.implied_prob_bid
        # Skip degenerate markets with no real orderbook (ask=1 or ask=0 causes division by zero)
        if ask >= 1.0 or ask <= 0.0 or bid >= 1.0 or bid < 0.0:
            return None

        # Use mid price as market implied
        implied_prob = (bid + ask) / 2

        edge = model_prob - implied_prob  # positive = model says higher than market

        # Near-expiry urgency: lower the minimum edge threshold when < 48h left.
        # The theta is highest near expiry and the opportunity closes fast.
        days_left = market.days_to_expiry or 30.0
        from src.core.constants import NEAR_EXPIRY_HOURS
        near_expiry = days_left <= (NEAR_EXPIRY_HOURS / 24.0)
        effective_min_edge = self._min_edge * 0.60 if near_expiry else self._min_edge

        if abs(edge) < effective_min_edge:
            return None

        # Determine trade direction
        if edge < 0:
            # Market overpriced relative to model: SHORT YES (BUY NO)
            trade_side = Side.NO
            trade_price = ask   # cost to buy NO ≈ 1 - ask_yes
            decimal_odds = 1.0 / (1.0 - ask)
            true_prob_for_side = 1.0 - model_prob
            implied_for_side = 1.0 - implied_prob
        else:
            # Market underpriced: LONG YES
            trade_side = Side.YES
            trade_price = ask
            decimal_odds = 1.0 / ask
            true_prob_for_side = model_prob
            implied_for_side = implied_prob

        from src.risk.kelly import kelly_fraction
        from config.settings import get_settings

        settings = get_settings()
        b = decimal_odds - 1.0  # net profit per unit staked if correct
        kf = kelly_fraction(
            p_win=true_prob_for_side,
            b=b,
            fraction=settings.kelly_fraction,
        )
        max_usd = min(
            settings.max_single_position_usd,
            settings.max_portfolio_exposure_usd * kf,
        )

        ev_usd = max_usd * (true_prob_for_side * b - (1 - true_prob_for_side))

        days = market.days_to_expiry or 30.0
        aroc = (ev_usd / max(max_usd, 1.0)) * (365.0 / max(days, 1.0))

        risk_flags: list[RiskFlag] = []
        if max_usd < 5.0:
            risk_flags.append(RiskFlag.LOW_LIQUIDITY)
        if aroc < settings.aroc_minimum_annual:
            risk_flags.append(RiskFlag.AROC_BELOW_MIN)

        # Near-expiry: higher confidence, AROC dominates anyway
        base_confidence = 0.85 if near_expiry else 0.70

        return DirectionalSignal(
            alpha_type=AlphaType.TIME_DECAY,
            market_id=market.market_id,
            exchange=market.exchange,
            side=trade_side,
            true_probability=true_prob_for_side,
            implied_probability=implied_for_side,
            edge=abs(edge),
            decimal_odds=decimal_odds,
            kelly_fraction_suggested=kf,
            recommended_size_usd=max_usd,
            expected_value_usd=ev_usd,
            expiry=market.expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            confidence=base_confidence,
            oracle_sources=["poisson_decay_model" + ("_near_expiry" if near_expiry else "")],
        )


# ---------------------------------------------------------------------------
# Convenience: fit a model from market metadata
# ---------------------------------------------------------------------------

def fit_poisson_model(
    market: Market,
    event_type: str = "generic_weekly",
    custom_lambda: Optional[float] = None,
) -> Optional[PoissonDecayModel]:
    """
    Fit a Poisson model to a market.

    The lambda is estimated from:
    1. custom_lambda if provided (e.g. from historical analysis)
    2. LAMBDA_ESTIMATES dict if event_type is known
    3. A naive estimate from market implied probability and remaining time
    """
    if market.expiry is None:
        return None

    if custom_lambda is not None:
        lam = custom_lambda
    elif event_type in LAMBDA_ESTIMATES:
        lam = LAMBDA_ESTIMATES[event_type]
    else:
        # Back-solve lambda from implied probability and remaining time
        yes = market.yes_outcome
        if not yes:
            return None
        implied = (yes.implied_prob_bid + yes.implied_prob_ask) / 2
        days_remaining = max(0.1, (market.expiry - datetime.now(timezone.utc)).total_seconds() / 86400)
        # P = 1 - exp(-λT)  →  λ = -ln(1 - P) / T
        if implied <= 0 or implied >= 1:
            return None
        lam = -math.log(1 - implied) / days_remaining
        logger.debug(
            "Back-solved λ=%.4f for market %s (implied=%.2f, days=%.1f)",
            lam, market.market_id, implied, days_remaining,
        )

    return PoissonDecayModel(
        lambda_rate=lam,
        reference_time=datetime.now(timezone.utc),
        resolution_time=market.expiry,
    )


# ---------------------------------------------------------------------------
# Scheduled-Event Probability Model (non-Poisson)
# ---------------------------------------------------------------------------

def step_function_probability(
    base_prob: float,
    event_date: datetime,
    days_before_event_range: float = 7.0,
) -> float:
    """
    For markets where probability spikes at a known scheduled event
    (e.g., "Fed cuts at Nov meeting?"), model the probability as roughly
    constant until the event, then resolves binary.

    Returns the theoretical probability for a given query time.
    """
    now = datetime.now(timezone.utc)
    days_to_event = (event_date - now).total_seconds() / 86400

    if days_to_event <= 0:
        # Event has passed — probability is either 0 or 1 (binary resolution)
        return base_prob  # return base; caller should check resolution

    # Scale: probability converges to base_prob as event approaches
    # Within 1 day: use base probability directly
    if days_to_event <= 1:
        return base_prob

    # Further out: slight discounting for time value
    # (Information arrives non-linearly; incorporate if needed)
    return base_prob
