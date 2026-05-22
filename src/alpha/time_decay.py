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
        # Cap at 0.85 to reflect model uncertainty — even high-lambda events
        # have a non-trivial chance of NOT occurring before resolution.
        return max(0.01, min(0.85, prob))

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

# Event type → estimated daily rate (lambda). None = skip (not Poisson).
LAMBDA_ESTIMATES: dict[str, Optional[float]] = {
    "elon_tweet_crypto": 3.0,          # ~3 times/day
    "elon_tweet_doge": 0.5,            # ~every 2 days
    "trump_truthsocial_post": 10.0,    # multiple times/day
    "fed_meeting": 1 / 45.0,           # every ~45 days (scheduled)
    "cpi_release": 1 / 30.0,           # monthly
    "breaking_news_major": 1 / 7.0,    # rough estimate
    "generic_weekly": 0.25,             # moderate default
    "weather": None,                    # not Poisson — skip
    "test_market": None,                # garbage data — skip
    "sports": None,                     # Not Poisson — skip
    "sports_prop": None,                # Not Poisson — skip
    "legal": None,                      # Court cases aren't Poisson — skip
    "crypto_social": 1.5,              # frequent events
    "economic": 1 / 30.0,              # monthly releases
    "political": 1 / 21.0,             # slow news cycle
}


def _classify_event_type(title: str) -> str:
    """Classify a market's event type from its title for lambda estimation."""
    t = title.lower()
    if any(w in t for w in ["temp", "temperature", "high temp", "weather", "rain", "snow", "wind"]):
        return "weather"
    if any(w in t for w in ["1+1", "2+2", "test", "quicksettle"]):
        return "test_market"
    if any(w in t for w in ["convicted", "court", "trial", "sentenced", "lawsuit", "sued", "legal", "arrested"]):
        return "legal"
    # Sports sub-categories
    if any(w in t for w in ["spread", "total", "over", "under", "points", "assists", "goals",
                             "rebounds", "strikeouts", "hits", "yards", "touchdown", "td ", "home run"]):
        return "sports_prop"  # player/game props resolve within a game
    if any(w in t for w in ["nba", "nfl", "mlb", "nhl", "game", "match", "score", "vs ",
                             "winner", "wins", "win", "lose", "lost", "playoff", "finals"]):
        return "sports"
    if any(w in t for w in ["tweet", "post", "elon", "bitcoin", "btc", "eth", "crypto"]):
        return "crypto_social"
    if any(w in t for w in ["cpi", "gdp", "jobs", "unemployment", "fed ", "rate cut", "rate hike"]):
        return "economic"
    if any(w in t for w in ["election", "vote", "president", "congress", "bill ", "law ", "governor"]):
        return "political"
    return "generic_weekly"


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class TimeDecaySignalGenerator:
    """
    Compares market-implied probability against Poisson model probability.
    If market is overpriced (too slow decay), generates a SHORT YES signal.
    If market is underpriced (too fast decay), generates a LONG YES signal.
    """

    def __init__(self, min_edge: float = 0.05) -> None:
        self._min_edge = min_edge  # Slightly aggressive (5%) to ensure activity in paper mode

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

        raw_model_prob = model.true_probability_at()

        ask = yes_outcome.implied_prob_ask
        bid = yes_outcome.implied_prob_bid
        # Skip degenerate markets
        if ask >= 1.0 or ask <= 0.0 or bid >= 1.0 or bid < 0.0:
            return None

        # Determine direction first based on model divergence from mid
        mid_prob = (bid + ask) / 2
        
        # Anchor model probability toward mid price to avoid nonsensical edges.
        # Blend: 40% model, 60% market. Trust the Poisson model enough to
        # preserve meaningful edge while still anchoring to reality.
        model_prob = 0.40 * raw_model_prob + 0.60 * mid_prob

        if model_prob > mid_prob:
            # We want to buy YES. Actual edge is model_prob - ask (cost to buy)
            edge = model_prob - ask
            trade_side = Side.YES
            trade_price = ask
            decimal_odds = 1.0 / ask
            true_prob_for_side = model_prob
            implied_for_side = ask
        else:
            # We want to buy NO. Actual edge is (1 - model_prob) - (1 - bid) = bid - model_prob
            edge = bid - model_prob
            trade_side = Side.NO
            trade_price = 1.0 - bid # cost to buy NO
            decimal_odds = 1.0 / (1.0 - bid)
            true_prob_for_side = 1.0 - model_prob
            implied_for_side = 1.0 - bid

        # Cap edge at 15% — larger edges are usually model errors.
        if abs(edge) > 0.15:
            return None

        # Near-expiry urgency: lower the minimum edge threshold when < 48h left.
        days_left = market.days_to_expiry or 30.0
        from src.core.constants import NEAR_EXPIRY_HOURS
        near_expiry = days_left <= (NEAR_EXPIRY_HOURS / 24.0)
        very_near_expiry = days_left <= 0.25  # < 6 hours
        if very_near_expiry:
            effective_min_edge = self._min_edge * 0.40  # AROC is huge, tiny edge is fine
        elif near_expiry:
            effective_min_edge = self._min_edge * 0.60
        else:
            effective_min_edge = self._min_edge

        if edge < effective_min_edge:
            return None

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
    event_type: Optional[str] = None,
    custom_lambda: Optional[float] = None,
) -> Optional[PoissonDecayModel]:
    """
    Fit a Poisson model to a market.

    Returns None for markets where Poisson model is inappropriate
    (weather, test markets, etc.)
    """
    if market.expiry is None:
        return None

    if custom_lambda is not None:
        lam = custom_lambda
    elif event_type and event_type in LAMBDA_ESTIMATES:
        lam_val = LAMBDA_ESTIMATES[event_type]
        if lam_val is None:
            return None  # Not a Poisson-compatible market
        lam = lam_val
    else:
        # Auto-classify from title
        classified_type = _classify_event_type(market.title)
        lam_val = LAMBDA_ESTIMATES.get(classified_type)
        if lam_val is None:
            return None  # Not a Poisson-compatible market
        lam = lam_val

    return PoissonDecayModel(
        lambda_rate=lam,
        reference_time=datetime.now(timezone.utc),
        resolution_time=market.expiry,
    )


# ---------------------------------------------------------------------------
# Scheduled-Event Probability Model (non-Poisson)
# ---------------------------------------------------------------------------

def geometric_decay_probability(
    current_prob: float,
    days_elapsed: float,
    half_life_days: float,
) -> float:
    """
    Geometric (exponential) decay model for probabilities that follow a
    random-walk-like drift toward zero over time.

    Useful for markets where the underlying process is not event-driven
    (no fixed lambda) but the probability erodes steadily — e.g.,
    "Will BTC hit $X before year-end?" where each passing day without
    the event reduces the implied probability multiplicatively.

    Formula: P(t) = P_0 * 0.5^(days_elapsed / half_life_days)

    Args:
        current_prob:   Observed market probability at t=0
        days_elapsed:   Days since the market was observed
        half_life_days: Days in which the probability halves if no new info

    Returns:
        Decayed probability in [0.0, 1.0]
    """
    if half_life_days <= 0:
        return 0.0
    decay_factor = 0.5 ** (days_elapsed / half_life_days)
    return max(0.0, min(1.0, current_prob * decay_factor))


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
