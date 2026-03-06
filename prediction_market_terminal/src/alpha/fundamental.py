"""
Expected Value (EV) Engine — Fundamental / Directional Trading.

Compares the terminal's "True Probability" (from oracle models) against
the market's implied probability to generate directional signals.

EV formula for a binary bet:
    EV = p * (b) - (1 - p) * 1
where:
    p  = true probability of winning
    b  = net profit per unit staked (decimal_odds - 1)
    1  = unit stake lost if wrong

A trade is taken only when:
    EV > 0  AND  |edge| > min_edge  AND  all risk checks pass
"""
from __future__ import annotations

import logging
from typing import Optional

from src.core.constants import MIN_TRADEABLE_PROB, MAX_TRADEABLE_PROB
from src.core.models import (
    AlphaType,
    DirectionalSignal,
    Exchange,
    Market,
    OracleEstimate,
    RiskFlag,
    Side,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


def compute_ev(
    true_prob: float,
    market_price: float,   # price you pay per share ($0-$1)
) -> float:
    """
    Expected Value of a YES bet per unit staked.

    EV = true_prob * (1/market_price - 1) - (1 - true_prob)
       = true_prob * b - (1 - true_prob)
    where b = 1/market_price - 1  (net return if correct)

    Equivalently: EV = (true_prob - market_price) / market_price
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1.0
    return true_prob * b - (1.0 - true_prob)


def compute_edge(true_prob: float, implied_prob: float) -> float:
    """
    Raw probability edge: true_prob - implied_prob.
    Positive = market underprices the outcome.
    """
    return true_prob - implied_prob


def decimal_odds_from_price(price: float) -> float:
    """Convert a market price (0-1) to decimal odds (e.g., 0.25 → 4.0)."""
    if price <= 0:
        return float("inf")
    return 1.0 / price


class EVEngine:
    """
    Core EV engine. Given oracle estimates and market data, produces
    a ranked list of DirectionalSignal objects.
    """

    def __init__(
        self,
        min_edge_pct: float = 0.05,       # Minimum edge to generate signal (5%)
        min_oracle_confidence: float = 0.50,  # Require confidence interval width < 50%
    ) -> None:
        self._min_edge = min_edge_pct
        self._min_oracle_confidence = min_oracle_confidence

    def evaluate_market(
        self,
        market: Market,
        oracle_estimates: list[OracleEstimate],
        bankroll_usd: float = 1000.0,
    ) -> Optional[DirectionalSignal]:
        """
        Evaluate a market against oracle estimates.
        Returns the best DirectionalSignal or None if no edge found.
        """
        if not market.is_active:
            return None

        yes = market.yes_outcome
        no = market.no_outcome
        if not yes or not no:
            return None

        # Filter to valid oracle estimates
        valid_estimates = [
            e for e in oracle_estimates
            if (e.confidence_interval_high - e.confidence_interval_low) <= self._min_oracle_confidence
        ]
        if not valid_estimates:
            return None

        # Consensus true probability (inverse-CI-width weighted average)
        true_prob_yes = self._consensus_probability(valid_estimates)
        if true_prob_yes is None:
            return None

        settings = get_settings()

        # Check tradeable probability bounds
        if not (MIN_TRADEABLE_PROB <= true_prob_yes <= MAX_TRADEABLE_PROB):
            return None

        # --- Evaluate YES leg ---
        implied_yes_ask = yes.implied_prob_ask   # cost to buy YES
        edge_yes = compute_edge(true_prob_yes, implied_yes_ask)
        ev_yes_per_unit = compute_ev(true_prob_yes, implied_yes_ask)

        # --- Evaluate NO leg ---
        true_prob_no = 1.0 - true_prob_yes
        implied_no_ask = no.implied_prob_ask
        edge_no = compute_edge(true_prob_no, implied_no_ask)
        ev_no_per_unit = compute_ev(true_prob_no, implied_no_ask)

        # Choose the higher EV direction
        if ev_yes_per_unit >= ev_no_per_unit and abs(edge_yes) >= self._min_edge:
            side = Side.YES
            trade_price = implied_yes_ask
            true_prob_for_side = true_prob_yes
            implied_for_side = implied_yes_ask
            edge = edge_yes
            ev_per_unit = ev_yes_per_unit
        elif ev_no_per_unit > ev_yes_per_unit and abs(edge_no) >= self._min_edge:
            side = Side.NO
            trade_price = implied_no_ask
            true_prob_for_side = true_prob_no
            implied_for_side = implied_no_ask
            edge = edge_no
            ev_per_unit = ev_no_per_unit
        else:
            return None  # No significant edge

        if ev_per_unit <= 0:
            return None

        # Kelly sizing — apply calibration and drawdown adjustments
        from src.risk.kelly import (
            kelly_fraction,
            drawdown_adjusted_kelly,
            confidence_adjusted_kelly,
        )
        from src.alpha.calibration import get_calibration_tracker
        from config.settings import get_settings as _gs
        _settings = _gs()

        # Calibrate the probability estimate
        calibrator = get_calibration_tracker()
        calibrated_prob = calibrator.calibrate(AlphaType.EV_DIRECTIONAL, true_prob_for_side)

        b = decimal_odds_from_price(trade_price) - 1.0
        kf = kelly_fraction(
            p_win=calibrated_prob,
            b=b,
            fraction=settings.kelly_fraction,
        )
        # Scale down when in drawdown (passed via bankroll proxy — actual DD
        # is applied by the router which has portfolio access)
        # Confidence adjustment: fewer oracle sources = lower confidence
        confidence_val = self._compute_confidence(valid_estimates)
        kf = confidence_adjusted_kelly(kf, confidence_val)

        size_usd = min(
            settings.max_single_position_usd,
            bankroll_usd * kf,
        )
        size_usd = max(5.0, size_usd)  # floor at $5 to cover fees

        # Account for fees
        fee_cost = size_usd * market.taker_fee
        net_size = size_usd - fee_cost
        ev_usd = net_size * ev_per_unit

        risk_flags: list[RiskFlag] = []
        if fee_cost / max(size_usd * ev_per_unit, 0.01) > settings.fee_edge_max_consumption:
            risk_flags.append(RiskFlag.FEE_EXCESSIVE)

        days = market.days_to_expiry or 30.0
        aroc = (ev_usd / max(size_usd, 1.0)) * (365.0 / max(days, 1.0))
        if aroc < settings.aroc_minimum_annual:
            risk_flags.append(RiskFlag.AROC_BELOW_MIN)

        # Liquidity check
        if yes.volume_24h < 1000 and market.exchange == Exchange.POLYMARKET:
            risk_flags.append(RiskFlag.LOW_LIQUIDITY)

        oracle_sources = [e.source for e in valid_estimates]

        return DirectionalSignal(
            alpha_type=AlphaType.EV_DIRECTIONAL,
            market_id=market.market_id,
            exchange=market.exchange,
            side=side,
            true_probability=true_prob_for_side,
            implied_probability=implied_for_side,
            edge=edge,
            decimal_odds=decimal_odds_from_price(trade_price),
            kelly_fraction_suggested=kf,
            recommended_size_usd=size_usd,
            expected_value_usd=ev_usd,
            expiry=market.expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            confidence=self._compute_confidence(valid_estimates),
            oracle_sources=oracle_sources,
        )

    def evaluate_universe(
        self,
        markets: list[Market],
        oracle_map: dict[str, list[OracleEstimate]],
        bankroll_usd: float = 1000.0,
    ) -> list[DirectionalSignal]:
        """
        Evaluate all markets and return signals sorted by EV descending.
        oracle_map: market_id → list of OracleEstimate objects
        """
        signals: list[DirectionalSignal] = []
        for market in markets:
            estimates = oracle_map.get(market.market_id, [])
            if not estimates:
                continue
            signal = self.evaluate_market(market, estimates, bankroll_usd)
            if signal and signal.is_actionable:
                signals.append(signal)

        logger.info(
            "EV scan: evaluated %d markets, %d signals generated",
            len(markets), len(signals),
        )
        return sorted(signals, key=lambda s: s.expected_value_usd, reverse=True)

    # ---------------------------------------------------------------- Helpers

    @staticmethod
    def _consensus_probability(estimates: list[OracleEstimate]) -> Optional[float]:
        if not estimates:
            return None
        total_weight = 0.0
        weighted_sum = 0.0
        for est in estimates:
            ci_width = max(0.01, est.confidence_interval_high - est.confidence_interval_low)
            weight = 1.0 / ci_width
            weighted_sum += est.true_probability * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else None

    @staticmethod
    def _compute_confidence(estimates: list[OracleEstimate]) -> float:
        """Aggregate source confidence: 1 source = 0.6, 2+ = 0.8, 3+ = 0.9."""
        n = len(estimates)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.60
        if n == 2:
            return 0.75
        return min(0.95, 0.80 + 0.05 * (n - 2))
