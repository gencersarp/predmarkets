"""
Kelly Criterion — adapted for binary prediction markets.

Standard Kelly formula for a binary bet:
    f* = (b*p - q) / b
where:
    f* = fraction of bankroll to bet
    b  = net profit per unit staked if correct (e.g. if you pay $0.30 and win $1, b = 0.70/0.30 ≈ 2.33)
    p  = probability of winning (true probability)
    q  = 1 - p = probability of losing
    b  = (1 - price) / price  for a binary market at price P

For a binary prediction market where you buy YES at price P (per $1 payout):
    Net profit if win: (1 - P) per unit of stake (you get $1, paid P)
    So b = (1 / P) - 1

Full Kelly is aggressive; we scale by `fraction` (default Quarter-Kelly = 0.25).

Multi-outcome (portfolio) Kelly:
    For correlated positions, the portfolio Kelly requires solving a QP.
    We use a simplified approximation: reduce individual Kelly by the
    fraction of available capital already allocated to correlated bets.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from src.core.constants import KELLY_MAX_FRACTION

logger = logging.getLogger(__name__)


def kelly_fraction(
    p_win: float,
    b: float,                    # net profit per unit if win
    fraction: float = 0.25,      # Kelly scaling factor (default quarter-Kelly)
) -> float:
    """
    Compute the fractional Kelly stake as a fraction of the bankroll.

    Returns a value in [0, KELLY_MAX_FRACTION].

    Edge cases:
    - If b <= 0: no bet (odds are worse than even)
    - If p_win = 0: no bet
    - If kelly < 0: no bet (negative edge)
    - Cap at KELLY_MAX_FRACTION regardless of formula

    Args:
        p_win:    True probability of the bet winning
        b:        Net profit per unit staked if correct
        fraction: Fraction of full Kelly to use (default 0.25 = Quarter-Kelly)
    """
    if p_win <= 0 or b <= 0:
        return 0.0
    if p_win >= 1.0:
        return KELLY_MAX_FRACTION

    q_lose = 1.0 - p_win
    # Full Kelly
    full_kelly = (b * p_win - q_lose) / b

    if full_kelly <= 0:
        return 0.0  # Negative edge

    scaled = full_kelly * fraction
    return min(scaled, KELLY_MAX_FRACTION)


def kelly_position_size_usd(
    p_win: float,
    market_price: float,     # price per $1 payout (0-1)
    bankroll_usd: float,
    fraction: float = 0.25,
    max_position_usd: Optional[float] = None,
) -> float:
    """
    Compute the dollar amount to bet on a binary market.

    Args:
        p_win:           True probability of winning
        market_price:    Price per $1 payout (i.e. the ask price)
        bankroll_usd:    Current available bankroll
        fraction:        Kelly scaling
        max_position_usd: Hard cap per position

    Returns:
        Dollar amount to bet (USD)
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    b = (1.0 / market_price) - 1.0  # net profit per unit if win
    kf = kelly_fraction(p_win=p_win, b=b, fraction=fraction)
    size = bankroll_usd * kf

    if max_position_usd is not None:
        size = min(size, max_position_usd)

    return max(0.0, size)


def portfolio_kelly_adjustment(
    base_kelly: float,
    existing_correlated_exposure: float,   # USD already in correlated bets
    bankroll_usd: float,
    correlation: float,                    # correlation coefficient [0,1]
    max_factor_exposure: float = 0.40,     # max fraction of bankroll per factor
) -> float:
    """
    Reduce the Kelly fraction when we already have correlated exposure.

    Intuition: if 40% of our bankroll is in "GOP wins" bets and we want to
    add "Trump wins PA" (highly correlated), reduce the new bet size
    proportionally.

    Simple linear scaling:
        adjustment = max(0, 1 - (correlated_exposure_pct / max_factor_exposure) * correlation)
    """
    correlated_pct = existing_correlated_exposure / max(bankroll_usd, 1.0)
    reduction = (correlated_pct / max(max_factor_exposure, 0.01)) * correlation
    adjusted = base_kelly * max(0.0, 1.0 - reduction)
    return min(adjusted, KELLY_MAX_FRACTION)


def expected_value_per_unit(p_win: float, b: float) -> float:
    """
    EV per unit staked.
    EV = p * b - (1 - p) * 1
    """
    return p_win * b - (1.0 - p_win)


def breakeven_probability(market_price: float) -> float:
    """
    The true probability needed for a bet at `market_price` to break even.
    Breakeven: p * (1/price) - 1 = 0  →  p = price
    So breakeven probability equals the market price itself (ignoring fees).
    """
    return market_price


def required_edge_after_fees(
    market_price: float,
    taker_fee: float,
) -> float:
    """
    Minimum true probability required for a net-positive EV trade after fees.

    Effective payout = (1 - taker_fee) per winning unit
    Breakeven: p * (1 - taker_fee) / price >= 1
    → p >= price / (1 - taker_fee)
    """
    if taker_fee >= 1.0:
        return 1.0
    return market_price / (1.0 - taker_fee)


def kelly_growth_rate(p_win: float, b: float, kelly_f: float) -> float:
    """
    Log-growth rate of the Kelly strategy (geometric mean approximation).
    G = p * ln(1 + b*f) + (1-p) * ln(1 - f)

    Useful for comparing strategies by long-run capital growth.
    """
    if kelly_f <= 0 or kelly_f >= 1:
        return 0.0
    win_component = p_win * math.log(1.0 + b * kelly_f)
    loss_component = (1.0 - p_win) * math.log(1.0 - kelly_f)
    return win_component + loss_component


def ruin_probability_approximation(
    p_win: float,
    b: float,
    fraction: float,
    num_bets: int = 100,
) -> float:
    """
    Approximate probability of losing 50%+ of bankroll over `num_bets` bets
    using normal approximation of the log-wealth process.

    log(W_n / W_0) ~ N(n*μ_g, n*σ²_g)
    where μ_g = growth rate, σ²_g = variance of log returns
    """
    q = 1.0 - p_win
    mu_g = kelly_growth_rate(p_win, b, fraction)
    # Variance of log-wealth increment
    win_log = math.log(1.0 + b * fraction)
    loss_log = math.log(1.0 - fraction)
    sigma_g_sq = p_win * (win_log - mu_g) ** 2 + q * (loss_log - mu_g) ** 2

    if sigma_g_sq <= 0 or num_bets <= 0:
        return 0.0

    # P(log(W_n/W_0) < -0.693) = P(50% loss)
    target = -math.log(2)  # log(0.5) = 50% loss threshold
    z = (target - num_bets * mu_g) / math.sqrt(num_bets * sigma_g_sq)
    # Normal CDF approximation
    return _normal_cdf(z)


def drawdown_adjusted_kelly(
    base_kelly: float,
    current_drawdown: float,
    max_drawdown: float,
) -> float:
    """
    Scale Kelly fraction linearly as drawdown deepens.

    Rationale: when you're losing, your edge estimates may be wrong
    (adverse selection, regime change). Protect remaining capital by
    betting smaller. At max_drawdown, Kelly → 0.

    Scale: 1.0 at 0% DD, 0.0 at max_drawdown.
    Full position halved when drawdown reaches 50% of the limit.

        adjusted = base_kelly * max(0, 1 - drawdown / max_drawdown)

    Args:
        base_kelly:        The Kelly fraction before adjustment
        current_drawdown:  Current portfolio drawdown (0.0–1.0)
        max_drawdown:      Hard limit drawdown (e.g. 0.20)
    """
    if max_drawdown <= 0:
        return 0.0
    scale = max(0.0, 1.0 - current_drawdown / max_drawdown)
    return base_kelly * scale


def confidence_adjusted_kelly(
    base_kelly: float,
    confidence: float,
    min_confidence: float = 0.50,
) -> float:
    """
    Scale Kelly by signal confidence.

    confidence = 1.0 → no reduction (full Kelly)
    confidence = min_confidence → Kelly → 0

    Linear interpolation:
        adjusted = base_kelly * (confidence - min_confidence) / (1.0 - min_confidence)
    """
    if confidence <= min_confidence:
        return 0.0
    scale = (confidence - min_confidence) / max(1.0 - min_confidence, 1e-6)
    return base_kelly * min(scale, 1.0)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Sizing Diagnostics (for logging / UI display)
# ---------------------------------------------------------------------------

def sizing_report(
    p_win: float,
    market_price: float,
    bankroll_usd: float,
    fraction: float = 0.25,
    taker_fee: float = 0.02,
) -> dict:
    """Return a comprehensive sizing diagnostic dictionary."""
    b = (1.0 / max(market_price, 1e-6)) - 1.0
    kf = kelly_fraction(p_win, b, fraction)
    size = kelly_position_size_usd(p_win, market_price, bankroll_usd, fraction)
    ev = expected_value_per_unit(p_win, b)
    growth = kelly_growth_rate(p_win, b, kf)
    req_edge = required_edge_after_fees(market_price, taker_fee)

    return {
        "p_win": p_win,
        "market_price": market_price,
        "decimal_odds": 1.0 / max(market_price, 1e-6),
        "net_profit_b": b,
        "implied_prob": market_price,
        "edge": p_win - market_price,
        "breakeven_prob": breakeven_probability(market_price),
        "required_prob_after_fees": req_edge,
        "ev_per_unit": ev,
        "full_kelly_fraction": kelly_fraction(p_win, b, 1.0),
        "scaled_kelly_fraction": kf,
        "recommended_size_usd": size,
        "log_growth_rate": growth,
        "ruin_prob_100bets": ruin_probability_approximation(p_win, b, kf),
    }
