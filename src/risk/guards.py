"""
Risk Guards — Hard Stops & Circuit Breakers.

These are the LAST LINE OF DEFENCE before any execution.
Every trade must pass through ALL guards before an order is sent.

Guards are designed to be:
  - Fail-safe: if a guard cannot be evaluated, it blocks by default
  - Composable: all guards run in sequence; any failure blocks execution
  - Auditable: every block is logged with sufficient detail for post-mortem
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.core.constants import (
    MAX_MARKET_SNAPSHOT_AGE_SEC,
    MIN_MARKET_VOLUME_24H_USD,
    MIN_TRADEABLE_PROB,
    MAX_TRADEABLE_PROB,
)
from src.core.exceptions import (
    CorrelationLimitBreached,
    DrawdownLimitBreached,
    GasLimitExceeded,
    PaperModeViolation,
    PositionSizeTooLarge,
    RiskLimitBreached,
    SlippageLimitExceeded,
    StaleDataError,
)
from src.core.models import (
    ArbitrageOpportunity,
    DirectionalSignal,
    Exchange,
    Market,
    RiskFlag,
)
from src.risk.portfolio import PortfolioManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    passed: bool
    guard_name: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Individual Guards
# ---------------------------------------------------------------------------

def guard_paper_mode(is_live_order: bool) -> GuardResult:
    """Block any attempt to sign/broadcast in paper mode."""
    settings = get_settings()
    if settings.is_paper and is_live_order:
        raise PaperModeViolation()
    return GuardResult(passed=True, guard_name="paper_mode")


def guard_drawdown(portfolio: PortfolioManager) -> GuardResult:
    """Block all new positions if max drawdown is breached."""
    settings = get_settings()
    dd = portfolio.drawdown
    if dd >= settings.max_drawdown_pct:
        raise DrawdownLimitBreached(
            limit_name="max_drawdown",
            current=dd,
            limit=settings.max_drawdown_pct,
        )
    return GuardResult(passed=True, guard_name="drawdown", reason=f"DD={dd:.2%}")


def guard_position_size(size_usd: float) -> GuardResult:
    """Block positions exceeding per-position size limit."""
    settings = get_settings()
    if size_usd > settings.max_single_position_usd:
        raise PositionSizeTooLarge(
            limit_name="max_single_position",
            current=size_usd,
            limit=settings.max_single_position_usd,
        )
    return GuardResult(passed=True, guard_name="position_size")


def guard_correlation(
    market: Market,
    size_usd: float,
    portfolio: PortfolioManager,
) -> GuardResult:
    """Block positions that breach factor correlation limits."""
    nav = portfolio.nav
    allowed, reason = portfolio.check_can_open(market, size_usd, nav)
    if not allowed:
        raise CorrelationLimitBreached(
            limit_name="factor_concentration",
            current=size_usd / max(nav, 1.0),
            limit=get_settings().max_correlation_exposure,
        )
    return GuardResult(passed=True, guard_name="correlation")


def guard_stale_data(market: Market) -> GuardResult:
    """Block trades on market data older than MAX_MARKET_SNAPSHOT_AGE_SEC."""
    import time
    age = time.monotonic()  # we check against fetched_at in production
    # Use fetched_at datetime for staleness
    age_seconds = (datetime.now(timezone.utc) - market.fetched_at).total_seconds()
    if age_seconds > MAX_MARKET_SNAPSHOT_AGE_SEC:
        raise StaleDataError(market.market_id, age_seconds)
    return GuardResult(
        passed=True,
        guard_name="stale_data",
        reason=f"age={age_seconds:.1f}s",
    )


def guard_probability_bounds(price: float) -> GuardResult:
    """Block trades on near-certain markets (too illiquid / high carry risk)."""
    if not (MIN_TRADEABLE_PROB <= price <= MAX_TRADEABLE_PROB):
        raise RiskLimitBreached(
            limit_name="probability_bounds",
            current=price,
            limit=MAX_TRADEABLE_PROB,
        )
    return GuardResult(passed=True, guard_name="probability_bounds")


def guard_gas_price(current_gwei: float) -> GuardResult:
    """Block Polygon transactions when gas is too expensive."""
    settings = get_settings()
    if current_gwei > settings.gas_price_gwei_max:
        raise GasLimitExceeded(current_gwei, float(settings.gas_price_gwei_max))
    return GuardResult(passed=True, guard_name="gas_price")


def guard_slippage(expected_slippage: float) -> GuardResult:
    """Block AMM trades with excessive price impact."""
    settings = get_settings()
    if expected_slippage > settings.slippage_tolerance_pct:
        raise SlippageLimitExceeded(expected_slippage, settings.slippage_tolerance_pct)
    return GuardResult(passed=True, guard_name="slippage")


def guard_resolution_risk(
    opportunity: ArbitrageOpportunity,
    max_resolution_risk: float = 0.70,
) -> GuardResult:
    """
    Block arb opportunities where resolution risk is unacceptably high.
    The threshold is: if risk_level > 0.7, treat as fundamentally unsafe.
    """
    if opportunity.resolution_risk and opportunity.resolution_risk.risk_level > max_resolution_risk:
        raise RiskLimitBreached(
            limit_name="resolution_risk",
            current=opportunity.resolution_risk.risk_level,
            limit=max_resolution_risk,
        )
    return GuardResult(passed=True, guard_name="resolution_risk")


def guard_fee_consumption(
    gross_edge_usd: float,
    total_costs_usd: float,
) -> GuardResult:
    """Block trades where fees consume too large a fraction of gross edge."""
    settings = get_settings()
    if gross_edge_usd <= 0:
        raise RiskLimitBreached(
            limit_name="fee_consumption",
            current=1.0,
            limit=settings.fee_edge_max_consumption,
        )
    consumption = total_costs_usd / gross_edge_usd
    if consumption > settings.fee_edge_max_consumption:
        raise RiskLimitBreached(
            limit_name="fee_consumption",
            current=consumption,
            limit=settings.fee_edge_max_consumption,
        )
    return GuardResult(passed=True, guard_name="fee_consumption")


def guard_liquidity(
    volume_24h_usd: float,
    min_volume: float = MIN_MARKET_VOLUME_24H_USD,
) -> GuardResult:
    """
    Block trades on illiquid markets where slippage/spread would eat edge.

    A market with < $1,000 daily volume cannot absorb even a $50 order without
    significant price impact not captured by the CLOB snapshot.
    """
    if volume_24h_usd < min_volume:
        raise RiskLimitBreached(
            limit_name="min_liquidity",
            current=volume_24h_usd,
            limit=min_volume,
        )
    return GuardResult(
        passed=True,
        guard_name="liquidity",
        reason=f"vol24h=${volume_24h_usd:,.0f}",
    )


def guard_aroc(aroc_annual: float) -> GuardResult:
    """Block positions below minimum annualised return threshold."""
    settings = get_settings()
    if aroc_annual < settings.aroc_minimum_annual:
        raise RiskLimitBreached(
            limit_name="aroc_minimum",
            current=aroc_annual,
            limit=settings.aroc_minimum_annual,
        )
    return GuardResult(passed=True, guard_name="aroc")


# ---------------------------------------------------------------------------
# Composite Guard Runner
# ---------------------------------------------------------------------------

class RiskGuardRunner:
    """
    Runs all applicable guards for a given trade action.
    Raises the FIRST exception encountered (fail-fast).
    All results are logged regardless.
    """

    def run_directional(
        self,
        signal: DirectionalSignal,
        market: Market,
        portfolio: PortfolioManager,
        current_gas_gwei: float = 0.0,
        expected_slippage: float = 0.0,
        is_live_order: bool = False,
    ) -> list[GuardResult]:
        """Run all guards for a directional signal. Raises on any failure."""
        results: list[GuardResult] = []

        yes = market.yes_outcome
        vol_24h = yes.volume_24h if yes else 0.0

        guards_to_run = [
            lambda: guard_paper_mode(is_live_order),
            lambda: guard_drawdown(portfolio),
            lambda: guard_position_size(signal.recommended_size_usd),
            lambda: guard_stale_data(market),
            lambda: guard_probability_bounds(signal.implied_probability),
            lambda: guard_liquidity(vol_24h),
            lambda: guard_correlation(market, signal.recommended_size_usd, portfolio),
            lambda: guard_aroc(signal.aroc_annual),
        ]

        if market.exchange == Exchange.POLYMARKET:
            guards_to_run.append(lambda: guard_gas_price(current_gas_gwei))
            guards_to_run.append(lambda: guard_slippage(expected_slippage))

        fee_cost = signal.recommended_size_usd * market.taker_fee
        guards_to_run.append(
            lambda: guard_fee_consumption(
                signal.expected_value_usd + fee_cost, fee_cost
            )
        )

        for guard_fn in guards_to_run:
            result = guard_fn()
            results.append(result)
            logger.debug("Guard %s: PASSED (%s)", result.guard_name, result.reason)

        return results

    def run_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        markets: list[Market],
        portfolio: PortfolioManager,
        current_gas_gwei: float = 0.0,
        is_live_order: bool = False,
    ) -> list[GuardResult]:
        """Run all guards for an arbitrage opportunity."""
        results: list[GuardResult] = []

        results.append(guard_paper_mode(is_live_order))
        results.append(guard_drawdown(portfolio))
        results.append(guard_position_size(opportunity.required_capital_usd))
        results.append(guard_resolution_risk(opportunity))
        results.append(guard_aroc(opportunity.aroc_annual))
        results.append(
            guard_fee_consumption(
                opportunity.gross_edge_usd,
                opportunity.fee_cost_usd + opportunity.gas_cost_usd,
            )
        )

        for market in markets:
            results.append(guard_stale_data(market))
            yes = market.yes_outcome
            if yes:
                results.append(guard_liquidity(yes.volume_24h))
            if market.exchange == Exchange.POLYMARKET:
                results.append(guard_gas_price(current_gas_gwei))

        for r in results:
            logger.debug("Guard %s: PASSED (%s)", r.guard_name, r.reason)

        return results

    def preview(
        self,
        signal: DirectionalSignal,
        market: Market,
        portfolio: PortfolioManager,
    ) -> dict[str, str]:
        """
        Non-raising preview: run each guard and collect pass/fail status.
        Useful for dashboard display and paper-trade mode diagnostics.
        """
        checks: dict[str, str] = {}
        guard_fns = {
            "drawdown": lambda: guard_drawdown(portfolio),
            "position_size": lambda: guard_position_size(signal.recommended_size_usd),
            "stale_data": lambda: guard_stale_data(market),
            "probability_bounds": lambda: guard_probability_bounds(signal.implied_probability),
            "correlation": lambda: guard_correlation(market, signal.recommended_size_usd, portfolio),
            "aroc": lambda: guard_aroc(signal.aroc_annual),
        }
        for name, fn in guard_fns.items():
            try:
                fn()
                checks[name] = "PASS"
            except Exception as exc:
                checks[name] = f"FAIL: {exc}"

        return checks
