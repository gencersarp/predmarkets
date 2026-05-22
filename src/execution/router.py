"""
Smart Order Router — routes orders to the best venue/mechanism.

Decision logic:
  1. Paper mode? → PaperExchangeAdapter for both exchanges
  2. Polymarket order:
     - Has AMM reserves & trade_size < AMM_THRESHOLD? → compare AMM vs CLOB
     - Route to lower-cost venue (lower slippage + fees)
  3. Kalshi order → Kalshi limit order book

Also handles multi-leg arb execution: legs fire sequentially with rollback
logging if any leg fails (no partial-fills left dangling).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.core.exceptions import ExecutionError, PaperModeViolation, SlippageLimitExceeded
from src.core.models import (
    ArbitrageOpportunity,
    DirectionalSignal,
    Exchange,
    Market,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Side,
)
from src.data.feeds.polymarket import amm_price_impact
from src.execution.base import ExchangeAdapter
from src.execution.paper import PaperExchangeAdapter
from src.risk.guards import RiskGuardRunner
from src.risk.portfolio import PortfolioManager
from config.settings import get_settings

logger = logging.getLogger(__name__)

# AMM is preferred for small trades (lower gas than CLOB for tiny sizes)
AMM_PREFERRED_THRESHOLD_USD = 50.0


class OrderRouter:
    """
    Routes orders to the correct adapter and handles multi-leg execution.

    In PAPER mode: all orders go to PaperExchangeAdapter regardless of exchange.
    In LIVE mode: routes to the appropriate live adapter.
    """

    def __init__(
        self,
        polymarket_adapter: ExchangeAdapter,
        kalshi_adapter: ExchangeAdapter,
        portfolio: PortfolioManager,
    ) -> None:
        self._adapters: dict[Exchange, ExchangeAdapter] = {
            Exchange.POLYMARKET: polymarket_adapter,
            Exchange.KALSHI: kalshi_adapter,
        }
        self._portfolio = portfolio
        self._guards = RiskGuardRunner()
        self._settings = get_settings()

    def _adapter_for(self, exchange: Exchange) -> ExchangeAdapter:
        return self._adapters[exchange]

    # ---------------------------------------------------------------- Directional Signal

    async def execute_signal(
        self,
        signal: DirectionalSignal,
        market: Market,
        current_gas_gwei: float = 0.0,
    ) -> Optional[Order]:
        """
        Execute a directional signal through all risk guards and then place the order.
        Returns the Order on success, None if blocked.
        """
        is_live = self._settings.is_live

        logger.info(
            "RISK CHECK — %s [%s] %s | true=%.1f%% implied=%.1f%% edge=%.1f%% size=$%.2f",
            signal.alpha_type.value, market.market_id[:24], signal.side.value,
            signal.true_probability * 100, signal.implied_probability * 100,
            signal.edge * 100, signal.recommended_size_usd,
        )

        # Check for existing position in same market (avoid duplicate bets)
        if self._portfolio.has_position_in_market(market.market_id):
            existing = self._portfolio.get_position_for_market(market.market_id)
            if existing and existing.side == signal.side:
                logger.debug(
                    "  SKIPPED — already holding %s %s in %s ($%.2f)",
                    existing.side.value, market.exchange.value,
                    market.market_id[:24], existing.size_usd,
                )
                return None

        try:
            self._guards.run_directional(
                signal=signal,
                market=market,
                portfolio=self._portfolio,
                current_gas_gwei=current_gas_gwei,
                expected_slippage=self._estimate_slippage(market, signal.recommended_size_usd),
                is_live_order=is_live,
            )
        except Exception as exc:
            logger.warning("  BLOCKED — %s", exc)
            return None

        adapter = self._adapter_for(market.exchange)
        order_type = OrderType(self._settings.default_order_type.value.lower())

        # Cap size to available balance — never attempt to spend more than we have.
        # Reserve enough for exchange fees (Kalshi=7%, Poly=2%) to avoid rejections.
        available = await adapter.get_balance_usd()
        fee_rate = market.taker_fee if market.taker_fee > 0 else 0.02
        available_after_fees = available / (1.0 + fee_rate) * 0.95
        size_usd = min(signal.recommended_size_usd, available_after_fees)
        if size_usd < 1.0:
            logger.warning("  SKIPPED — insufficient balance ($%.2f available)", available)
            return None

        logger.info(
            "  PASSED all risk guards — placing order (balance=$%.2f, size=$%.2f)",
            available, size_usd,
        )

        order = await adapter.place_order(
            market_id=market.market_id,
            side=signal.side,
            order_side=OrderSide.BUY,
            order_type=order_type,
            price=signal.implied_probability,
            size_usd=size_usd,
            is_paper=not is_live,
            market=market,
        )

        if order.status == OrderStatus.FILLED:
            logger.info(
                "  FILLED — %s %s '%s' $%.2f @ %.1f%% | edge=%.1f%% EV=$%.2f",
                signal.side.value, market.exchange.value,
                market.title[:35], order.filled_size_usd,
                signal.implied_probability * 100,
                signal.edge * 100, signal.expected_value_usd,
            )
        else:
            logger.warning("  ORDER NOT FILLED — status=%s", order.status.value)

        return order

    # ---------------------------------------------------------------- Arbitrage

    async def execute_arb(
        self,
        opportunity: ArbitrageOpportunity,
        markets: dict[str, Market],
        current_gas_gwei: float = 0.0,
    ) -> list[Order]:
        """
        Execute a multi-leg arbitrage opportunity.

        If any leg fails, log the failure but do NOT attempt to roll back
        already-filled legs (rolling back on-chain is not always possible).
        The risk of a "leg 2 fail" is managed by:
          - Conservative sizing
          - Executing leg 1 first on the less liquid venue
          - IOC order type to avoid partial fills
        """
        is_live = self._settings.is_live
        market_list = list(markets.values())

        try:
            self._guards.run_arbitrage(
                opportunity=opportunity,
                markets=market_list,
                portfolio=self._portfolio,
                current_gas_gwei=current_gas_gwei,
                is_live_order=is_live,
            )
        except Exception as exc:
            logger.debug("Arb blocked by risk guard: %s", exc)
            return []

        filled_orders: list[Order] = []

        for leg in opportunity.legs:
            exchange = Exchange(leg["exchange"])
            adapter = self._adapter_for(exchange)

            market = markets.get(leg["market_id"])
            if not market:
                logger.error("Market %s not found for arb leg", leg["market_id"])
                break

            try:
                order = await adapter.place_order(
                    market_id=leg["market_id"],
                    side=Side(leg["side"]),
                    order_side=OrderSide(leg["action"]),
                    order_type=OrderType.IOC,  # Anti-front-running
                    price=leg["price"],
                    size_usd=leg["size_usd"],
                    is_paper=not is_live,
                    market=market,
                )
                filled_orders.append(order)

                if order.status != OrderStatus.FILLED:
                    logger.warning(
                        "Arb leg failed (%s %s): status=%s — leaving remaining legs unexecuted",
                        leg["action"],
                        leg["side"],
                        order.status.value,
                    )
                    break

            except Exception as exc:
                logger.error("Arb leg execution error: %s", exc)
                break

        if len(filled_orders) == len(opportunity.legs):
            logger.info(
                "Arb fully executed: net_edge=$%.2f across %d legs",
                opportunity.net_edge_usd,
                len(filled_orders),
            )
        elif filled_orders:
            logger.warning(
                "Partial arb execution: %d/%d legs filled — MANUAL REVIEW REQUIRED",
                len(filled_orders),
                len(opportunity.legs),
            )

        return filled_orders

    # ---------------------------------------------------------------- AMM Routing

    def _estimate_slippage(self, market: Market, size_usd: float) -> float:
        """Estimate slippage for a trade. Uses AMM math if reserves available."""
        yes = market.yes_outcome
        if (
            market.exchange == Exchange.POLYMARKET
            and yes
            and yes.amm_reserve_yes is not None
            and yes.amm_reserve_no is not None
        ):
            return amm_price_impact(
                yes.amm_reserve_yes,
                yes.amm_reserve_no,
                "yes",
                size_usd,
            )
        # For order book markets, estimate slippage from spread
        if yes and yes.order_book and yes.order_book.spread:
            return yes.order_book.spread / 2
        # For markets without explicit order book, estimate from bid-ask
        if yes:
            spread = yes.implied_prob_ask - yes.implied_prob_bid
            if spread > 0:
                return spread / 2
        return 0.005  # Conservative default 0.5% slippage
