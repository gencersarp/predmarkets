"""
Paper Trading Execution Adapter.

Simulates the full order lifecycle without real money:
  - Order placement → logged to SQLite, status set to FILLED (instant fill model)
  - All orders carry is_paper=True
  - Raises PaperModeViolation if is_paper=False is attempted

Instant-fill model:
  - IOC/FOK/market orders fill immediately at the market mid price
  - GTC/limit orders fill if price <= ask (BUY) or price >= bid (SELL)

This gives a conservative simulation: no partial fills, no queue priority.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.core.exceptions import OrderRejected, PaperModeViolation
from src.core.models import (
    Exchange,
    Market,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionStatus,
    Side,
)
from src.execution.base import ExchangeAdapter

logger = logging.getLogger(__name__)


class PaperExchangeAdapter(ExchangeAdapter):
    """
    Paper-trading adapter. Stores orders in memory; logs to DB via callback.

    Usage:
        adapter = PaperExchangeAdapter(exchange=Exchange.KALSHI, initial_balance=1000.0)
        order = await adapter.place_order(...)
    """

    def __init__(
        self,
        exchange: Exchange,
        initial_balance_usd: float = 1000.0,
        db_log_callback=None,  # async callable(order) for DB persistence
        taker_fee: float = 0.0,
    ) -> None:
        self._exchange = exchange
        self._balance = initial_balance_usd
        self._orders: dict[str, Order] = {}
        self._positions: list[Position] = []
        self._db_log = db_log_callback
        self._taker_fee = taker_fee

    @property
    def name(self) -> str:
        return f"paper:{self._exchange.value}"

    # ---------------------------------------------------------------- ExchangeAdapter

    async def place_order(
        self,
        market_id: str,
        side: Side,
        order_side: OrderSide,
        order_type: OrderType,
        price: float,
        size_usd: float,
        is_paper: bool = True,
        market: Optional[Market] = None,
    ) -> Order:
        if not is_paper:
            raise PaperModeViolation()

        order = Order(
            order_id=str(uuid.uuid4()),
            exchange=self._exchange,
            market_id=market_id,
            side=side,
            order_side=order_side,
            order_type=order_type,
            price=price,
            size_usd=size_usd,
            status=OrderStatus.PENDING,
            is_paper=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Simulate fill
        fill_price = self._simulate_fill_price(order, market)

        if fill_price is not None:
            # We no longer deduct from self._balance here.
            # Cash management is now centralized in PortfolioManager.
            # This adapter just provides the fill price and status.
            order.status = OrderStatus.FILLED
            order.filled_size_usd = size_usd
            order.avg_fill_price = fill_price
        else:
            # Limit order not immediately fillable
            if order_type in (OrderType.IOC, OrderType.FOK):
                order.status = OrderStatus.CANCELLED
                logger.debug("IOC/FOK order %s not filled — cancelled", order.order_id[:8])
            else:
                # GTC: remains open
                order.status = OrderStatus.OPEN

        order.updated_at = datetime.now(timezone.utc)
        self._orders[order.order_id] = order

        logger.info(
            "PAPER %s %s %s $%.2f @ %.3f → %s",
            order_side.value.upper(),
            side.value.upper(),
            market_id[:32],
            size_usd,
            price,
            order.status.value,
        )

        if self._db_log and order.status == OrderStatus.FILLED:
            try:
                await self._db_log(order)
            except Exception as exc:
                logger.warning("DB log failed: %s", exc)

        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.OPEN:
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc)
        return True

    async def get_order_status(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if not order:
            raise OrderRejected(f"Order {order_id} not found", order_id)
        return order

    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        orders = [o for o in self._orders.values() if o.status == OrderStatus.OPEN]
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        return orders

    async def get_positions(self) -> list[Position]:
        return list(self._positions)

    async def get_balance_usd(self) -> float:
        return self._balance

    # ---------------------------------------------------------------- Simulation

    def _simulate_fill_price(
        self, order: Order, market: Optional[Market]
    ) -> Optional[float]:
        """
        Determine fill price for a paper order.

        For taker orders (IOC/FOK/market): fill at current ask (buy) or bid (sell).
        For maker orders (GTC/limit): fill if price crosses the book.
        """
        if market is None:
            # No live market data: assume fills at stated price
            return order.price

        yes = market.yes_outcome
        no = market.no_outcome

        if order.side == Side.YES:
            outcome = yes
        else:
            outcome = no

        if outcome is None:
            return order.price

        if order.order_type in (OrderType.IOC, OrderType.FOK, OrderType.MARKET):
            # IOC/FOK in paper mode: fill at the current market price (taker).
            # We don't enforce a limit price check here — the signal already
            # passed risk guards and the edge calculation used the ask price.
            if order.order_side == OrderSide.BUY:
                return outcome.implied_prob_ask
            else:
                return outcome.implied_prob_bid
        else:
            # GTC limit order: fill if price is at or better than book
            if order.order_side == OrderSide.BUY and order.price >= outcome.implied_prob_ask:
                return outcome.implied_prob_ask
            elif order.order_side == OrderSide.SELL and order.price <= outcome.implied_prob_bid:
                return outcome.implied_prob_bid
            return None  # not yet fillable

    # ---------------------------------------------------------------- Diagnostics

    def portfolio_summary(self) -> dict:
        filled_orders = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        total_traded_usd = sum(o.filled_size_usd for o in filled_orders)
        return {
            "balance_usd": self._balance,
            "total_orders": len(self._orders),
            "filled_orders": len(filled_orders),
            "total_traded_usd": total_traded_usd,
            "open_orders": len([o for o in self._orders.values() if o.status == OrderStatus.OPEN]),
        }
