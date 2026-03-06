"""
Abstract base class for exchange execution adapters.

Every exchange connector must implement this interface.
The router calls these methods; strategy code never touches exchange APIs directly.
"""
from __future__ import annotations

import abc
from typing import Optional

from src.core.models import (
    Market,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Side,
)


class ExchangeAdapter(abc.ABC):
    """
    Abstract exchange adapter. All exchange-specific logic lives in subclasses.

    Design principles:
    - All methods are async (non-blocking)
    - No side effects on failure — raise, never silently swallow
    - Every order must have is_paper set correctly before submission
    - Private keys are NEVER stored in this class (injected via secrets)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Exchange name (for logging/display)."""

    @abc.abstractmethod
    async def place_order(
        self,
        market_id: str,
        side: Side,
        order_side: OrderSide,
        order_type: OrderType,
        price: float,
        size_usd: float,
        is_paper: bool = True,
    ) -> Order:
        """
        Place an order on the exchange.

        Args:
            market_id:  Exchange-native market identifier
            side:       YES or NO outcome
            order_side: BUY or SELL
            order_type: LIMIT, IOC, FOK, GTC, etc.
            price:      Price per $1 payout (0.0 to 1.0)
            size_usd:   Total USD value of the order
            is_paper:   If True, log the order but do NOT sign/broadcast

        Returns:
            Order object with status PENDING or FILLED (for immediate fills)

        Raises:
            OrderRejected, SlippageLimitExceeded, GasLimitExceeded, PaperModeViolation
        """

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled, False if already filled."""

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Fetch the current status of an order."""

    @abc.abstractmethod
    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        """List all open orders, optionally filtered by market."""

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch current on-exchange positions (balances for Polymarket CTF tokens)."""

    @abc.abstractmethod
    async def get_balance_usd(self) -> float:
        """Return available USD/USDC balance."""

    async def cancel_all_orders(self, market_id: Optional[str] = None) -> int:
        """
        Cancel all open orders. Returns count of cancelled orders.
        Default implementation calls cancel_order() in sequence;
        override with a batch endpoint if available.
        """
        orders = await self.get_open_orders(market_id)
        count = 0
        for order in orders:
            if order.status == OrderStatus.OPEN:
                cancelled = await self.cancel_order(order.order_id)
                if cancelled:
                    count += 1
        return count

    async def place_maker_order(
        self,
        market_id: str,
        side: Side,
        order_side: OrderSide,
        price: float,
        size_usd: float,
        is_paper: bool = True,
    ) -> Order:
        """
        Convenience: place a GTC limit order at the maker side.
        Uses the exchange's maker fee (lower cost).
        """
        return await self.place_order(
            market_id=market_id,
            side=side,
            order_side=order_side,
            order_type=OrderType.GTC,
            price=price,
            size_usd=size_usd,
            is_paper=is_paper,
        )

    async def place_ioc_order(
        self,
        market_id: str,
        side: Side,
        order_side: OrderSide,
        price: float,
        size_usd: float,
        is_paper: bool = True,
    ) -> Order:
        """
        Convenience: IOC order (hide intent, anti-front-running).
        """
        return await self.place_order(
            market_id=market_id,
            side=side,
            order_side=order_side,
            order_type=OrderType.IOC,
            price=price,
            size_usd=size_usd,
            is_paper=is_paper,
        )
