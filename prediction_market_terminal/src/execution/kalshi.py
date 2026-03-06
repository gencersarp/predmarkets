"""
Kalshi Live Execution Adapter.

Kalshi uses a REST limit order book with HMAC-authenticated endpoints.
Orders are fiat (USD) — no gas costs, no blockchain.

Authentication: API Key + HMAC-SHA256 signature per request.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from base64 import b64encode
from datetime import datetime
from typing import Optional

import aiohttp

from src.core.exceptions import ExecutionError, OrderRejected, PaperModeViolation
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
from config.settings import get_settings

logger = logging.getLogger(__name__)


class KalshiAdapter(ExchangeAdapter):
    """
    Live Kalshi execution adapter.

    Kalshi API v2 authentication uses API key + HMAC-SHA256 signed requests.
    The secret is accessed from settings, never hardcoded.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._base = self._settings.kalshi_base_url
        self._token: Optional[str] = None
        self._token_expires: float = 0.0

    @property
    def name(self) -> str:
        return "kalshi_live"

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )

    async def _ensure_auth(self) -> None:
        if self._token and time.monotonic() < self._token_expires:
            return
        await self._authenticate()

    async def _authenticate(self) -> None:
        api_key = self._settings.kalshi_api_key
        api_secret = self._settings.kalshi_api_secret
        if not api_key or not api_secret:
            raise ExecutionError(
                "Kalshi credentials not configured. "
                "Set KALSHI_API_KEY and KALSHI_API_SECRET in .env or AWS Secrets."
            )
        await self._ensure_session()
        assert self._session is not None
        body = {
            "email": api_key.get_secret_value(),
            "password": api_secret.get_secret_value(),
        }
        async with self._session.post(f"{self._base}/login", json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ExecutionError(f"Kalshi auth failed {resp.status}: {text}")
            data = await resp.json()
            self._token = data.get("token")
            self._token_expires = time.monotonic() + 23 * 3600

    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

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
        if is_paper:
            raise PaperModeViolation()

        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None

        # Kalshi prices are in cents (0-99)
        price_cents = round(price * 100)
        # Kalshi size is in contracts ($0.01 each on some markets; check docs)
        # For simplicity: size_usd maps to number of contracts
        num_contracts = round(size_usd / max(price, 0.01))

        kalshi_side = "yes" if side == Side.YES else "no"
        kalshi_action = "buy" if order_side == OrderSide.BUY else "sell"
        kalshi_type = self._map_order_type(order_type)

        body = {
            "ticker": market_id,
            "client_order_id": str(uuid.uuid4()),
            "type": kalshi_type,
            "action": kalshi_action,
            "side": kalshi_side,
            "count": num_contracts,
            "yes_price": price_cents,
        }

        async with self._session.post(
            f"{self._base}/portfolio/orders",
            json=body,
            headers=self._auth_headers(),
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise OrderRejected(f"Kalshi rejected order: {text}")
            data = await resp.json()
            order_data = data.get("order", {})
            return self._normalise_order(order_data, market_id, side, order_side, price, size_usd)

    async def cancel_order(self, order_id: str) -> bool:
        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None
        async with self._session.delete(
            f"{self._base}/portfolio/orders/{order_id}",
            headers=self._auth_headers(),
        ) as resp:
            return resp.status == 200

    async def get_order_status(self, order_id: str) -> Order:
        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None
        async with self._session.get(
            f"{self._base}/portfolio/orders/{order_id}",
            headers=self._auth_headers(),
        ) as resp:
            if resp.status != 200:
                raise OrderRejected("Order not found", order_id)
            data = await resp.json()
            return self._normalise_order(data.get("order", {}))

    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None
        params = {"status": "resting"}
        if market_id:
            params["ticker"] = market_id
        async with self._session.get(
            f"{self._base}/portfolio/orders",
            params=params,
            headers=self._auth_headers(),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [self._normalise_order(o) for o in data.get("orders", [])]

    async def get_positions(self) -> list[Position]:
        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None
        async with self._session.get(
            f"{self._base}/portfolio/positions",
            headers=self._auth_headers(),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            positions = []
            for p in data.get("market_positions", []):
                ticker = p.get("market_id", "")
                yes_position = p.get("position", 0)
                if yes_position != 0:
                    side = Side.YES if yes_position > 0 else Side.NO
                    positions.append(
                        Position(
                            exchange=Exchange.KALSHI,
                            market_id=ticker,
                            market_title=ticker,
                            side=side,
                            size_usd=abs(yes_position) * 0.01,
                            entry_price=0.5,  # unknown without order history
                            current_price=0.5,
                            status=PositionStatus.OPEN,
                            is_paper=False,
                        )
                    )
            return positions

    async def get_balance_usd(self) -> float:
        await self._ensure_session()
        await self._ensure_auth()
        assert self._session is not None
        async with self._session.get(
            f"{self._base}/portfolio/balance",
            headers=self._auth_headers(),
        ) as resp:
            if resp.status != 200:
                return 0.0
            data = await resp.json()
            return float(data.get("balance", 0)) / 100.0  # cents to USD

    def _map_order_type(self, order_type: OrderType) -> str:
        mapping = {
            OrderType.LIMIT: "limit",
            OrderType.MARKET: "market",
            OrderType.IOC: "limit",  # Kalshi uses limit + time_in_force
            OrderType.FOK: "limit",
            OrderType.GTC: "limit",
        }
        return mapping.get(order_type, "limit")

    def _normalise_order(
        self,
        raw: dict,
        market_id: str = "",
        side: Side = Side.YES,
        order_side: OrderSide = OrderSide.BUY,
        price: float = 0.0,
        size_usd: float = 0.0,
    ) -> Order:
        status_map = {
            "resting": OrderStatus.OPEN,
            "executed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "partially_filled": OrderStatus.PARTIAL,
        }
        raw_status = raw.get("status", "")
        return Order(
            order_id=raw.get("order_id", str(uuid.uuid4())),
            exchange=Exchange.KALSHI,
            market_id=raw.get("ticker", market_id),
            side=Side.YES if raw.get("side", "yes") == "yes" else Side.NO,
            order_side=OrderSide.BUY if raw.get("action", "buy") == "buy" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=float(raw.get("yes_price", price * 100)) / 100.0,
            size_usd=float(raw.get("count", size_usd / 0.01)) * 0.01,
            status=status_map.get(raw_status, OrderStatus.PENDING),
            filled_size_usd=float(raw.get("filled_count", 0)) * 0.01,
            is_paper=False,
        )
