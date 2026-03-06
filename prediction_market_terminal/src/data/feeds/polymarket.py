"""
Polymarket data feed — REST + WebSocket.

Ingests:
  - Active markets from Gamma API (metadata, resolution rules)
  - Order book snapshots from CLOB API
  - AMM reserves from Polygon on-chain state
  - WebSocket live order book updates
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import aiohttp
import websockets
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider

from config.settings import get_settings
from src.core.constants import (
    MAX_MARKET_SNAPSHOT_AGE_SEC,
    POLYMARKET_MAKER_FEE,
    POLYMARKET_TAKER_FEE,
)
from src.core.exceptions import FeedError, StaleDataError
from src.core.models import (
    Exchange,
    Market,
    MarketOutcome,
    OrderBook,
    ResolutionSource,
    Side,
)

logger = logging.getLogger(__name__)

# Minimal ERC-1155 / CTF ABI for reserve reads
_CTF_ABI = [
    {
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]

# Polymarket CLOB WebSocket endpoint
_WS_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/"


class PolymarketFeed:
    """
    Async feed for Polymarket. Maintains an internal cache of Market objects.
    Call `start()` to begin background refresh; `stop()` to shut down.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._w3: Optional[AsyncWeb3] = None
        self._markets: dict[str, Market] = {}       # market_id -> Market
        self._last_refresh: dict[str, float] = {}   # market_id -> epoch
        self._ws_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # ---------------------------------------------------------------- Lifecycle

    async def start(self) -> None:
        """Initialise HTTP session, Web3 provider, and start WS listener."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "PMT/1.0"},
        )
        self._w3 = AsyncWeb3(AsyncHTTPProvider(self._settings.polygon_rpc_url))
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_listener())
        logger.info("PolymarketFeed started")

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()
        logger.info("PolymarketFeed stopped")

    # ---------------------------------------------------------------- Public API

    async def fetch_active_markets(
        self,
        category: Optional[str] = None,
        limit: int = 200,
    ) -> list[Market]:
        """Fetch and cache active markets from Gamma API."""
        params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": limit,
        }
        if category:
            params["category"] = category

        raw = await self._gamma_get("/markets", params)
        markets: list[Market] = []
        for m in raw:
            try:
                market = self._normalise_market(m)
                self._markets[market.market_id] = market
                self._last_refresh[market.market_id] = time.monotonic()
                markets.append(market)
            except Exception as exc:
                logger.warning("Failed to parse market %s: %s", m.get("id"), exc)
        logger.info("Fetched %d active Polymarket markets", len(markets))
        return markets

    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Fetch Level-2 order book for a single market from CLOB."""
        raw = await self._clob_get(f"/book", {"token_id": market_id})
        return self._normalise_order_book(raw)

    async def get_market(self, market_id: str) -> Market:
        """Return cached market, refreshing if stale."""
        age = time.monotonic() - self._last_refresh.get(market_id, 0)
        if age > MAX_MARKET_SNAPSHOT_AGE_SEC or market_id not in self._markets:
            await self._refresh_single(market_id)
        market = self._markets.get(market_id)
        if market is None:
            from src.core.exceptions import MarketNotFoundError
            raise MarketNotFoundError(market_id)
        return market

    async def stream_order_book_updates(
        self, market_ids: list[str]
    ) -> AsyncIterator[tuple[str, OrderBook]]:
        """
        Yield (market_id, OrderBook) tuples from the WebSocket subscription.
        This is a thin wrapper — the main WS loop processes and caches internally.
        """
        # The WS listener updates self._markets; this async generator
        # just yields whenever the loop ticks. In production, use an
        # asyncio.Queue for push-based delivery.
        while self._running:
            for mid in market_ids:
                if mid in self._markets:
                    m = self._markets[mid]
                    if m.yes_outcome and m.yes_outcome.order_book:
                        yield mid, m.yes_outcome.order_book
            await asyncio.sleep(0.5)

    # ---------------------------------------------------------------- Internals

    async def _refresh_single(self, market_id: str) -> None:
        raw_list = await self._gamma_get(f"/markets/{market_id}", {})
        # Gamma returns a single object for direct ID lookup
        raw = raw_list if isinstance(raw_list, dict) else (raw_list[0] if raw_list else None)
        if raw:
            market = self._normalise_market(raw)
            self._markets[market.market_id] = market
            self._last_refresh[market.market_id] = time.monotonic()

    def _normalise_market(self, raw: dict[str, Any]) -> Market:
        """Convert raw Gamma API response to canonical Market model."""
        market_id = str(raw["id"])

        # Parse expiry
        expiry: Optional[datetime] = None
        if end_date := raw.get("endDate") or raw.get("end_date_iso"):
            try:
                expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Outcomes
        outcomes: list[MarketOutcome] = []
        tokens = raw.get("tokens", [])
        for token in tokens:
            side = Side.YES if token.get("outcome", "").lower() == "yes" else Side.NO
            # Prices come as floats 0-1 from Gamma
            price = float(token.get("price", 0.5))
            outcomes.append(
                MarketOutcome(
                    outcome_id=str(token.get("token_id", f"{market_id}_{side.value}")),
                    side=side,
                    implied_prob_bid=max(0.0, price - 0.01),
                    implied_prob_ask=min(1.0, price + 0.01),
                    amm_token_address=token.get("token_id"),
                )
            )

        # Resolution source heuristic
        res_source = ResolutionSource.UMA_ORACLE
        criteria = raw.get("resolutionSource", "") or raw.get("resolution", "")
        if "uma" in criteria.lower():
            res_source = ResolutionSource.UMA_ORACLE

        return Market(
            market_id=market_id,
            exchange=Exchange.POLYMARKET,
            title=raw.get("question", raw.get("title", "")),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            resolution_source=res_source,
            resolution_criteria=criteria,
            expiry=expiry,
            outcomes=outcomes,
            taker_fee=POLYMARKET_TAKER_FEE,
            maker_fee=POLYMARKET_MAKER_FEE,
            raw_data=raw,
        )

    def _normalise_order_book(self, raw: dict[str, Any]) -> OrderBook:
        """Convert CLOB /book response to OrderBook."""
        def parse_levels(levels: list[dict[str, Any]]) -> list[tuple[float, float]]:
            result = []
            for lvl in levels:
                price = float(lvl.get("price", 0))
                size = float(lvl.get("size", 0))
                result.append((price, size))
            return sorted(result, key=lambda x: x[0], reverse=True)  # desc for bids

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            bids=parse_levels(raw.get("bids", [])),
            asks=sorted(parse_levels(raw.get("asks", [])), key=lambda x: x[0]),
        )

    async def _gamma_get(self, path: str, params: dict[str, Any]) -> Any:
        assert self._session is not None
        url = f"{self._settings.polymarket_gamma_endpoint}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise FeedError(f"Gamma API {path} returned {resp.status}: {text}")
            return await resp.json()

    async def _clob_get(self, path: str, params: dict[str, Any]) -> Any:
        assert self._session is not None
        url = f"{self._settings.polymarket_clob_endpoint}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise FeedError(f"CLOB API {path} returned {resp.status}: {text}")
            return await resp.json()

    async def _ws_listener(self) -> None:
        """
        Maintain a persistent WebSocket connection to the CLOB.
        On disconnect, exponentially back off and reconnect.
        """
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    _WS_ENDPOINT + "market",
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    backoff = 1.0  # reset on successful connect
                    logger.info("Polymarket WS connected")

                    # Subscribe to all tracked markets
                    if self._markets:
                        sub_msg = {
                            "type": "subscribe",
                            "channel": "live_activity",
                            "markets": list(self._markets.keys())[:50],
                        }
                        await ws.send(json.dumps(sub_msg))

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            self._process_ws_message(msg)
                        except Exception as exc:
                            logger.debug("WS parse error: %s", exc)

            except (websockets.ConnectionClosed, OSError) as exc:
                logger.warning("Polymarket WS disconnected: %s. Retrying in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _process_ws_message(self, msg: dict[str, Any]) -> None:
        """Update cached market from WebSocket tick."""
        event_type = msg.get("type")
        if event_type not in ("book", "price_change", "last_trade_price"):
            return

        market_id = msg.get("market") or msg.get("asset_id")
        if not market_id or market_id not in self._markets:
            return

        market = self._markets[market_id]
        outcome = market.yes_outcome
        if not outcome:
            return

        if event_type == "book":
            outcome.order_book = OrderBook(
                timestamp=datetime.now(timezone.utc),
                bids=[(float(b["price"]), float(b["size"])) for b in msg.get("bids", [])],
                asks=[(float(a["price"]), float(a["size"])) for a in msg.get("asks", [])],
            )
            if outcome.order_book.best_bid:
                outcome.implied_prob_bid = outcome.order_book.best_bid
            if outcome.order_book.best_ask:
                outcome.implied_prob_ask = outcome.order_book.best_ask
        elif event_type == "price_change":
            price = float(msg.get("price", outcome.implied_prob_bid))
            outcome.implied_prob_bid = price - 0.005
            outcome.implied_prob_ask = price + 0.005

        self._last_refresh[market_id] = time.monotonic()


# ---- AMM Math (Constant Product) ------------------------------------------

def amm_spot_price(reserve_yes: float, reserve_no: float) -> float:
    """
    Implied probability of YES from CPMM reserves.
    P(YES) = reserve_no / (reserve_yes + reserve_no)
    """
    total = reserve_yes + reserve_no
    if total == 0:
        return 0.5
    return reserve_no / total


def amm_buy_cost(
    reserve_yes: float,
    reserve_no: float,
    outcome: str,   # "yes" or "no"
    shares_out: float,
) -> float:
    """
    Exact USDC cost to buy `shares_out` outcome shares from CPMM.
    Uses constant product invariant: R_yes * R_no = k
    """
    k = reserve_yes * reserve_no
    if outcome.lower() == "yes":
        new_reserve_yes = reserve_yes - shares_out
        if new_reserve_yes <= 0:
            raise ValueError("Insufficient AMM liquidity for this trade size")
        new_reserve_no = k / new_reserve_yes
        return new_reserve_no - reserve_no
    else:
        new_reserve_no = reserve_no - shares_out
        if new_reserve_no <= 0:
            raise ValueError("Insufficient AMM liquidity for this trade size")
        new_reserve_yes = k / new_reserve_no
        return new_reserve_yes - reserve_yes


def amm_price_impact(
    reserve_yes: float,
    reserve_no: float,
    outcome: str,
    trade_size_usd: float,
) -> float:
    """
    Return price impact (slippage) as a fraction of the spot price,
    given a trade of `trade_size_usd` USDC.
    """
    spot = amm_spot_price(reserve_yes, reserve_no)
    if spot == 0 or spot == 1:
        return 1.0  # degenerate pool
    # Approximate shares received = trade_size / spot_price
    approx_shares = trade_size_usd / spot
    try:
        actual_cost = amm_buy_cost(reserve_yes, reserve_no, outcome, approx_shares)
        effective_price = actual_cost / approx_shares
        return abs(effective_price - spot) / spot
    except ValueError:
        return 1.0  # trade too large for pool
