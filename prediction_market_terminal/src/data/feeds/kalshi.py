"""
Kalshi data feed — REST polling + WebSocket.

Kalshi uses a limit order book (fiat, not crypto).
Resolution is done internally by Kalshi — NOT via UMA oracle.
This distinction is critical for cross-exchange arb resolution-risk assessment.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from base64 import b64encode
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import aiohttp

from config.settings import get_settings, KalshiEnv
from src.core.constants import KALSHI_MAKER_FEE, KALSHI_TAKER_FEE
from src.core.exceptions import FeedError
from src.core.models import (
    Exchange,
    Market,
    MarketOutcome,
    OrderBook,
    ResolutionSource,
    Side,
)

logger = logging.getLogger(__name__)

_KALSHI_WS = "wss://trading-api.kalshi.com/trade-api/ws/v2"
_KALSHI_DEMO_WS = "wss://demo-api.kalshi.co/trade-api/ws/v2"


class KalshiFeed:
    """
    Async feed for Kalshi markets.

    Authentication: HMAC-SHA256 signed requests using API key / secret.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._markets: dict[str, Market] = {}
        self._last_refresh: dict[str, float] = {}
        self._ws_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False
        self._token: Optional[str] = None        # Kalshi session token
        self._token_expires: float = 0.0

    # ---------------------------------------------------------------- Lifecycle

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
        )
        await self._authenticate()
        self._running = True
        if self._rsa_mode:
            # Kalshi WS requires a Bearer session token, not per-request RSA signatures.
            # RSA key auth is REST-only; fall back to 60s polling for price updates.
            logger.info(
                "KalshiFeed started in REST-only mode (RSA auth — WS requires session token). "
                "Prices refresh every 60s via REST polling."
            )
        else:
            self._ws_task = asyncio.create_task(self._ws_listener())
        logger.info("KalshiFeed started (env=%s)", self._settings.kalshi_env.value)

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._session:
            await self._session.close()
        logger.info("KalshiFeed stopped")

    # ---------------------------------------------------------------- Public API

    async def fetch_active_markets(
        self,
        category: Optional[str] = None,
        limit: int = 200,
    ) -> list[Market]:
        """Fetch active Kalshi markets."""
        params: dict[str, Any] = {"limit": limit, "status": "open"}
        if category:
            params["category"] = category

        raw = await self._get("/markets", params)
        markets_raw = raw.get("markets", [])
        markets: list[Market] = []
        for m in markets_raw:
            try:
                market = self._normalise_market(m)
                self._markets[market.market_id] = market
                self._last_refresh[market.market_id] = time.monotonic()
                markets.append(market)
            except Exception as exc:
                logger.warning("Failed to parse Kalshi market %s: %s", m.get("ticker"), exc)

        logger.info("Fetched %d active Kalshi markets", len(markets))
        return markets

    async def fetch_order_book(self, ticker: str, depth: int = 10) -> OrderBook:
        """Fetch order book for a Kalshi market ticker."""
        raw = await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        return self._normalise_order_book(raw.get("orderbook", {}))

    async def get_market(self, market_id: str) -> Market:
        age = time.monotonic() - self._last_refresh.get(market_id, 0)
        if age > 30 or market_id not in self._markets:
            await self._refresh_single(market_id)
        market = self._markets.get(market_id)
        if market is None:
            from src.core.exceptions import MarketNotFoundError
            raise MarketNotFoundError(market_id)
        return market

    # ---------------------------------------------------------------- Internals

    @property
    def _rsa_mode(self) -> bool:
        """True when using RSA key auth (no password needed — works with Google OAuth)."""
        return bool(
            self._settings.kalshi_api_key and self._settings.kalshi_private_key
        )

    async def _authenticate(self) -> None:
        """
        Authenticate with Kalshi.

        Two supported methods:
          1. RSA key auth — set KALSHI_API_KEY=<Key ID> and KALSHI_PRIVATE_KEY=<PEM>.
             Each request is signed with the private key; no login endpoint needed.
             Use this if you signed up via Google (no password).

          2. Email + password — set KALSHI_API_KEY=email and KALSHI_API_SECRET=password.
             POSTs to /login, gets a 24h session token.

        Falls back gracefully if no credentials are configured.
        """
        if self._rsa_mode:
            # RSA mode: no login step needed — requests are signed per-call.
            logger.info(
                "Kalshi using RSA key auth (key_id=%s)",
                self._settings.kalshi_api_key.get_secret_value()[:8] + "...",  # type: ignore[union-attr]
            )
            return

        api_key = self._settings.kalshi_api_key
        api_secret = self._settings.kalshi_api_secret
        if not api_key or not api_secret:
            logger.warning(
                "Kalshi credentials not configured — no market data. "
                "Set KALSHI_API_KEY + KALSHI_API_SECRET (email/password) "
                "or KALSHI_API_KEY + KALSHI_PRIVATE_KEY (RSA key, for Google OAuth users)."
            )
            return

        body = {
            "email": api_key.get_secret_value(),
            "password": api_secret.get_secret_value(),
        }
        assert self._session is not None
        async with self._session.post(
            f"{self._settings.kalshi_base_url}/login", json=body
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise FeedError(f"Kalshi auth failed {resp.status}: {text}")
            data = await resp.json()
            self._token = data.get("token")
            self._token_expires = time.monotonic() + 23 * 3600
            logger.info("Kalshi authenticated OK (session token)")

    def _auth_headers(self, method: str = "GET", path: str = "") -> dict[str, str]:
        """
        Return auth headers for a request.

        RSA mode: signs each request individually with the private key.
          Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP

        Session token mode: returns Bearer token header.
        """
        if self._rsa_mode:
            return self._rsa_headers(method, path)
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def _rsa_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Generate per-request RSA-SHA256 signed headers for Kalshi API key auth.

        Signing message: <timestamp_ms><METHOD><path>
        e.g. "1709123456789GET/trade-api/v2/markets"
        """
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
        except ImportError:
            raise FeedError(
                "RSA auth requires the 'cryptography' package: pip install cryptography"
            )

        import base64

        key_id = self._settings.kalshi_api_key.get_secret_value()  # type: ignore[union-attr]
        pem = self._settings.kalshi_private_key.get_secret_value()  # type: ignore[union-attr]
        # Support both literal \n (from .env) and actual newlines
        pem = pem.replace("\\n", "\n")

        timestamp_ms = str(int(time.time() * 1000))
        # Path must be just the path portion, no query string, no base URL
        clean_path = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{clean_path}".encode()

        private_key = serialization.load_pem_private_key(pem.encode(), password=None)
        signature = private_key.sign(message, asym_padding.PKCS1v15(), hashes.SHA256())
        sig_b64 = base64.b64encode(signature).decode()

        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        assert self._session is not None

        # Session token mode: refresh if near expiry
        if not self._rsa_mode and self._token and time.monotonic() > self._token_expires:
            await self._authenticate()

        url = f"{self._settings.kalshi_base_url}{path}"
        headers = self._auth_headers("GET", path)

        async with self._session.get(url, params=params, headers=headers) as resp:
            if resp.status == 401:
                if not self._token and not self._rsa_mode:
                    logger.debug(
                        "Kalshi 401 on %s — no credentials configured, returning empty", path
                    )
                    return {}
                if not self._rsa_mode:
                    # Session token expired — re-auth and retry once
                    await self._authenticate()
                    headers = self._auth_headers("GET", path)
                    async with self._session.get(url, params=params, headers=headers) as resp2:
                        resp2.raise_for_status()
                        return await resp2.json()
                # RSA mode 401 = wrong key or wrong signing — don't retry
                text = await resp.text()
                raise FeedError(f"Kalshi RSA auth rejected on {path}: {text}")
            if resp.status != 200:
                text = await resp.text()
                raise FeedError(f"Kalshi GET {path} returned {resp.status}: {text}")
            return await resp.json()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        assert self._session is not None
        url = f"{self._settings.kalshi_base_url}{path}"
        headers = self._auth_headers("POST", path)
        async with self._session.post(url, json=body, headers=headers) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise FeedError(f"Kalshi POST {path} returned {resp.status}: {text}")
            return await resp.json()

    async def _refresh_single(self, ticker: str) -> None:
        raw = await self._get(f"/markets/{ticker}", {})
        market_raw = raw.get("market", {})
        if market_raw:
            market = self._normalise_market(market_raw)
            self._markets[market.market_id] = market
            self._last_refresh[market.market_id] = time.monotonic()

    def _normalise_market(self, raw: dict[str, Any]) -> Market:
        """Normalise Kalshi market into canonical Market model."""
        ticker = raw.get("ticker", "")
        market_id = ticker

        # Expiry
        expiry: Optional[datetime] = None
        for field in ("close_time", "expiration_time", "expected_expiration_time",
                      "expected_expiration_ts", "end_date_iso", "close_date"):
            val = raw.get(field)
            if val:
                try:
                    if isinstance(val, (int, float)):
                        expiry = datetime.fromtimestamp(val, tz=timezone.utc)
                    else:
                        expiry = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                    break
                except (ValueError, OSError):
                    pass

        # Prices — Kalshi uses integer cents (0-99) or floats 0-1
        yes_bid_raw = raw.get("yes_bid", 50)
        yes_ask_raw = raw.get("yes_ask", 50)
        no_bid_raw = raw.get("no_bid", 50)
        no_ask_raw = raw.get("no_ask", 50)

        def to_prob(v: Any) -> float:
            f = float(v)
            return f / 100.0 if f > 1.0 else f

        yes_bid = to_prob(yes_bid_raw)
        yes_ask = to_prob(yes_ask_raw)
        no_bid = to_prob(no_bid_raw)
        no_ask = to_prob(no_ask_raw)

        outcomes = [
            MarketOutcome(
                outcome_id=f"{ticker}_yes",
                side=Side.YES,
                implied_prob_bid=yes_bid,
                implied_prob_ask=yes_ask,
                volume_24h=float(raw.get("volume_24h", 0)),
                open_interest=float(raw.get("open_interest", 0)),
            ),
            MarketOutcome(
                outcome_id=f"{ticker}_no",
                side=Side.NO,
                implied_prob_bid=no_bid,
                implied_prob_ask=no_ask,
            ),
        ]

        # Resolution source: Kalshi resolves internally
        resolution_criteria = raw.get("settlement_sources", "") or raw.get("rules_primary", "")

        return Market(
            market_id=market_id,
            exchange=Exchange.KALSHI,
            title=raw.get("title", raw.get("subtitle", "")),
            description=raw.get("rules_primary", ""),
            category=raw.get("category", ""),
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
            resolution_criteria=str(resolution_criteria),
            expiry=expiry,
            outcomes=outcomes,
            taker_fee=KALSHI_TAKER_FEE,
            maker_fee=KALSHI_MAKER_FEE,
            raw_data=raw,
        )

    def _normalise_order_book(self, raw: dict[str, Any]) -> OrderBook:
        def parse_levels(levels: list[list[Any]]) -> list[tuple[float, float]]:
            result = []
            for lvl in levels:
                if len(lvl) >= 2:
                    price = float(lvl[0]) / 100.0  # Kalshi uses cents
                    size = float(lvl[1])
                    result.append((price, size))
            return result

        bids = parse_levels(raw.get("yes", []))
        asks = [(1.0 - p, s) for p, s in parse_levels(raw.get("no", []))]

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            bids=sorted(bids, key=lambda x: x[0], reverse=True),
            asks=sorted(asks, key=lambda x: x[0]),
        )

    async def _ws_listener(self) -> None:
        """WebSocket listener for real-time Kalshi updates."""
        ws_url = (
            _KALSHI_WS if self._settings.kalshi_env == KalshiEnv.PROD else _KALSHI_DEMO_WS
        )
        backoff = 1.0
        while self._running:
            try:
                import websockets as ws_lib
                async with ws_lib.connect(
                    ws_url,
                    additional_headers=self._auth_headers("GET", "/trade-api/ws/v2"),
                    ping_interval=30,
                ) as ws:
                    backoff = 1.0
                    logger.info("Kalshi WS connected")
                    cmd = {
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["orderbook_delta", "ticker"]},
                    }
                    await ws.send(json.dumps(cmd))

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            self._process_ws_message(msg)
                        except Exception as exc:
                            logger.debug("Kalshi WS parse error: %s", exc)

            except Exception as exc:
                logger.warning("Kalshi WS disconnected: %s. Retry in %ss", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _process_ws_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type") or msg.get("msg")
        if msg_type not in ("orderbook_snapshot", "orderbook_delta", "ticker"):
            return

        ticker = msg.get("market_ticker") or msg.get("ticker")
        if not ticker or ticker not in self._markets:
            return

        market = self._markets[ticker]
        yes = market.yes_outcome
        no = market.no_outcome
        if not yes or not no:
            return

        if msg_type == "ticker":
            data = msg.get("msg", {})
            if "yes_bid" in data:
                yes.implied_prob_bid = float(data["yes_bid"]) / 100
            if "yes_ask" in data:
                yes.implied_prob_ask = float(data["yes_ask"]) / 100
            if "no_bid" in data:
                no.implied_prob_bid = float(data["no_bid"]) / 100
            if "no_ask" in data:
                no.implied_prob_ask = float(data["no_ask"]) / 100

        self._last_refresh[ticker] = time.monotonic()
