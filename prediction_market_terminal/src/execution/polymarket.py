"""
Polymarket Live Execution Adapter.

Uses Polymarket CLOB API with EIP-712 signed orders.
Private key is NEVER stored here — injected via settings/AWS Secrets Manager.

CLOB Order Flow:
  1. Build order params (tokenId, makerAmount, takerAmount, side, expiration)
  2. Sign with EIP-712 using eth_account (comes with web3.py)
  3. POST to /order endpoint with signature + owner address
  4. Poll /order/{id} for fill status

Contract addresses (Polygon mainnet):
  CTF Exchange: 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E
  Neg Risk CTF Exchange: 0xC5d563A36AE78145C45a50134d48A1215220f80a
  USDC (PoS): 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174

Token decimals:
  USDC: 6 decimals
  CTF outcome tokens: 6 decimals

Authentication:
  Polymarket uses two-layer auth:
  1. L1: EIP-712 signed nonce to derive L2 API credentials
  2. L2: HMAC / JWT-style headers using the derived key pair

For full details: https://docs.polymarket.com/#authentication
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp

from src.core.exceptions import (
    ExecutionError,
    GasLimitExceeded,
    OrderRejected,
    PaperModeViolation,
)
from src.core.models import (
    Exchange,
    Market,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from src.execution.base import ExchangeAdapter
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---- Polygon mainnet contract addresses ------------------------------------
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
CHAIN_ID_POLYGON = 137

# EIP-712 Order struct types for Polymarket CTF Exchange
_ORDER_EIP712_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "salt", "type": "uint256"},
        {"name": "maker", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "taker", "type": "address"},
        {"name": "tokenId", "type": "uint256"},
        {"name": "makerAmount", "type": "uint256"},
        {"name": "takerAmount", "type": "uint256"},
        {"name": "expiration", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "feeRateBps", "type": "uint256"},
        {"name": "side", "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ],
}

# ClobAuth types for L1 → L2 credential derivation
_CLOBAUTH_EIP712_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
    ],
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "int256"},
        {"name": "message", "type": "string"},
    ],
}


class PolymarketAdapter(ExchangeAdapter):
    """
    Live Polymarket execution adapter via CLOB API.

    Authentication: EIP-712 signed orders using private key.
    The private key is accessed from settings (injected via env or AWS Secrets).

    SAFETY: In PAPER mode, raises PaperModeViolation for any order placement.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._clob = self._settings.polymarket_clob_endpoint

    @property
    def name(self) -> str:
        return "polymarket_live"

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "PMT/1.0"},
            )

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

        private_key = self._settings.polymarket_private_key
        if not private_key:
            raise ExecutionError(
                "POLYMARKET_PRIVATE_KEY not configured. "
                "Set in .env or AWS Secrets Manager."
            )

        pk_value = private_key.get_secret_value()
        wallet_address = self._derive_address(pk_value)

        # Build order parameters
        clob_side = 0 if order_side == OrderSide.BUY else 1
        clob_type = self._map_order_type(order_type)

        # USDC has 6 decimal places; CTF tokens have 6 decimals
        usdc_decimals = 6
        token_decimals = 6

        # makerAmount = USDC you spend; takerAmount = tokens you receive
        if order_side == OrderSide.BUY:
            maker_amount = int(size_usd * 10 ** usdc_decimals)
            # shares = size_usd / price
            shares = size_usd / max(price, 0.001)
            taker_amount = int(shares * 10 ** token_decimals)
        else:
            # Selling: makerAmount = tokens given, takerAmount = USDC received
            shares = size_usd / max(price, 0.001)
            maker_amount = int(shares * 10 ** token_decimals)
            taker_amount = int(size_usd * 10 ** usdc_decimals)

        salt = random.randint(1, 2 ** 256 - 1)

        order_message = {
            "salt": salt,
            "maker": wallet_address,
            "signer": wallet_address,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": int(market_id) if market_id.isdigit() else self._hash_market_id(market_id),
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "expiration": 0,    # 0 = GTC / no expiry
            "nonce": 0,
            "feeRateBps": 0,
            "side": clob_side,
            "signatureType": 0,  # EOA = 0
        }

        # Sign with EIP-712
        signature = self._sign_order(pk_value, order_message)

        # Build API payload
        payload = {
            "order": {**order_message, "salt": str(salt)},
            "signature": signature,
            "owner": wallet_address,
            "orderType": clob_type,
        }

        await self._ensure_session()
        assert self._session is not None

        async with self._session.post(
            f"{self._clob}/order",
            json=payload,
            headers=self._l2_auth_headers("POST", "/order", payload),
        ) as resp:
            body = await resp.json()
            if resp.status not in (200, 201):
                raise OrderRejected(
                    f"CLOB order rejected ({resp.status}): {body}",
                    order_id=str(salt),
                )

        exchange_order_id = body.get("orderID", body.get("id", str(salt)))
        order = Order(
            order_id=str(uuid.uuid4()),
            exchange=Exchange.POLYMARKET,
            market_id=market_id,
            side=side,
            order_side=order_side,
            order_type=order_type,
            price=price,
            size_usd=size_usd,
            status=OrderStatus.OPEN,
            is_paper=False,
            exchange_order_id=exchange_order_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        logger.info(
            "POLY LIVE %s %s $%.2f @ %.3f → %s (exchange_id=%s)",
            order_side.value.upper(), side.value.upper(),
            size_usd, price, order.status.value, exchange_order_id[:16],
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        await self._ensure_session()
        assert self._session is not None
        async with self._session.delete(
            f"{self._clob}/order/{order_id}",
            headers=self._l2_auth_headers("DELETE", f"/order/{order_id}", {}),
        ) as resp:
            return resp.status == 200

    async def get_order_status(self, order_id: str) -> Order:
        await self._ensure_session()
        assert self._session is not None
        async with self._session.get(
            f"{self._clob}/order/{order_id}",
            headers=self._l2_auth_headers("GET", f"/order/{order_id}", {}),
        ) as resp:
            if resp.status != 200:
                raise OrderRejected("Order not found", order_id)
            data = await resp.json()
            return self._normalise_order(data)

    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        await self._ensure_session()
        assert self._session is not None
        params: dict[str, str] = {"status": "live"}
        if market_id:
            params["market"] = market_id
        async with self._session.get(
            f"{self._clob}/orders",
            params=params,
            headers=self._l2_auth_headers("GET", "/orders", {}),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [self._normalise_order(o) for o in data.get("data", [])]

    async def get_positions(self) -> list[Position]:
        """
        Fetch open positions from Polymarket data API.
        Positions are on-chain CTF token balances held by the wallet.
        """
        wallet = self._get_wallet_address()
        if not wallet:
            return []
        await self._ensure_session()
        assert self._session is not None
        try:
            async with self._session.get(
                f"https://data-api.polymarket.com/positions",
                params={"user": wallet},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                positions = []
                for raw in data if isinstance(data, list) else data.get("data", []):
                    try:
                        positions.append(self._normalise_position(raw))
                    except Exception:
                        pass
                return positions
        except Exception as exc:
            logger.warning("Failed to fetch Polymarket positions: %s", exc)
            return []

    async def get_balance_usd(self) -> float:
        """
        Fetch USDC balance from Polymarket data API.
        """
        wallet = self._get_wallet_address()
        if not wallet:
            return 0.0
        await self._ensure_session()
        assert self._session is not None
        try:
            async with self._session.get(
                f"https://data-api.polymarket.com/value",
                params={"user": wallet},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
                # Returns portfolioValue in USDC cents or USD depending on version
                val = data.get("portfolioValue", data.get("value", 0.0))
                return float(val)
        except Exception as exc:
            logger.warning("Failed to fetch Polymarket balance: %s", exc)
            return 0.0

    # ---------------------------------------------------------------- Auth / Signing

    def _sign_order(self, private_key: str, order_message: dict) -> str:
        """
        Sign a Polymarket order using EIP-712 structured data.
        Uses eth_account from web3.py.
        """
        try:
            from eth_account import Account
            from eth_account.messages import encode_typed_data_hash

            structured_data = {
                "types": _ORDER_EIP712_TYPES,
                "domain": {
                    "name": "CTFExchange",
                    "version": "1",
                    "chainId": CHAIN_ID_POLYGON,
                    "verifyingContract": CTF_EXCHANGE,
                },
                "primaryType": "Order",
                "message": order_message,
            }

            # eth_account sign_typed_data (EIP-712)
            account = Account.from_key(private_key)
            signed = account.sign_typed_data(
                domain_data=structured_data["domain"],
                message_types={"Order": _ORDER_EIP712_TYPES["Order"]},
                message_data=order_message,
            )
            return signed.signature.hex()

        except ImportError:
            raise ExecutionError(
                "eth_account not available. Install: pip install web3"
            )
        except Exception as exc:
            raise ExecutionError(f"EIP-712 signing failed: {exc}") from exc

    def _l2_auth_headers(
        self,
        method: str,
        path: str,
        body: dict,
    ) -> dict[str, str]:
        """
        Generate Polymarket L2 authentication headers.
        L2 credentials are derived from the L1 wallet signature.
        Uses api_key / api_secret from settings.
        """
        api_key = self._settings.polymarket_api_key
        api_secret = self._settings.polymarket_api_secret
        passphrase = self._settings.polymarket_api_passphrase

        if not api_key or not api_secret:
            return {"User-Agent": "PMT/1.0"}

        ts = str(int(time.time()))
        body_str = json.dumps(body, separators=(",", ":")) if body else ""

        # Polymarket L2 auth: HMAC-SHA256 of timestamp + method + path + body
        message = ts + method.upper() + path + body_str
        secret_val = api_secret.get_secret_value()
        sig = hmac.new(
            secret_val.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "POLY_ADDRESS": self._get_wallet_address() or "",
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": ts,
            "POLY_PASSPHRASE": passphrase.get_secret_value() if passphrase else "",
            "Content-Type": "application/json",
        }
        if api_key:
            headers["POLY_API_KEY"] = api_key.get_secret_value()
        return headers

    def _derive_address(self, private_key: str) -> str:
        """Derive Ethereum address from private key."""
        try:
            from eth_account import Account
            account = Account.from_key(private_key)
            return account.address
        except Exception:
            return ""

    def _get_wallet_address(self) -> str:
        pk = self._settings.polymarket_private_key
        if not pk:
            return ""
        return self._derive_address(pk.get_secret_value())

    @staticmethod
    def _hash_market_id(market_id: str) -> int:
        """Convert a string market_id to a uint256 token ID."""
        return int(hashlib.sha256(market_id.encode()).hexdigest(), 16) % (2 ** 256)

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        return {
            OrderType.IOC: "IOC",
            OrderType.FOK: "FOK",
            OrderType.GTC: "GTC",
            OrderType.LIMIT: "GTC",
            OrderType.MARKET: "FOK",
        }.get(order_type, "GTC")

    def _normalise_order(self, raw: dict) -> Order:
        status_map = {
            "live": OrderStatus.OPEN,
            "matched": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "partial": OrderStatus.PARTIAL,
        }
        return Order(
            order_id=str(uuid.uuid4()),
            exchange=Exchange.POLYMARKET,
            market_id=raw.get("market", raw.get("asset_id", "")),
            side=Side.YES if str(raw.get("side", "")).upper() == "BUY" else Side.NO,
            order_side=OrderSide.BUY if str(raw.get("side", "")).upper() == "BUY" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=float(raw.get("price", 0)) / 100.0 if float(raw.get("price", 0)) > 1 else float(raw.get("price", 0)),
            size_usd=float(raw.get("original_size", raw.get("size", 0))),
            status=status_map.get(raw.get("status", ""), OrderStatus.PENDING),
            filled_size_usd=float(raw.get("size_matched", raw.get("filled_size", 0))),
            exchange_order_id=raw.get("id", raw.get("orderID", "")),
            is_paper=False,
        )

    def _normalise_position(self, raw: dict) -> Position:
        """Convert Polymarket data API position to canonical Position."""
        market_id = str(raw.get("conditionId", raw.get("market", "")))
        size_usd = float(raw.get("size", raw.get("currentValue", 0)))
        entry = float(raw.get("avgPrice", raw.get("initialValue", 0.5)))
        current = float(raw.get("curPrice", entry))
        side_raw = str(raw.get("outcome", "Yes")).lower()
        side = Side.YES if "yes" in side_raw else Side.NO

        return Position(
            exchange=Exchange.POLYMARKET,
            market_id=market_id,
            market_title=raw.get("title", raw.get("question", f"Market {market_id[:8]}")),
            side=side,
            size_usd=size_usd,
            entry_price=entry,
            current_price=current,
            unrealised_pnl=size_usd * (current - entry),
            is_paper=False,
        )
