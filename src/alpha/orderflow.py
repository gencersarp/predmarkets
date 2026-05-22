"""
Order Flow Analysis — Detect Informed Trading and Momentum Signals.

Uses Polymarket's PUBLIC CLOB trades endpoint (no authentication required):
    GET https://clob.polymarket.com/trades?market=<token_id>&limit=500

Signals:
  1. ORDER_FLOW_MOMENTUM  — sustained net buy or sell imbalance over recent trades
  2. SHARP_MONEY          — single trades > $500 in a direction (informed trading)

Order Flow Imbalance (OFI):
    OFI = Σ(buy_volume) - Σ(sell_volume)  over recent N trades
    ofi_pct = OFI / total_volume   ∈ [-1, +1]

    ofi_pct > +0.65  → strong buy pressure → look for YES buys
    ofi_pct < -0.65  → strong sell pressure → look for NO buys (or avoid)

Why this generates alpha:
    Polymarket is a CLOB. Large informed traders (sharp money) take liquidity.
    After a sharp buy, prices adjust within minutes. Being second in is still
    profitable if you catch the early move. OFI persistence over 50+ trades
    indicates genuine information, not noise.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

from src.core.constants import (
    ORDER_FLOW_IMBALANCE_THRESHOLD,
    ORDER_FLOW_WINDOW_TRADES,
    SHARP_MONEY_TRADE_THRESHOLD_USD,
)
from src.core.models import (
    AlphaType,
    DirectionalSignal,
    Exchange,
    Market,
    RiskFlag,
    Side,
)

logger = logging.getLogger(__name__)

# Polymarket CLOB public API (no auth required)
_CLOB_BASE = "https://clob.polymarket.com"
# Polymarket data API (activity feed, also public)
_DATA_BASE = "https://data-api.polymarket.com"


class TradeRecord:
    """A single trade from the CLOB."""
    __slots__ = ("price", "size_usd", "side", "timestamp", "tx_hash")

    def __init__(
        self,
        price: float,
        size_usd: float,
        side: str,       # "BUY" or "SELL"
        timestamp: datetime,
        tx_hash: str = "",
    ) -> None:
        self.price = price
        self.size_usd = size_usd
        self.side = side
        self.timestamp = timestamp
        self.tx_hash = tx_hash


class OrderFlowAnalyzer:
    """
    Fetches and analyzes trade history for prediction market tokens.

    Usage:
        analyzer = OrderFlowAnalyzer()
        async with aiohttp.ClientSession() as session:
            signal = await analyzer.analyze_market(session, market)
    """

    def __init__(
        self,
        window_trades: int = ORDER_FLOW_WINDOW_TRADES,
        ofi_threshold: float = ORDER_FLOW_IMBALANCE_THRESHOLD,
        sharp_money_usd: float = SHARP_MONEY_TRADE_THRESHOLD_USD,
        min_edge_pct: float = 0.03,
    ) -> None:
        self._window = window_trades
        self._ofi_threshold = ofi_threshold
        self._sharp_threshold = sharp_money_usd
        self._min_edge = min_edge_pct

    async def analyze_market(
        self,
        session: aiohttp.ClientSession,
        market: Market,
        bankroll_usd: float = 1000.0,
    ) -> Optional[DirectionalSignal]:
        """
        Fetch recent trades and generate an order-flow signal if significant.
        Returns None if no actionable signal detected.
        """
        yes = market.yes_outcome
        if yes is None or yes.amm_token_address is None:
            return None

        token_id = yes.amm_token_address
        trades = await self._fetch_trades(session, token_id, limit=self._window)
        if len(trades) < 10:
            return None  # not enough data

        ofi_pct, sharp_trades = self._compute_ofi(trades)
        signal_side, confidence = self._classify_signal(ofi_pct, sharp_trades, trades)

        if signal_side is None:
            return None

        # Map signal direction to a tradeable price
        if signal_side == Side.YES:
            trade_price = yes.implied_prob_ask
        else:
            no = market.no_outcome
            if no is None:
                return None
            trade_price = no.implied_prob_ask

        if not (0.03 <= trade_price <= 0.97):
            return None

        # Implied edge: OFI suggests true prob is shifted from market price
        # Conservative estimate: |ofi_pct| / 3 as implied edge
        raw_edge = abs(ofi_pct) / 3.0

        if raw_edge < self._min_edge:
            return None

        # Adjust true_probability toward signal direction
        # For both sides: OFI suggests true prob is higher than market ask
        true_prob = min(0.97, max(0.03, trade_price + raw_edge))

        # Kelly sizing (conservative — OFI signals are short-term)
        from src.risk.kelly import kelly_fraction
        b = (1.0 / max(trade_price, 0.01)) - 1.0
        kf = kelly_fraction(p_win=true_prob, b=b, fraction=0.10)  # 10% Kelly for OFI
        size_usd = min(50.0, bankroll_usd * kf)  # hard cap $50 for OFI signals

        if size_usd < 5.0:
            return None

        ev_usd = size_usd * (true_prob * b - (1.0 - true_prob))
        if ev_usd <= 0:
            return None

        days = market.days_to_expiry or 30.0
        aroc = (ev_usd / size_usd) * (365.0 / max(days, 1.0))

        risk_flags: list[RiskFlag] = []
        if yes.volume_24h < 5000:
            risk_flags.append(RiskFlag.LOW_LIQUIDITY)

        # Flag if there are large trades AGAINST our direction
        opposing_sharp = [
            t for t in sharp_trades
            if (signal_side == Side.YES and t.side == "SELL")
            or (signal_side == Side.NO and t.side == "BUY")
        ]
        if len(opposing_sharp) >= 2:
            risk_flags.append(RiskFlag.SHARP_MONEY_AGAINST)

        return DirectionalSignal(
            alpha_type=AlphaType.ORDER_FLOW,
            market_id=market.market_id,
            exchange=market.exchange,
            side=signal_side,
            true_probability=true_prob,
            implied_probability=trade_price,
            edge=raw_edge,
            decimal_odds=1.0 / max(trade_price, 0.01),
            kelly_fraction_suggested=kf,
            recommended_size_usd=size_usd,
            expected_value_usd=ev_usd,
            expiry=market.expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            confidence=confidence,
            oracle_sources=[f"order_flow(ofi={ofi_pct:+.2f},n={len(trades)})"],
        )

    async def scan_universe(
        self,
        session: aiohttp.ClientSession,
        markets: list[Market],
        bankroll_usd: float = 1000.0,
    ) -> list[DirectionalSignal]:
        """Scan a list of markets for order-flow signals. Returns sorted by EV."""
        signals: list[DirectionalSignal] = []
        for market in markets:
            if not market.is_active:
                continue
            try:
                sig = await self.analyze_market(session, market, bankroll_usd)
                if sig and sig.is_actionable:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("OFI scan failed for %s: %s", market.market_id[:16], exc)
        logger.debug(
            "OFI scan: %d markets → %d actionable signals",
            len(markets), len(signals),
        )
        return sorted(signals, key=lambda s: s.expected_value_usd, reverse=True)

    # ---------------------------------------------------------------- internals

    async def _fetch_trades(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """
        Fetch recent trades from Polymarket CLOB public endpoint.
        Falls back to data API if CLOB is unavailable.
        """
        trades = await self._fetch_from_clob(session, token_id, limit)
        if not trades:
            trades = await self._fetch_from_data_api(session, token_id, limit)
        return trades

    async def _fetch_from_clob(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
        limit: int,
    ) -> list[TradeRecord]:
        try:
            url = f"{_CLOB_BASE}/trades"
            params = {"market": token_id, "limit": min(limit, 500)}
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                records = data if isinstance(data, list) else data.get("data", [])
                return [self._parse_clob_trade(r) for r in records if r]
        except Exception as exc:
            logger.debug("CLOB trades fetch failed for %s: %s", token_id[:16], exc)
            return []

    async def _fetch_from_data_api(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
        limit: int,
    ) -> list[TradeRecord]:
        """Polymarket data API — activity feed for a specific market."""
        try:
            url = f"{_DATA_BASE}/activity"
            params = {"market": token_id, "limit": min(limit, 200)}
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                records = data if isinstance(data, list) else data.get("data", [])
                return [self._parse_data_trade(r) for r in records if r]
        except Exception as exc:
            logger.debug("Data API trades fetch failed for %s: %s", token_id[:16], exc)
            return []

    @staticmethod
    def _parse_clob_trade(raw: dict) -> TradeRecord:
        # CLOB API format
        price = float(raw.get("price", 0))
        size = float(raw.get("size", raw.get("amount", 0)))
        side = str(raw.get("side", raw.get("outcome", "BUY"))).upper()
        if side not in ("BUY", "SELL"):
            side = "BUY"
        ts_raw = raw.get("timestamp", raw.get("created_at", ""))
        try:
            if isinstance(ts_raw, (int, float)):
                ts = datetime.utcfromtimestamp(float(ts_raw))
            else:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (ValueError, OSError):
            ts = datetime.now(timezone.utc)
        size_usd = size * price if price > 0 else size
        return TradeRecord(price=price, size_usd=size_usd, side=side, timestamp=ts,
                           tx_hash=str(raw.get("transactionHash", "")))

    @staticmethod
    def _parse_data_trade(raw: dict) -> TradeRecord:
        # Polymarket data API format
        price = float(raw.get("price", raw.get("outcomeIndex", 0.5)))
        usd_amount = float(raw.get("usdcSize", raw.get("amount", 0)))
        outcome = str(raw.get("outcome", "Buy")).lower()
        side = "BUY" if "buy" in outcome or "yes" in outcome else "SELL"
        ts_raw = raw.get("timestamp", raw.get("createdAt", 0))
        try:
            ts = datetime.utcfromtimestamp(float(ts_raw)) if ts_raw else datetime.now(timezone.utc)
        except (ValueError, OSError):
            ts = datetime.now(timezone.utc)
        return TradeRecord(price=price, size_usd=usd_amount, side=side, timestamp=ts)

    def _compute_ofi(
        self,
        trades: list[TradeRecord],
    ) -> tuple[float, list[TradeRecord]]:
        """
        Compute order flow imbalance and identify sharp money trades.

        Returns:
            ofi_pct: float in [-1, +1], positive = net buying
            sharp_trades: trades exceeding the sharp money threshold
        """
        buy_volume = sum(t.size_usd for t in trades if t.side == "BUY")
        sell_volume = sum(t.size_usd for t in trades if t.side == "SELL")
        total = buy_volume + sell_volume

        ofi_pct = 0.0
        if total > 0:
            ofi_pct = (buy_volume - sell_volume) / total

        sharp_trades = [t for t in trades if t.size_usd >= self._sharp_threshold]
        return ofi_pct, sharp_trades

    def _classify_signal(
        self,
        ofi_pct: float,
        sharp_trades: list[TradeRecord],
        all_trades: list[TradeRecord],
    ) -> tuple[Optional[Side], float]:
        """
        Determine signal direction and confidence from OFI and sharp money.

        Returns (side, confidence) or (None, 0.0) if no signal.
        """
        # Count sharp money direction
        sharp_buys = sum(1 for t in sharp_trades if t.side == "BUY")
        sharp_sells = sum(1 for t in sharp_trades if t.side == "SELL")
        sharp_net = sharp_buys - sharp_sells

        # Strong OFI alone
        if abs(ofi_pct) >= self._ofi_threshold:
            side = Side.YES if ofi_pct > 0 else Side.NO
            # Base confidence from OFI magnitude
            confidence = min(0.80, 0.50 + abs(ofi_pct) * 0.40)
            # Boost if sharp money agrees
            if sharp_net > 0 and side == Side.YES:
                confidence = min(0.88, confidence + 0.08)
            elif sharp_net < 0 and side == Side.NO:
                confidence = min(0.88, confidence + 0.08)
            return side, confidence

        # Weaker OFI but clear sharp money consensus (≥3 large trades one way)
        if abs(sharp_net) >= 3:
            side = Side.YES if sharp_net > 0 else Side.NO
            confidence = min(0.75, 0.55 + abs(sharp_net) * 0.05)
            return side, confidence

        return None, 0.0


# ---------------------------------------------------------------------------
# Convenience: compute OFI from a list of trade dicts (for testing/analysis)
# ---------------------------------------------------------------------------

def compute_ofi_from_records(
    trades: list[dict],
    price_key: str = "price",
    size_key: str = "size",
    side_key: str = "side",
) -> float:
    """
    Standalone OFI calculator for backtesting / analysis.
    Returns ofi_pct in [-1, +1].
    """
    buy_vol = sell_vol = 0.0
    for t in trades:
        price = float(t.get(price_key, 0))
        size = float(t.get(size_key, 0))
        usd = size * price if price > 0 else size
        side = str(t.get(side_key, "buy")).upper()
        if side == "BUY":
            buy_vol += usd
        else:
            sell_vol += usd
    total = buy_vol + sell_vol
    return (buy_vol - sell_vol) / total if total > 0 else 0.0
