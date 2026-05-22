"""
Terminal State Manager — the single source of truth.

Aggregates data from all feeds into a coherent world state.
All strategy modules READ from this state; they never call feeds directly.
Thread/coroutine-safe via asyncio primitives.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.core.models import (
    Exchange,
    Market,
    NewsEvent,
    OracleEstimate,
    PortfolioSnapshot,
    Position,
)

logger = logging.getLogger(__name__)


class TerminalState:
    """
    Central in-memory state store.

    Conventions:
    - market_id is globally unique: f"{exchange}:{raw_id}"
    - All prices are in [0,1] probability space (i.e. fraction of $1 payout)
    - Thread safety: use async methods; internal dicts are protected by asyncio locks
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # Market state
        self._markets: dict[str, Market] = {}             # market_id → Market
        self._market_last_update: dict[str, float] = {}   # market_id → monotonic

        # Oracle estimates: market_id → list of estimates from various sources
        self._oracle_estimates: dict[str, list[OracleEstimate]] = {}

        # News
        self._recent_news: list[NewsEvent] = []

        # Portfolio
        self._positions: dict[str, Position] = {}         # position_id → Position
        self._portfolio: Optional[PortfolioSnapshot] = None

        # Metrics
        self._tick_count: int = 0
        self._started_at: datetime = datetime.now(timezone.utc)

    # ---------------------------------------------------------------- Markets

    async def upsert_markets(self, markets: list[Market]) -> None:
        async with self._lock:
            for m in markets:
                key = self._market_key(m.exchange, m.market_id)
                self._markets[key] = m
                self._market_last_update[key] = time.monotonic()
        logger.debug("State: upserted %d markets", len(markets))

    async def get_market(self, exchange: Exchange, raw_id: str) -> Optional[Market]:
        key = self._market_key(exchange, raw_id)
        return self._markets.get(key)

    async def get_all_markets(
        self,
        exchange: Optional[Exchange] = None,
        max_age_sec: float = 60.0,
    ) -> list[Market]:
        now = time.monotonic()
        async with self._lock:
            result = []
            for key, market in self._markets.items():
                if exchange and market.exchange != exchange:
                    continue
                age = now - self._market_last_update.get(key, 0)
                if age <= max_age_sec:
                    result.append(market)
        return result

    async def get_markets_by_category(self, category: str) -> list[Market]:
        async with self._lock:
            return [
                m for m in self._markets.values()
                if category.lower() in m.category.lower()
            ]

    async def get_market_pair(
        self, title_fragment: str
    ) -> list[Market]:
        """Find markets across exchanges with similar titles (for arb detection)."""
        frag_lower = title_fragment.lower()
        async with self._lock:
            return [
                m for m in self._markets.values()
                if frag_lower in m.title.lower()
            ]

    # ---------------------------------------------------------------- Oracles

    async def upsert_oracle_estimate(
        self, market_id: str, estimate: OracleEstimate
    ) -> None:
        async with self._lock:
            if market_id not in self._oracle_estimates:
                self._oracle_estimates[market_id] = []
            # Replace estimate from same source
            estimates = self._oracle_estimates[market_id]
            estimates[:] = [e for e in estimates if e.source != estimate.source]
            estimates.append(estimate)

    async def get_oracle_estimates(self, market_id: str) -> list[OracleEstimate]:
        return self._oracle_estimates.get(market_id, [])

    async def get_consensus_probability(self, market_id: str) -> Optional[float]:
        """
        Compute a consensus true probability from all oracle sources
        via inverse-variance weighting (simplified: equal weights here).
        """
        estimates = self._oracle_estimates.get(market_id, [])
        if not estimates:
            return None
        # Weighted average; weight by confidence interval width (inverse)
        probs = []
        weights = []
        for est in estimates:
            ci_width = est.confidence_interval_high - est.confidence_interval_low
            weight = 1.0 / max(ci_width, 0.01)
            probs.append(est.true_probability * weight)
            weights.append(weight)
        return sum(probs) / sum(weights)

    # ---------------------------------------------------------------- News

    async def upsert_news(self, events: list[NewsEvent]) -> None:
        async with self._lock:
            self._recent_news = events[:100]  # keep last 100

    async def get_recent_news(self, limit: int = 20) -> list[NewsEvent]:
        return self._recent_news[:limit]

    # ---------------------------------------------------------------- Portfolio

    async def upsert_position(self, position: Position) -> None:
        async with self._lock:
            self._positions[position.position_id] = position

    async def remove_position(self, position_id: str) -> None:
        async with self._lock:
            self._positions.pop(position_id, None)

    async def get_open_positions(self) -> list[Position]:
        from src.core.models import PositionStatus
        return [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.LOCKED)
        ]

    async def get_all_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def set_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self._portfolio = snapshot

    async def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        return self._portfolio

    # ---------------------------------------------------------------- Ticker

    async def tick(self) -> None:
        """Called once per main loop iteration — tracks heartbeat."""
        self._tick_count += 1
        if self._tick_count % 100 == 0:
            markets = len(self._markets)
            positions = len(self._positions)
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
            logger.info(
                "State tick %d | markets=%d positions=%d uptime=%.0fs",
                self._tick_count, markets, positions, uptime,
            )

    # ---------------------------------------------------------------- Helpers

    @staticmethod
    def _market_key(exchange: Exchange, raw_id: str) -> str:
        return f"{exchange.value}:{raw_id}"

    async def summary(self) -> dict:
        """Return a concise summary dict for the terminal dashboard."""
        async with self._lock:
            open_pos = [
                p for p in self._positions.values()
                if p.status.value in ("open", "locked")
            ]
            return {
                "total_markets": len(self._markets),
                "polymarket_markets": sum(
                    1 for m in self._markets.values()
                    if m.exchange == Exchange.POLYMARKET
                ),
                "kalshi_markets": sum(
                    1 for m in self._markets.values()
                    if m.exchange == Exchange.KALSHI
                ),
                "open_positions": len(open_pos),
                "oracle_covered_markets": len(self._oracle_estimates),
                "recent_news_items": len(self._recent_news),
                "tick_count": self._tick_count,
                "uptime_sec": (datetime.now(timezone.utc) - self._started_at).total_seconds(),
            }
