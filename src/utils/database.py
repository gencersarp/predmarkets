"""
Async database layer for paper-trade persistence and audit logging.

Uses SQLAlchemy (async) + aiosqlite for local SQLite.
Swap the connection string for TimescaleDB/PostgreSQL in production.

Schema:
    paper_orders  — every simulated order (immutable audit trail)
    paper_trades  — filled paper trades (linked to orders)
    arb_log       — logged arbitrage opportunities (actionable or not)
    signal_log    — logged directional signals
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.models import ArbitrageOpportunity, DirectionalSignal, Order

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PaperOrderRow(Base):
    __tablename__ = "paper_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    exchange = Column(String(32))
    market_id = Column(String(256))
    side = Column(String(8))
    order_side = Column(String(8))
    order_type = Column(String(16))
    price = Column(Float)
    size_usd = Column(Float)
    status = Column(String(32))
    filled_size_usd = Column(Float, default=0.0)
    avg_fill_price = Column(Float, nullable=True)
    alpha_type = Column(String(64), nullable=True)
    signal_id = Column(String(64), nullable=True)
    metadata_json = Column(Text, default="{}")


class ArbLogRow(Base):
    __tablename__ = "arb_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opp_id = Column(String(64), unique=True, index=True)
    detected_at = Column(DateTime)
    alpha_type = Column(String(64))
    market_ids = Column(Text)   # JSON list
    exchanges = Column(Text)    # JSON list
    net_edge_pct = Column(Float)
    net_edge_usd = Column(Float)
    required_capital_usd = Column(Float)
    aroc_annual = Column(Float)
    is_actionable = Column(Boolean)
    risk_flags = Column(Text)   # JSON list
    legs_json = Column(Text)    # JSON


class SignalLogRow(Base):
    __tablename__ = "signal_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(64), unique=True, index=True)
    detected_at = Column(DateTime)
    alpha_type = Column(String(64))
    market_id = Column(String(256))
    exchange = Column(String(32))
    side = Column(String(8))
    true_probability = Column(Float)
    implied_probability = Column(Float)
    edge = Column(Float)
    expected_value_usd = Column(Float)
    recommended_size_usd = Column(Float)
    aroc_annual = Column(Float)
    confidence = Column(Float)
    is_actionable = Column(Boolean)
    risk_flags = Column(Text)   # JSON list
    oracle_sources = Column(Text)  # JSON list


class Database:
    """Async database manager. Call `init()` once at startup."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialised at %s", self._url)

    async def close(self) -> None:
        await self._engine.dispose()

    # ---------------------------------------------------------------- Orders

    async def log_paper_order(self, order: Order, alpha_type: str = "") -> None:
        async with self._session_factory() as session:
            row = PaperOrderRow(
                order_id=order.order_id,
                created_at=order.created_at,
                exchange=order.exchange.value,
                market_id=order.market_id,
                side=order.side.value,
                order_side=order.order_side.value,
                order_type=order.order_type.value,
                price=order.price,
                size_usd=order.size_usd,
                status=order.status.value,
                filled_size_usd=order.filled_size_usd,
                avg_fill_price=order.avg_fill_price,
                alpha_type=alpha_type,
                signal_id=order.metadata.get("signal_id"),
                metadata_json=json.dumps(order.metadata),
            )
            session.add(row)
            await session.commit()

    async def get_paper_orders(
        self,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            q = select(PaperOrderRow).order_by(PaperOrderRow.created_at.desc()).limit(limit)
            if status:
                q = q.where(PaperOrderRow.status == status)
            result = await session.execute(q)
            rows = result.scalars().all()
            return [
                {
                    "order_id": r.order_id,
                    "created_at": r.created_at,
                    "exchange": r.exchange,
                    "market_id": r.market_id,
                    "side": r.side,
                    "price": r.price,
                    "size_usd": r.size_usd,
                    "status": r.status,
                    "filled_size_usd": r.filled_size_usd,
                    "alpha_type": r.alpha_type,
                }
                for r in rows
            ]

    # ---------------------------------------------------------------- Arb Log

    async def log_arb_opportunity(self, opp: ArbitrageOpportunity) -> None:
        async with self._session_factory() as session:
            row = ArbLogRow(
                opp_id=opp.opp_id,
                detected_at=opp.detected_at,
                alpha_type=opp.alpha_type.value,
                market_ids=json.dumps(opp.market_ids),
                exchanges=json.dumps([e.value for e in opp.exchanges]),
                net_edge_pct=opp.net_edge_pct,
                net_edge_usd=opp.net_edge_usd,
                required_capital_usd=opp.required_capital_usd,
                aroc_annual=opp.aroc_annual,
                is_actionable=opp.is_actionable,
                risk_flags=json.dumps([f.value for f in opp.risk_flags]),
                legs_json=json.dumps(opp.legs),
            )
            session.add(row)
            await session.commit()

    # ---------------------------------------------------------------- Signal Log

    async def log_signal(self, signal: DirectionalSignal) -> None:
        async with self._session_factory() as session:
            row = SignalLogRow(
                signal_id=signal.signal_id,
                detected_at=signal.detected_at,
                alpha_type=signal.alpha_type.value,
                market_id=signal.market_id,
                exchange=signal.exchange.value,
                side=signal.side.value,
                true_probability=signal.true_probability,
                implied_probability=signal.implied_probability,
                edge=signal.edge,
                expected_value_usd=signal.expected_value_usd,
                recommended_size_usd=signal.recommended_size_usd,
                aroc_annual=signal.aroc_annual,
                confidence=signal.confidence,
                is_actionable=signal.is_actionable,
                risk_flags=json.dumps([f.value for f in signal.risk_flags]),
                oracle_sources=json.dumps(signal.oracle_sources),
            )
            session.add(row)
            await session.commit()

    async def get_recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            q = select(SignalLogRow).order_by(
                SignalLogRow.detected_at.desc()
            ).limit(limit)
            result = await session.execute(q)
            rows = result.scalars().all()
            return [
                {
                    "signal_id": r.signal_id[:8],
                    "alpha_type": r.alpha_type,
                    "market_id": r.market_id[:40],
                    "exchange": r.exchange,
                    "side": r.side,
                    "edge": r.edge,
                    "ev_usd": r.expected_value_usd,
                    "size_usd": r.recommended_size_usd,
                    "aroc": r.aroc_annual,
                    "actionable": r.is_actionable,
                }
                for r in rows
            ]
