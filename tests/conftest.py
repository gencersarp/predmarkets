"""
Shared pytest fixtures for the PMT test suite.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.core.models import (
    AlphaType,
    ArbitrageOpportunity,
    DirectionalSignal,
    Exchange,
    Market,
    MarketOutcome,
    OrderBook,
    Position,
    PositionStatus,
    ResolutionSource,
    RiskFlag,
    Side,
)

# ---- Force paper mode for all tests ----------------------------------------
os.environ.setdefault("PMT_MODE", "paper")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./pmt_test.db")


def _make_order_book(bid: float, ask: float) -> OrderBook:
    return OrderBook(
        timestamp=datetime.now(timezone.utc),
        bids=[(bid, 1000.0)],
        asks=[(ask, 1000.0)],
    )


def make_market(
    market_id: str = "test-001",
    exchange: Exchange = Exchange.POLYMARKET,
    title: str = "Will X happen?",
    yes_bid: float = 0.48,
    yes_ask: float = 0.52,
    no_bid: float = 0.47,
    no_ask: float = 0.50,
    taker_fee: float = 0.02,
    maker_fee: float = 0.00,
    expiry_days: float = 30.0,
    volume_24h: float = 50000.0,
    resolution_source: ResolutionSource = ResolutionSource.UMA_ORACLE,
    resolution_criteria: str = "UMA oracle determines outcome.",
    category: str = "general",
    amm_reserve_yes: Optional[float] = None,
    amm_reserve_no: Optional[float] = None,
) -> Market:
    expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
    outcomes = [
        MarketOutcome(
            outcome_id=f"{market_id}_yes",
            side=Side.YES,
            implied_prob_bid=yes_bid,
            implied_prob_ask=yes_ask,
            order_book=_make_order_book(yes_bid, yes_ask),
            volume_24h=volume_24h,
            amm_reserve_yes=amm_reserve_yes,
            amm_reserve_no=amm_reserve_no,
        ),
        MarketOutcome(
            outcome_id=f"{market_id}_no",
            side=Side.NO,
            implied_prob_bid=no_bid,
            implied_prob_ask=no_ask,
            order_book=_make_order_book(no_bid, no_ask),
        ),
    ]
    return Market(
        market_id=market_id,
        exchange=exchange,
        title=title,
        description=f"Test market {market_id}",
        category=category,
        resolution_source=resolution_source,
        resolution_criteria=resolution_criteria,
        expiry=expiry,
        outcomes=outcomes,
        taker_fee=taker_fee,
        maker_fee=maker_fee,
        fetched_at=datetime.now(timezone.utc),
        is_active=True,
    )


def make_arb_opportunity(
    net_edge_pct: float = 0.03,
    net_edge_usd: float = 15.0,
    required_capital: float = 100.0,
    aroc: float = 0.80,
    risk_flags: Optional[list[RiskFlag]] = None,
    fee_cost_usd: float = 2.0,
    gas_cost_usd: float = 0.0,
) -> ArbitrageOpportunity:
    gross_edge_usd = net_edge_usd + fee_cost_usd + gas_cost_usd + 2.0
    return ArbitrageOpportunity(
        alpha_type=AlphaType.CROSS_EXCHANGE_ARB,
        market_ids=["pm-001", "kal-001"],
        exchanges=[Exchange.POLYMARKET, Exchange.KALSHI],
        gross_edge_pct=net_edge_pct + 0.01,
        net_edge_pct=net_edge_pct,
        gross_edge_usd=gross_edge_usd,
        net_edge_usd=net_edge_usd,
        required_capital_usd=required_capital,
        fee_cost_usd=fee_cost_usd,
        gas_cost_usd=gas_cost_usd,
        aroc_annual=aroc,
        risk_flags=risk_flags or [],
        confidence=0.90,
        expiry=datetime.now(timezone.utc) + timedelta(days=14),
        legs=[
            {"action": "buy", "side": "yes", "price": 0.45, "size_usd": 50.0,
             "market_id": "pm-001", "exchange": "polymarket"},
            {"action": "buy", "side": "no", "price": 0.52, "size_usd": 50.0,
             "market_id": "kal-001", "exchange": "kalshi"},
        ],
    )


def make_directional_signal(
    edge: float = 0.08,
    ev_usd: float = 25.0,
    size_usd: float = 100.0,
    aroc: float = 1.20,
    risk_flags: Optional[list[RiskFlag]] = None,
) -> DirectionalSignal:
    return DirectionalSignal(
        alpha_type=AlphaType.EV_DIRECTIONAL,
        market_id="test-001",
        exchange=Exchange.POLYMARKET,
        side=Side.YES,
        true_probability=0.60,
        implied_probability=0.52,
        edge=edge,
        decimal_odds=1.0 / 0.52,
        kelly_fraction_suggested=0.08,
        recommended_size_usd=size_usd,
        expected_value_usd=ev_usd,
        expiry=datetime.now(timezone.utc) + timedelta(days=14),
        aroc_annual=aroc,
        risk_flags=risk_flags or [],
        confidence=0.75,
        oracle_sources=["test_model"],
    )


def make_position(
    market_id: str = "test-001",
    size_usd: float = 100.0,
    entry_price: float = 0.52,
    current_price: float = 0.55,
    expiry_days: float = 14.0,
    status: PositionStatus = PositionStatus.OPEN,
) -> Position:
    return Position(
        exchange=Exchange.POLYMARKET,
        market_id=market_id,
        market_title=f"Test market {market_id}",
        side=Side.YES,
        size_usd=size_usd,
        entry_price=entry_price,
        current_price=current_price,
        expiry=datetime.now(timezone.utc) + timedelta(days=expiry_days),
        status=status,
        is_paper=True,
    )


# Fixtures
@pytest.fixture
def market() -> Market:
    return make_market()


@pytest.fixture
def cheap_market() -> Market:
    """Market where ask_yes + ask_no < 1 → intra-market arb."""
    return make_market(yes_ask=0.47, no_ask=0.48)


@pytest.fixture
def fair_market() -> Market:
    """Market with no arb (fair prices)."""
    return make_market(yes_ask=0.52, no_ask=0.49)


@pytest.fixture
def arb_opportunity() -> ArbitrageOpportunity:
    return make_arb_opportunity()


@pytest.fixture
def directional_signal() -> DirectionalSignal:
    return make_directional_signal()


@pytest.fixture
def position() -> Position:
    return make_position()
