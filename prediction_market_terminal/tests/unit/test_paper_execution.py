"""Unit tests for paper trading execution adapter."""
from __future__ import annotations

import pytest

from src.core.exceptions import PaperModeViolation
from src.core.models import Exchange, OrderSide, OrderStatus, OrderType, Side
from src.execution.paper import PaperExchangeAdapter
from tests.conftest import make_market


@pytest.fixture
def paper_adapter():
    return PaperExchangeAdapter(
        exchange=Exchange.POLYMARKET,
        initial_balance_usd=1000.0,
    )


class TestPaperOrderPlacement:
    async def test_ioc_buy_fills_at_ask(self, paper_adapter):
        market = make_market(yes_ask=0.55)
        order = await paper_adapter.place_order(
            market_id="test-001",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.55,
            size_usd=100.0,
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_size_usd == 100.0
        assert order.avg_fill_price == pytest.approx(0.55)

    async def test_buy_deducts_balance(self, paper_adapter):
        market = make_market(yes_ask=0.50)
        initial = await paper_adapter.get_balance_usd()
        await paper_adapter.place_order(
            market_id="test-001",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.50,
            size_usd=200.0,
            is_paper=True,
            market=market,
        )
        after = await paper_adapter.get_balance_usd()
        assert after == initial - 200.0

    async def test_live_order_raises_paper_violation(self, paper_adapter):
        with pytest.raises(PaperModeViolation):
            await paper_adapter.place_order(
                market_id="test",
                side=Side.YES,
                order_side=OrderSide.BUY,
                order_type=OrderType.IOC,
                price=0.50,
                size_usd=100.0,
                is_paper=False,  # <-- live flag
            )

    async def test_insufficient_balance_rejects_order(self, paper_adapter):
        market = make_market(yes_ask=0.50)
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.50,
            size_usd=9999.0,  # way over balance
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.REJECTED

    async def test_gtc_order_stays_open_when_price_not_met(self, paper_adapter):
        # Market ask=0.60, limit price=0.40 → GTC not immediately fillable
        market = make_market(yes_ask=0.60)
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.GTC,
            price=0.40,  # below ask
            size_usd=100.0,
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.OPEN

    async def test_ioc_cancelled_when_price_not_met(self, paper_adapter):
        # Market ask=0.70, IOC at 0.40 → cancelled
        market = make_market(yes_ask=0.70)
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.40,  # can't fill
            size_usd=100.0,
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.CANCELLED

    async def test_cancel_open_order(self, paper_adapter):
        market = make_market(yes_ask=0.70)
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.GTC,
            price=0.40,
            size_usd=100.0,
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.OPEN
        cancelled = await paper_adapter.cancel_order(order.order_id)
        assert cancelled
        updated = await paper_adapter.get_order_status(order.order_id)
        assert updated.status == OrderStatus.CANCELLED

    async def test_cancel_filled_order_returns_false(self, paper_adapter):
        market = make_market(yes_ask=0.50)
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.55,
            size_usd=50.0,
            is_paper=True,
            market=market,
        )
        assert order.status == OrderStatus.FILLED
        cancelled = await paper_adapter.cancel_order(order.order_id)
        assert not cancelled

    async def test_get_open_orders(self, paper_adapter):
        market = make_market(yes_ask=0.80)
        await paper_adapter.place_order(
            "test", Side.YES, OrderSide.BUY, OrderType.GTC,
            0.40, 50.0, is_paper=True, market=market,
        )
        await paper_adapter.place_order(
            "test", Side.YES, OrderSide.BUY, OrderType.GTC,
            0.42, 50.0, is_paper=True, market=market,
        )
        open_orders = await paper_adapter.get_open_orders("test")
        assert len(open_orders) == 2

    async def test_no_position_initially(self, paper_adapter):
        positions = await paper_adapter.get_positions()
        assert positions == []

    async def test_portfolio_summary(self, paper_adapter):
        market = make_market(yes_ask=0.50)
        await paper_adapter.place_order(
            "test", Side.YES, OrderSide.BUY, OrderType.IOC,
            0.55, 100.0, is_paper=True, market=market,
        )
        summary = paper_adapter.portfolio_summary()
        assert summary["filled_orders"] == 1
        assert summary["total_traded_usd"] == 100.0
        assert summary["balance_usd"] == 900.0

    async def test_no_market_data_fills_at_stated_price(self, paper_adapter):
        """Without market data, IOC fills at the stated limit price."""
        order = await paper_adapter.place_order(
            market_id="test",
            side=Side.YES,
            order_side=OrderSide.BUY,
            order_type=OrderType.IOC,
            price=0.45,
            size_usd=100.0,
            is_paper=True,
            market=None,  # no market data
        )
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(0.45)
