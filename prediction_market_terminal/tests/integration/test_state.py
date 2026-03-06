"""Integration tests for the TerminalState manager."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from src.core.models import Exchange, OracleEstimate
from src.data.state import TerminalState
from tests.conftest import make_market


@pytest.fixture
def state():
    return TerminalState()


class TestMarketCRUD:
    async def test_upsert_and_retrieve(self, state):
        m = make_market("pm-001", Exchange.POLYMARKET)
        await state.upsert_markets([m])
        retrieved = await state.get_market(Exchange.POLYMARKET, "pm-001")
        assert retrieved is not None
        assert retrieved.market_id == "pm-001"

    async def test_get_all_markets_by_exchange(self, state):
        pm = make_market("pm-001", Exchange.POLYMARKET)
        km = make_market("kal-001", Exchange.KALSHI)
        await state.upsert_markets([pm, km])
        poly_markets = await state.get_all_markets(exchange=Exchange.POLYMARKET)
        kalshi_markets = await state.get_all_markets(exchange=Exchange.KALSHI)
        assert all(m.exchange == Exchange.POLYMARKET for m in poly_markets)
        assert all(m.exchange == Exchange.KALSHI for m in kalshi_markets)

    async def test_returns_none_for_missing_market(self, state):
        result = await state.get_market(Exchange.POLYMARKET, "nonexistent")
        assert result is None

    async def test_upsert_overwrites_existing(self, state):
        m1 = make_market("pm-001", title="Original title")
        await state.upsert_markets([m1])
        m2 = make_market("pm-001", title="Updated title")
        await state.upsert_markets([m2])
        retrieved = await state.get_market(Exchange.POLYMARKET, "pm-001")
        assert retrieved is not None
        assert retrieved.title == "Updated title"

    async def test_market_pair_search(self, state):
        m1 = make_market("p1", title="Will the Fed cut rates in December?")
        m2 = make_market("p2", title="Will the Fed cut rates in January?")
        m3 = make_market("p3", title="Will Bitcoin hit $100k?")
        await state.upsert_markets([m1, m2, m3])
        results = await state.get_market_pair("Fed cut")
        assert len(results) == 2


class TestOracleCRUD:
    async def test_upsert_and_retrieve_oracle(self, state):
        estimate = OracleEstimate(
            source="test_model",
            market_id="pm-001",
            true_probability=0.65,
            confidence_interval_low=0.60,
            confidence_interval_high=0.70,
        )
        await state.upsert_oracle_estimate("pm-001", estimate)
        estimates = await state.get_oracle_estimates("pm-001")
        assert len(estimates) == 1
        assert estimates[0].true_probability == 0.65

    async def test_same_source_overwritten(self, state):
        e1 = OracleEstimate(source="model_a", market_id="m1", true_probability=0.60)
        e2 = OracleEstimate(source="model_a", market_id="m1", true_probability=0.70)
        await state.upsert_oracle_estimate("m1", e1)
        await state.upsert_oracle_estimate("m1", e2)
        estimates = await state.get_oracle_estimates("m1")
        assert len(estimates) == 1
        assert estimates[0].true_probability == 0.70

    async def test_different_sources_accumulated(self, state):
        e1 = OracleEstimate(source="model_a", market_id="m1", true_probability=0.60)
        e2 = OracleEstimate(source="model_b", market_id="m1", true_probability=0.65)
        await state.upsert_oracle_estimate("m1", e1)
        await state.upsert_oracle_estimate("m1", e2)
        estimates = await state.get_oracle_estimates("m1")
        assert len(estimates) == 2

    async def test_consensus_probability_weighted(self, state):
        # model_a: narrow CI → high weight → should dominate
        e1 = OracleEstimate(
            source="model_a", market_id="m1", true_probability=0.80,
            confidence_interval_low=0.78, confidence_interval_high=0.82,
        )
        # model_b: wide CI → low weight
        e2 = OracleEstimate(
            source="model_b", market_id="m1", true_probability=0.20,
            confidence_interval_low=0.0, confidence_interval_high=0.80,
        )
        await state.upsert_oracle_estimate("m1", e1)
        await state.upsert_oracle_estimate("m1", e2)
        consensus = await state.get_consensus_probability("m1")
        assert consensus is not None
        assert consensus > 0.60  # narrow model_a dominates

    async def test_consensus_none_without_estimates(self, state):
        prob = await state.get_consensus_probability("unknown-market")
        assert prob is None


class TestSummary:
    async def test_summary_structure(self, state):
        pm = make_market("pm-001", Exchange.POLYMARKET)
        km = make_market("kal-001", Exchange.KALSHI)
        await state.upsert_markets([pm, km])
        summary = await state.summary()
        assert summary["total_markets"] == 2
        assert summary["polymarket_markets"] == 1
        assert summary["kalshi_markets"] == 1
        assert "uptime_sec" in summary
        assert "tick_count" in summary

    async def test_tick_increments(self, state):
        initial = state._tick_count
        await state.tick()
        assert state._tick_count == initial + 1


class TestPortfolioState:
    async def test_upsert_and_get_position(self, state):
        from tests.conftest import make_position
        pos = make_position()
        await state.upsert_position(pos)
        positions = await state.get_open_positions()
        assert any(p.position_id == pos.position_id for p in positions)

    async def test_remove_position(self, state):
        from tests.conftest import make_position
        pos = make_position()
        await state.upsert_position(pos)
        await state.remove_position(pos.position_id)
        positions = await state.get_open_positions()
        assert not any(p.position_id == pos.position_id for p in positions)
