"""Unit tests for portfolio manager."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.models import Exchange, PositionStatus, ResolutionSource, Side
from src.risk.portfolio import PortfolioManager
from tests.conftest import make_market, make_position


@pytest.fixture
def pm():
    return PortfolioManager(initial_nav_usd=1000.0)


class TestOpenClose:
    def test_opens_position_deducts_cash(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.50)
        market = make_market()
        pm.open_position(pos, market)
        assert pm._cash_usd == 900.0
        assert len(pm.open_positions) == 1

    def test_closes_position_adds_cash(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.50, current_price=0.60)
        market = make_market()
        pm.open_position(pos, market)
        pnl = pm.close_position(pos.position_id, close_price=1.0)
        # shares = 100/0.50 = 200; pnl = (1.0-0.50)*200 = 100
        assert pnl > 0
        assert pm._cash_usd > 900.0  # got money back + profit

    def test_close_nonexistent_raises(self, pm):
        with pytest.raises(KeyError):
            pm.close_position("nonexistent-id", 1.0)

    def test_insufficient_cash_raises(self, pm):
        pos = make_position(size_usd=5000.0)
        market = make_market()
        with pytest.raises(ValueError, match="Insufficient cash"):
            pm.open_position(pos, market)

    def test_loss_on_wrong_resolution(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.80)
        market = make_market()
        pm.open_position(pos, market)
        pnl = pm.close_position(pos.position_id, close_price=0.0)  # lost bet
        assert pnl < 0


class TestMarkToMarket:
    def test_updates_unrealised_pnl(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.50)
        market = make_market()
        pm.open_position(pos, market)
        pm.mark_to_market(pos.position_id, current_price=0.60)
        updated = pm._positions[pos.position_id]
        assert updated.current_price == 0.60
        # pnl = (0.60 - 0.50) * (100/0.50) = 0.10 * 200 = 20
        assert abs(updated.unrealised_pnl - 20.0) < 0.01

    def test_negative_pnl_on_price_drop(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.70)
        market = make_market()
        pm.open_position(pos, market)
        pm.mark_to_market(pos.position_id, current_price=0.40)
        updated = pm._positions[pos.position_id]
        assert updated.unrealised_pnl < 0

    def test_mark_nonexistent_position_no_crash(self, pm):
        pm.mark_to_market("nonexistent", 0.50)  # should not raise


class TestNAVAndDrawdown:
    def test_nav_at_start(self, pm):
        assert pm.nav == 1000.0

    def test_nav_decreases_with_open_position(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.60)
        market = make_market()
        pm.open_position(pos, market)
        # NAV = cash + locked = 900 + 100 = 1000 (unchanged before MtM)
        assert abs(pm.nav - 1000.0) < 0.01

    def test_drawdown_zero_initially(self, pm):
        assert pm.drawdown == 0.0

    def test_drawdown_after_loss(self, pm):
        pos = make_position(size_usd=200.0, entry_price=0.50)
        market = make_market()
        pm.open_position(pos, market)
        pm.close_position(pos.position_id, close_price=0.0)  # total loss
        assert pm.drawdown > 0

    def test_peak_nav_tracks_high_watermark(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.50)
        market = make_market()
        pm.open_position(pos, market)
        pm.mark_to_market(pos.position_id, 0.80)  # paper gain
        snap = pm.compute_snapshot()
        assert snap.peak_nav_usd >= 1000.0


class TestUMADisputeHandling:
    def test_uma_resolution_locks_position(self, pm):
        pos = make_position()
        market = make_market(resolution_source=ResolutionSource.UMA_ORACLE)
        pm.open_position(pos, market)
        pm.handle_resolution(
            pos.position_id,
            resolved_yes=True,
            resolution_source=ResolutionSource.UMA_ORACLE,
        )
        updated = pm._positions[pos.position_id]
        assert updated.status == PositionStatus.LOCKED
        assert updated.uma_dispute_deadline is not None

    def test_kalshi_resolution_closes_immediately(self, pm):
        pos = make_position()
        market = make_market()
        pm.open_position(pos, market)
        pm.handle_resolution(
            pos.position_id,
            resolved_yes=True,
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
        )
        updated = pm._positions[pos.position_id]
        assert updated.status in (PositionStatus.RESOLVED_WIN, PositionStatus.RESOLVED_LOSS)

    def test_dispute_expiration_detection(self, pm):
        pos = make_position()
        market = make_market(resolution_source=ResolutionSource.UMA_ORACLE)
        pm.open_position(pos, market)
        pm.handle_resolution(
            pos.position_id,
            resolved_yes=True,
            resolution_source=ResolutionSource.UMA_ORACLE,
        )
        # Manually set deadline to past
        updated = pm._positions[pos.position_id]
        updated.uma_dispute_deadline = datetime.now(timezone.utc) - timedelta(hours=1)

        finalisable = pm.check_dispute_expirations()
        assert pos.position_id in finalisable

    def test_active_dispute_not_in_finalisable(self, pm):
        pos = make_position()
        market = make_market(resolution_source=ResolutionSource.UMA_ORACLE)
        pm.open_position(pos, market)
        pm.handle_resolution(
            pos.position_id,
            resolved_yes=True,
            resolution_source=ResolutionSource.UMA_ORACLE,
        )
        # Deadline is in the future → not finalisable yet
        finalisable = pm.check_dispute_expirations()
        assert pos.position_id not in finalisable


class TestPortfolioSnapshot:
    def test_snapshot_structure(self, pm):
        snap = pm.compute_snapshot()
        assert snap.total_nav_usd == 1000.0
        assert snap.available_capital_usd == 1000.0
        assert snap.locked_capital_usd == 0.0
        assert snap.current_drawdown_pct == 0.0

    def test_snapshot_with_positions(self, pm):
        pos = make_position(size_usd=200.0, entry_price=0.50)
        market = make_market()
        pm.open_position(pos, market)
        snap = pm.compute_snapshot()
        assert snap.locked_capital_usd == 200.0
        assert snap.available_capital_usd == 800.0

    def test_aroc_report_structure(self, pm):
        pos = make_position(size_usd=100.0, entry_price=0.50, current_price=0.55)
        market = make_market()
        pm.open_position(pos, market)
        pm.mark_to_market(pos.position_id, 0.55)
        report = pm.aroc_report()
        assert len(report) == 1
        assert "aroc_annual" in report[0]
        assert "market" in report[0]

    def test_cashflow_by_week(self, pm):
        pos = make_position(size_usd=100.0, expiry_days=7.0)
        market = make_market()
        pm.open_position(pos, market)
        cashflows = pm.expected_cashflows(weeks=8)
        assert isinstance(cashflows, dict)
        assert len(cashflows) > 0


class TestCheckCanOpen:
    def test_allows_valid_position(self, pm):
        market = make_market()
        allowed, reason = pm.check_can_open(market, 100.0, 1000.0)
        assert allowed
        assert reason == ""

    def test_blocks_insufficient_cash(self, pm):
        market = make_market()
        allowed, reason = pm.check_can_open(market, 5000.0, 1000.0)
        assert not allowed
        assert "Insufficient cash" in reason

    def test_blocks_oversized_position(self, pm):
        market = make_market()
        allowed, reason = pm.check_can_open(market, 999_999.0, 10_000_000.0)
        assert not allowed
