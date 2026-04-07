"""
Portfolio Manager — maintains NAV, positions, P&L, and AROC tracking.

Key responsibilities:
  1. Track cash balance and locked capital
  2. Compute unrealised P&L from live market prices
  3. Calculate AROC for each position and the aggregate portfolio
  4. Handle UMA dispute period: keep position in LOCKED state until
     dispute window expires, even after apparent resolution
  5. Generate portfolio snapshot for the terminal dashboard
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.core.constants import UMA_DISPUTE_PERIOD_HOURS
from src.core.models import (
    Exchange,
    Market,
    Position,
    PortfolioSnapshot,
    PositionStatus,
    ResolutionSource,
    Side,
)
from src.risk.correlation import FactorExposureTracker

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    In-memory portfolio tracker. In production, back this with TimescaleDB.
    """

    def __init__(self, initial_nav_usd: float = 1000.0) -> None:
        self._initial_nav = initial_nav_usd
        self._cash_usd = initial_nav_usd          # undeployed capital
        self._positions: dict[str, Position] = {} # position_id → Position
        self._realised_pnl: float = 0.0
        self._peak_nav: float = initial_nav_usd
        self._factor_tracker = FactorExposureTracker()

    # ---------------------------------------------------------------- Position Management

    def open_position(
        self,
        position: Position,
        market: Market,
    ) -> None:
        """
        Record a newly opened position.
        Deducts cost from cash balance.
        """
        if position.size_usd > self._cash_usd:
            raise ValueError(
                f"Insufficient cash: need ${position.size_usd:.2f}, "
                f"have ${self._cash_usd:.2f}"
            )
        self._cash_usd -= position.size_usd
        self._positions[position.position_id] = position
        self._factor_tracker.add_position(position, market)
        logger.info(
            "Opened position %s: %s %s $%.2f @ %.3f",
            position.position_id[:8],
            position.side.value,
            position.market_title[:40],
            position.size_usd,
            position.entry_price,
        )

    def close_position(
        self,
        position_id: str,
        close_price: float,
        reason: str = "",
    ) -> float:
        """
        Close a position at `close_price`. Returns realised P&L.
        """
        pos = self._positions.get(position_id)
        if not pos:
            raise KeyError(f"Position {position_id} not found")

        pnl = (close_price - pos.entry_price) * (pos.size_usd / max(pos.entry_price, 1e-6))
        self._realised_pnl += pnl
        self._cash_usd += pos.size_usd + pnl

        pos.status = PositionStatus.RESOLVED_WIN if pnl > 0 else PositionStatus.RESOLVED_LOSS
        pos.resolved_at = datetime.now(timezone.utc)
        pos.unrealised_pnl = 0.0
        self._factor_tracker.remove_position(position_id)

        logger.info(
            "Closed position %s: P&L=%.2f (%s)",
            position_id[:8], pnl, reason,
        )
        return pnl

    def mark_to_market(self, position_id: str, current_price: float) -> None:
        """Update unrealised P&L for an open position."""
        pos = self._positions.get(position_id)
        if not pos:
            return
        pos.current_price = current_price
        # P&L = (current_price - entry_price) * shares_held
        shares = pos.size_usd / max(pos.entry_price, 1e-6)
        pos.unrealised_pnl = (current_price - pos.entry_price) * shares

    def handle_resolution(
        self,
        position_id: str,
        resolved_yes: bool,
        resolution_source: ResolutionSource,
    ) -> None:
        """
        Mark position as resolved. For UMA oracle markets, enter LOCKED state
        and set dispute deadline. For Kalshi, resolve immediately.
        """
        pos = self._positions.get(position_id)
        if not pos:
            return

        if resolution_source == ResolutionSource.UMA_ORACLE:
            # Do NOT immediately close — wait for dispute period
            pos.status = PositionStatus.LOCKED
            pos.uma_dispute_deadline = datetime.now(timezone.utc) + timedelta(
                hours=UMA_DISPUTE_PERIOD_HOURS
            )
            logger.warning(
                "Position %s in UMA dispute lockup until %s",
                position_id[:8],
                pos.uma_dispute_deadline.isoformat(),
            )
        else:
            # Kalshi or other internal resolution — close immediately
            close_price = 1.0 if (
                (pos.side == Side.YES and resolved_yes) or
                (pos.side == Side.NO and not resolved_yes)
            ) else 0.0
            self.close_position(position_id, close_price, reason="market_resolved")

    def check_dispute_expirations(self) -> list[str]:
        """
        Check if any LOCKED positions have passed their UMA dispute window.
        Returns list of position_ids that are now safe to finalise.
        """
        now = datetime.now(timezone.utc)
        finalisable = []
        for pos_id, pos in self._positions.items():
            if (
                pos.status == PositionStatus.LOCKED
                and pos.uma_dispute_deadline
                and now > pos.uma_dispute_deadline
            ):
                finalisable.append(pos_id)
                logger.info(
                    "Position %s dispute period expired — safe to finalise",
                    pos_id[:8],
                )
        return finalisable

    # ---------------------------------------------------------------- Sizing Guard

    def check_can_open(
        self,
        market: Market,
        size_usd: float,
        nav_usd: float,
    ) -> tuple[bool, str]:
        """
        Pre-flight check before opening a position.
        Returns (allowed, reason).
        """
        from config.settings import get_settings
        settings = get_settings()

        # 1. Cash check
        if size_usd > self._cash_usd:
            return False, f"Insufficient cash: ${self._cash_usd:.2f} available"

        # 2. Single position limit
        if size_usd > settings.max_single_position_usd:
            return False, (
                f"Position ${size_usd:.2f} exceeds single-position limit "
                f"${settings.max_single_position_usd:.2f}"
            )

        # 3. Factor concentration
        allowed, reason = self._factor_tracker.check_new_position(market, size_usd, nav_usd)
        if not allowed:
            return False, reason

        return True, ""

    # ---------------------------------------------------------------- Portfolio Analytics

    def compute_snapshot(self) -> PortfolioSnapshot:
        """Compute current portfolio state."""
        open_pos = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.LOCKED)
        ]
        locked_usd = sum(p.size_usd for p in open_pos)
        unrealised_pnl = sum(p.unrealised_pnl for p in open_pos)
        nav = self._cash_usd + locked_usd + unrealised_pnl
        self._peak_nav = max(self._peak_nav, nav)
        drawdown = (self._peak_nav - nav) / max(self._peak_nav, 1.0)

        return PortfolioSnapshot(
            total_nav_usd=nav,
            available_capital_usd=self._cash_usd,
            locked_capital_usd=locked_usd,
            unrealised_pnl_usd=unrealised_pnl,
            realised_pnl_usd=self._realised_pnl,
            peak_nav_usd=self._peak_nav,
            current_drawdown_pct=drawdown,
            positions=list(open_pos),
            factor_exposures=self._factor_tracker.get_exposure_pct(nav),
        )

    def aroc_report(self) -> list[dict]:
        """
        Return AROC for each open position.
        AROC = (expected_gain / capital_locked) * (365 / days_to_expiry)
        """
        report = []
        for pos in self._positions.values():
            if pos.status not in (PositionStatus.OPEN, PositionStatus.LOCKED):
                continue
            if pos.expiry:
                days = max(0.1, (pos.expiry - datetime.now(timezone.utc)).total_seconds() / 86400)
            else:
                days = 30.0
            expected_gain = pos.unrealised_pnl
            aroc = (expected_gain / max(pos.size_usd, 1.0)) * (365.0 / max(days, 1.0))
            report.append({
                "position_id": pos.position_id[:8],
                "market": pos.market_title[:40],
                "side": pos.side.value,
                "size_usd": pos.size_usd,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealised_pnl": pos.unrealised_pnl,
                "days_to_expiry": days,
                "aroc_annual": aroc,
                "status": pos.status.value,
            })
        return sorted(report, key=lambda x: x["aroc_annual"], reverse=True)

    def expected_cashflows(self, weeks: int = 8) -> dict[str, float]:
        """
        Aggregate expected resolution cashflows by week (for liquidity planning).
        Returns: dict of ISO week string → expected cash received
        """
        cashflows: dict[str, float] = {}
        for pos in self._positions.values():
            if pos.expiry is None:
                continue
            week_str = pos.expiry.strftime("%Y-W%W")
            # Expected cash = current_price * shares (rough fair value)
            shares = pos.size_usd / max(pos.entry_price, 1e-6)
            expected = pos.current_price * shares
            cashflows[week_str] = cashflows.get(week_str, 0.0) + expected
        return dict(sorted(cashflows.items()))

    @property
    def nav(self) -> float:
        snap = self.compute_snapshot()
        return snap.total_nav_usd

    @property
    def drawdown(self) -> float:
        snap = self.compute_snapshot()
        return snap.current_drawdown_pct

    @property
    def open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.LOCKED)
        ]

    def has_position_in_market(self, market_id: str) -> bool:
        return any(
            p.market_id == market_id
            for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.LOCKED)
        )

    def get_position_for_market(self, market_id: str) -> Optional[Position]:
        for p in self._positions.values():
            if p.market_id == market_id and p.status in (PositionStatus.OPEN, PositionStatus.LOCKED):
                return p
        return None

    def check_stop_losses(self, max_loss_pct: float = 0.50) -> list[str]:
        """Return position_ids that have breached stop-loss threshold."""
        stop_positions = []
        for pos_id, pos in self._positions.items():
            if pos.status != PositionStatus.OPEN:
                continue
            if pos.size_usd <= 0:
                continue
            
            # Exempt Arbitrage legs from individual stop-losses
            # They are delta-neutral and meant to hedge each other.
            if pos.signal_id and pos.signal_id.startswith("arb:"):
                continue

            loss_pct = -pos.unrealised_pnl / pos.size_usd
            if loss_pct >= max_loss_pct:
                stop_positions.append(pos_id)
                logger.warning(
                    "STOP LOSS triggered for %s: loss=%.1f%% (threshold=%.1f%%)",
                    pos_id[:8], loss_pct * 100, max_loss_pct * 100,
                )
        return stop_positions
