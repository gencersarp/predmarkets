"""
Main Trading Orchestrator — the central event loop.

Lifecycle:
  1. Start all data feeds (Polymarket, Kalshi, Oracles)
  2. Initialize portfolio, execution adapters, risk guards
  3. Main loop:
     a. Refresh market universe
     b. Update oracle estimates
     c. Run arb scanner → execute/log opportunities
     d. Run EV/time-decay/mean-reversion engines → execute/log signals
     e. Update portfolio mark-to-market
     f. Sleep until next cycle

All trading is paper-mode by default.
Live mode requires explicit PMT_MODE=live in .env.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from config.settings import get_settings
from src.alpha.arbitrage import ArbitrageScanner
from src.alpha.calibration import get_calibration_tracker
from src.alpha.fundamental import EVEngine
from src.alpha.mean_reversion import MeanReversionDetector
from src.alpha.trend_following import TrendFollowingDetector
from src.alpha.orderflow import OrderFlowAnalyzer
from src.alpha.time_decay import TimeDecaySignalGenerator, fit_poisson_model
from src.core.constants import (
    MARKET_TITLE_BLACKLIST_PATTERNS,
    NEAR_EXPIRY_HOURS,
    NEAR_EXPIRY_SCAN_INTERVAL_SEC,
    POSITION_STOP_LOSS_PCT,
)
from src.core.models import (
    ArbitrageOpportunity,
    DirectionalSignal,
    Exchange,
    Market,
    OracleEstimate,
    RiskFlag,
    Side,
)
from src.data.feeds.kalshi import KalshiFeed
from src.data.feeds.oracles import OracleFeed
from src.data.feeds.polymarket import PolymarketFeed
from src.data.state import TerminalState
from src.execution.paper import PaperExchangeAdapter
from src.execution.router import OrderRouter
from src.risk.kelly import drawdown_adjusted_kelly
from src.risk.portfolio import PortfolioManager
from src.utils.database import Database

logger = logging.getLogger(__name__)

# How often to run each strategy cycle (seconds)
ARB_SCAN_INTERVAL = 10.0
SIGNAL_SCAN_INTERVAL = 30.0
ORDER_FLOW_SCAN_INTERVAL = 20.0    # OFI scan runs every 20s
NEAR_EXPIRY_SCAN_INTERVAL = NEAR_EXPIRY_SCAN_INTERVAL_SEC  # 300s
MARKET_REFRESH_INTERVAL = 60.0
ORACLE_REFRESH_INTERVAL = 300.0
PORTFOLIO_MTM_INTERVAL = 15.0


class TradingOrchestrator:
    """
    Top-level trading orchestrator. Owns all components.

    Usage:
        orchestrator = TradingOrchestrator()
        await orchestrator.start()
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._state = TerminalState()

        # Portfolio
        self._portfolio = PortfolioManager(
            initial_nav_usd=self._settings.paper_initial_balance_usd
        )

        # Database
        self._db = Database(self._settings.database_url)

        # Feeds
        self._poly_feed = PolymarketFeed()
        self._kalshi_feed = KalshiFeed()
        self._oracle_feed = OracleFeed()

        # Execution (paper adapters always; swap for live adapters in LIVE mode)
        self._poly_adapter = PaperExchangeAdapter(
            exchange=Exchange.POLYMARKET,
            initial_balance_usd=self._settings.paper_initial_balance_usd / 2,
            db_log_callback=self._db.log_paper_order,
            taker_fee=0.02,
        )
        self._kalshi_adapter = PaperExchangeAdapter(
            exchange=Exchange.KALSHI,
            initial_balance_usd=self._settings.paper_initial_balance_usd / 2,
            db_log_callback=self._db.log_paper_order,
            taker_fee=0.07,
        )

        if self._settings.is_live:
            logger.warning(
                "LIVE MODE ENABLED — real money execution active. "
                "Ensure credentials are set and risk limits are correct."
            )
            from src.execution.polymarket import PolymarketAdapter
            from src.execution.kalshi import KalshiAdapter
            self._poly_adapter = PolymarketAdapter()
            self._kalshi_adapter = KalshiAdapter()

        self._router = OrderRouter(
            polymarket_adapter=self._poly_adapter,
            kalshi_adapter=self._kalshi_adapter,
            portfolio=self._portfolio,
        )

        # Strategy engines
        self._arb_scanner = ArbitrageScanner()
        self._ev_engine = EVEngine(min_edge_pct=0.05)
        self._mr_detector = MeanReversionDetector(min_edge=0.03)
        self._trend_detector = TrendFollowingDetector()
        self._td_generator = TimeDecaySignalGenerator(min_edge=0.03)
        self._ofi_analyzer = OrderFlowAnalyzer()
        self._calibration = get_calibration_tracker()

        # Shared HTTP session for public API calls (order flow, etc.)
        self._http_session: aiohttp.ClientSession | None = None

        # Timing
        self._last_arb_scan: float = 0.0
        self._last_signal_scan: float = 0.0
        self._last_ofi_scan: float = 0.0
        self._last_near_expiry_scan: float = 0.0
        self._last_market_refresh: float = 0.0
        self._last_oracle_refresh: float = 0.0
        self._last_mtm: float = 0.0
        self._last_status_log: float = 0.0

        # Live signals / opportunities (for dashboard consumption)
        self.latest_arb_opportunities: list[ArbitrageOpportunity] = []
        self.latest_signals: list[DirectionalSignal] = []

        # Execution log for dashboard (last N actions with reasons)
        self.execution_log: list[dict] = []  # [{time, action, market, detail, result}]
        self._max_exec_log = 50

        # Strategy P&L tracking by alpha_type
        self.strategy_stats: dict[str, dict] = {}  # alpha_type -> {trades, wins, pnl}

        # Scan counters (for dashboard info)
        self.scan_counts: dict[str, int] = {
            "arb_scans": 0, "arb_found": 0,
            "signal_scans": 0, "signals_found": 0, "signals_executed": 0,
            "ofi_scans": 0, "ofi_found": 0,
            "markets_filtered": 0,
        }

        self._start_time: float = 0.0
        self._total_orders_placed: int = 0
        self._total_orders_filled: int = 0
        self._rejected_cooldown: dict[str, float] = {}  # market_id -> cooldown_until
        self._closed_at_loss_cooldown: dict[str, float] = {}  # market_id -> cooldown_until
        self._running = False

    # ---------------------------------------------------------------- Lifecycle

    async def start(self) -> None:
        """Start all components and run the main loop."""
        logger.info(
            "Prediction Market Terminal starting — mode=%s",
            self._settings.pmt_mode.value.upper(),
        )

        await self._db.init()

        # Shared HTTP session for public (no-auth) API calls
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "PMT/1.0"},
        )

        # Start feeds — each is isolated so a missing credential doesn't abort startup
        for name, feed in [
            ("polymarket", self._poly_feed),
            ("kalshi", self._kalshi_feed),
            ("oracle", self._oracle_feed),
        ]:
            try:
                await feed.start()
            except Exception as exc:
                logger.warning(
                    "%s feed failed to start (continuing): %s", name, exc
                )

        self._running = True
        self._start_time = time.time()

        # Initial market refresh
        await self._refresh_markets()

        # Set up graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        logger.info("Terminal running — press Ctrl+C to stop")

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        self._print_shutdown_report()
        logger.info("Stopping terminal...")
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await self._poly_feed.stop()
        await self._kalshi_feed.stop()
        await self._oracle_feed.stop()
        await self._db.close()
        logger.info("Terminal stopped")

    def _print_shutdown_report(self) -> None:
        """Print a final summary report on shutdown."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        snap = self._portfolio.compute_snapshot()
        positions = snap.positions
        initial_nav = self._settings.paper_initial_balance_usd

        total_return = snap.total_nav_usd - initial_nav
        total_return_pct = (total_return / initial_nav) * 100 if initial_nav else 0

        report_lines = [
            "",
            "=" * 65,
            "  PREDICTION MARKET TERMINAL — SHUTDOWN REPORT",
            "=" * 65,
            f"  Mode:     {self._settings.pmt_mode.value.upper()}",
            f"  Runtime:  {hours}h {minutes}m {seconds}s",
            f"  Started:  {datetime.fromtimestamp(self._start_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if self._start_time else 'N/A'}",
            "",
            "  --- Portfolio ---",
            f"  Initial NAV:    ${initial_nav:>10.2f}",
            f"  Final NAV:      ${snap.total_nav_usd:>10.2f}",
            f"  Total Return:   ${total_return:>10.2f}  ({total_return_pct:+.2f}%)",
            f"  Unrealised PnL: ${snap.unrealised_pnl_usd:>10.2f}",
            f"  Realised PnL:   ${snap.realised_pnl_usd:>10.2f}",
            f"  Peak NAV:       ${snap.peak_nav_usd:>10.2f}",
            f"  Max Drawdown:   {snap.current_drawdown_pct * 100:>9.1f}%",
            f"  Cash Available: ${snap.available_capital_usd:>10.2f}",
            "",
            "  --- Scanning ---",
            f"  Arb scans:        {self.scan_counts['arb_scans']:>6}",
            f"  Arb found:        {self.scan_counts['arb_found']:>6}",
            f"  Signal scans:     {self.scan_counts['signal_scans']:>6}",
            f"  Signals found:    {self.scan_counts['signals_found']:>6}",
            f"  Signals executed: {self.scan_counts['signals_executed']:>6}",
            f"  OFI scans:        {self.scan_counts['ofi_scans']:>6}",
            f"  OFI found:        {self.scan_counts['ofi_found']:>6}",
            f"  Markets filtered: {self.scan_counts['markets_filtered']:>6}",
        ]

        if positions:
            report_lines.append("")
            report_lines.append(f"  --- Open Positions ({len(positions)}) ---")
            for p in positions:
                pnl_sign = "+" if p.unrealised_pnl >= 0 else ""
                report_lines.append(
                    f"  {p.side.value.upper():<3} {p.market_title[:40]:<40} "
                    f"${p.size_usd:<7.0f} @{p.entry_price:.2f}->{p.current_price:.2f} "
                    f"{pnl_sign}${p.unrealised_pnl:.2f}"
                )

        if self.execution_log:
            report_lines.append("")
            report_lines.append(f"  --- Recent Trades (last {min(10, len(self.execution_log))}) ---")
            for entry in self.execution_log[-10:]:
                report_lines.append(
                    f"  [{entry['time']}] {entry['action']:<10} {entry['market']:<30} "
                    f"{entry['detail']} → {entry['result']}"
                )

        report_lines.append("")
        report_lines.append("=" * 65)

        # Print directly to stdout so it always shows, even if logger is noisy
        print("\n".join(report_lines))

    def _request_shutdown(self) -> None:
        logger.info("Shutdown requested")
        self._running = False
        # Cancel all running tasks so the loop exits immediately instead of
        # waiting for the current scan/sleep to finish naturally.
        loop = asyncio.get_event_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    # ---------------------------------------------------------------- Helpers

    def _log_execution(self, action: str, market: str, detail: str, result: str) -> None:
        """Append to execution log for dashboard display."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "market": market[:30],
            "detail": detail,
            "result": result,
        }
        self.execution_log.append(entry)
        if len(self.execution_log) > self._max_exec_log:
            self.execution_log = self.execution_log[-self._max_exec_log:]

    @staticmethod
    def _is_blacklisted(market: Market) -> bool:
        """Check if a market matches blacklist patterns (test/garbage markets)."""
        title_lower = market.title.lower()
        return any(pat in title_lower for pat in MARKET_TITLE_BLACKLIST_PATTERNS)

    def _filter_markets(self, markets: list[Market]) -> list[Market]:
        """Filter out blacklisted/test markets."""
        filtered = [m for m in markets if not self._is_blacklisted(m)]
        n_removed = len(markets) - len(filtered)
        if n_removed:
            self.scan_counts["markets_filtered"] += n_removed
        return filtered

    # ---------------------------------------------------------------- Main Loop

    async def _main_loop(self) -> None:
        while self._running:
            now = time.monotonic()

            try:
                # Market data refresh
                if now - self._last_market_refresh >= MARKET_REFRESH_INTERVAL:
                    await self._refresh_markets()
                    self._last_market_refresh = now

                # Oracle refresh
                if now - self._last_oracle_refresh >= ORACLE_REFRESH_INTERVAL:
                    await self._refresh_oracles()
                    self._last_oracle_refresh = now

                # Arb scan
                if now - self._last_arb_scan >= ARB_SCAN_INTERVAL:
                    await self._run_arb_scan()
                    self._last_arb_scan = now

                # Directional signal scan
                if now - self._last_signal_scan >= SIGNAL_SCAN_INTERVAL:
                    await self._run_signal_scan()
                    self._last_signal_scan = now

                # Order flow scan (uses public Polymarket API)
                if now - self._last_ofi_scan >= ORDER_FLOW_SCAN_INTERVAL:
                    await self._run_ofi_scan()
                    self._last_ofi_scan = now

                # Near-expiry fast scan (high theta window)
                if now - self._last_near_expiry_scan >= NEAR_EXPIRY_SCAN_INTERVAL:
                    await self._run_near_expiry_scan()
                    self._last_near_expiry_scan = now

                # Portfolio mark-to-market
                if now - self._last_mtm >= PORTFOLIO_MTM_INTERVAL:
                    await self._update_mtm()
                    self._last_mtm = now

                # Periodic portfolio status (every 60s for no-dashboard mode)
                if now - self._last_status_log >= 60.0:
                    self._log_portfolio_status()
                    self._last_status_log = now

                await self._state.tick()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Main loop error: %s", exc, exc_info=True)

            await asyncio.sleep(1.0)

    # ---------------------------------------------------------------- Strategy Cycles

    async def _refresh_markets(self) -> None:
        """Fetch fresh market data from both exchanges."""
        try:
            poly_markets = await self._poly_feed.fetch_active_markets(limit=1000)
            await self._state.upsert_markets(poly_markets)
        except Exception as exc:
            logger.warning("Polymarket market refresh failed: %s", exc)

        try:
            kalshi_markets = await self._kalshi_feed.fetch_active_markets(limit=1000)
            await self._state.upsert_markets(kalshi_markets)
        except Exception as exc:
            logger.warning("Kalshi market refresh failed: %s", exc)

        summary = await self._state.summary()
        logger.info(
            "Markets: poly=%d kalshi=%d total=%d",
            summary["polymarket_markets"],
            summary["kalshi_markets"],
            summary["total_markets"],
        )

    async def _refresh_oracles(self) -> None:
        """Update oracle probability estimates — query per-market topic."""
        markets = await self._state.get_all_markets(max_age_sec=300.0)
        # Pull news for the highest-volume markets (up to 10)
        top_markets = sorted(
            [m for m in markets if m.yes_outcome],
            key=lambda m: m.yes_outcome.volume_24h if m.yes_outcome else 0,
            reverse=True,
        )[:10]
        for market in top_markets:
            try:
                # Use market title as the search query for topical news
                query = market.title[:60]
                await self._oracle_feed.fetch_news(query)
            except Exception as exc:
                logger.debug("Oracle refresh error for %s: %s", market.market_id[:16], exc)

    async def _run_arb_scan(self) -> None:
        """Scan universe for arbitrage opportunities."""
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        if not markets:
            return

        markets = self._filter_markets(markets)
        self.scan_counts["arb_scans"] += 1

        opportunities = self._arb_scanner.scan(markets)
        self.latest_arb_opportunities = opportunities[:20]

        if opportunities:
            self.scan_counts["arb_found"] += len(opportunities)
            logger.info("ARB SCAN — found %d opportunities (%d actionable)",
                        len(opportunities), sum(1 for o in opportunities if o.is_actionable))
            market_map = {m.market_id: m for m in markets}
            
            # Identify markets we already have positions in to avoid duplicates
            held_markets = {p.market_id for p in self._portfolio.open_positions}
            
            executed = 0
            blocked = 0
            for opp in opportunities[:5]:
                await self._db.log_arb_opportunity(opp)
                
                # Deduplicate: don't enter an arb if we already hold any of its markets
                # OR if any of the markets are on a loss-cooldown
                now_ts = time.monotonic()
                if any(mid in held_markets for mid in opp.market_ids):
                    continue
                
                if any(self._closed_at_loss_cooldown.get(mid, 0) > now_ts for mid in opp.market_ids):
                    logger.debug("Skipping Arb %s due to loss cooldown on one or more legs", opp.opp_id[:8])
                    continue

                if opp.is_actionable:
                    orders = await self._router.execute_arb(opp, market_map)
                    if orders and len(orders) == len(opp.legs):
                        executed += 1
                        self._log_execution(
                            "ARB", ", ".join(opp.market_ids[:2]),
                            f"edge=${opp.net_edge_usd:.2f} aroc={opp.aroc_annual*100:.0f}%",
                            "FILLED",
                        )
                        # Record each leg in the portfolio for tracking and P&L
                        from src.core.models import Position, Side
                        for i, order in enumerate(orders):
                            leg_data = opp.legs[i]
                            m = market_map.get(order.market_id)
                            pos = Position(
                                exchange=order.exchange,
                                market_id=order.market_id,
                                market_title=m.title if m else "Arb Leg",
                                side=Side(leg_data["side"]),
                                size_usd=order.filled_size_usd,
                                entry_price=order.avg_fill_price or leg_data["price"],
                                current_price=order.avg_fill_price or leg_data["price"],
                                expiry=opp.expiry,
                                is_paper=order.is_paper,
                                signal_id=f"arb:{opp.opp_id}", # Tag as arb to disable stop-loss
                            )
                            # PortfolioManager handles cash deduction
                            self._portfolio.open_position(pos, m)
                            held_markets.add(order.market_id)
                    else:
                        blocked += 1
            if blocked and not executed:
                logger.info(
                    "ARB SCAN — %d/%d blocked by risk guards (resolution risk, drawdown, etc.)",
                    blocked, len(opportunities),
                )
        else:
            # Only log periodically to reduce noise
            if self.scan_counts["arb_scans"] % 6 == 1:
                poly_n = sum(1 for m in markets if m.exchange == Exchange.POLYMARKET)
                kalshi_n = sum(1 for m in markets if m.exchange == Exchange.KALSHI)
                logger.info(
                    "ARB SCAN — 0 opportunities across %d poly × %d kalshi (efficiently priced)",
                    poly_n, kalshi_n,
                )

    async def _run_signal_scan(self) -> None:
        """Run all directional signal generators."""
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        if not markets:
            return

        markets = self._filter_markets(markets)
        self.scan_counts["signal_scans"] += 1
        logger.info("SIGNAL SCAN — evaluating %d markets across 4 strategies", len(markets))
        all_signals: list[DirectionalSignal] = []

        # 1. EV-based fundamental signals (requires oracle/news data)
        oracle_map: dict[str, list[OracleEstimate]] = {}
        for market in markets:
            estimates = await self._state.get_oracle_estimates(market.market_id)
            if estimates:
                oracle_map[market.market_id] = estimates

        if oracle_map:
            logger.info(
                "  [1/3] FUNDAMENTAL EV — oracle data for %d/%d markets",
                len(oracle_map), len(markets),
            )
            nav = self._portfolio.nav
            ev_signals = self._ev_engine.evaluate_universe(markets, oracle_map, nav)
            all_signals.extend(ev_signals)
            logger.info("  [1/3] FUNDAMENTAL EV — %d signals generated", len(ev_signals))
        else:
            logger.info(
                "  [1/3] FUNDAMENTAL EV — skipped (no oracle data; set NEWSAPI_KEY to enable)"
            )

        # 2. Time-decay signals (Poisson model: does market price decay fast enough?)
        # Model probability is capped at 0.85 to prevent overconfidence on long-horizon
        # markets. This allows scanning up to 30 days out while keeping edge realistic.
        has_expiry = [m for m in markets if m.expiry and m.yes_outcome]
        eligible = [
            m for m in has_expiry
            if m.days_to_expiry is not None
            and 0 < m.days_to_expiry <= 30
        ]
        if self.scan_counts["signal_scans"] <= 2:
            # Diagnostic logging on first few scans to help debug market coverage
            dte_ranges = {"no_expiry": 0, "0-7d": 0, "7-14d": 0, "14-30d": 0, "30d+": 0}
            for m in markets:
                if not m.expiry or not m.yes_outcome:
                    dte_ranges["no_expiry"] += 1
                elif m.days_to_expiry is not None:
                    d = m.days_to_expiry
                    if d <= 7: dte_ranges["0-7d"] += 1
                    elif d <= 14: dte_ranges["7-14d"] += 1
                    elif d <= 30: dte_ranges["14-30d"] += 1
                    else: dte_ranges["30d+"] += 1
            logger.info(
                "  [2/3] TIME DECAY — market breakdown: %s", dte_ranges,
            )
        logger.info(
            "  [2/3] TIME DECAY — checking %d markets (expiry ≤30d, with YES outcome)",
            len(eligible),
        )
        td_count = 0
        for market in eligible:
            model = fit_poisson_model(market)
            if model:
                signal = self._td_generator.generate_signal(market, model)
                if signal:
                    all_signals.append(signal)
                    td_count += 1
                    logger.debug(
                        "  [2/3] TIME DECAY signal: %s %s model=%.1f%% market=%.1f%% edge=%.1f%%",
                        market.market_id, signal.side.value,
                        signal.true_probability * 100,
                        signal.implied_probability * 100,
                        signal.edge * 100,
                    )
        logger.info("  [2/4] TIME DECAY — %d signals from %d eligible markets", td_count, len(eligible))

        # 3. Mean-reversion signals (requires price history — builds up over time)
        mr_count = 0
        trend_count = 0
        for market in markets:
            if market.implied_prob_yes_mid is not None:
                vol = market.yes_outcome.volume_24h if market.yes_outcome else 0.0
                self._mr_detector.update(market.market_id, market.implied_prob_yes_mid, vol)
                self._trend_detector.update(market.market_id, market.implied_prob_yes_mid, vol)

            oracle_prob = await self._state.get_consensus_probability(market.market_id)
            
            # MR check
            mr_signal = self._mr_detector.detect_overreaction(market, oracle_prob)
            if mr_signal:
                all_signals.append(mr_signal)
                mr_count += 1
            
            # Trend check
            trend_signal = self._trend_detector.detect_trend(market)
            if trend_signal:
                all_signals.append(trend_signal)
                trend_count += 1

        logger.info(
            "  [3/4] MEAN REVERSION — %d signals", mr_count
        )
        logger.info(
            "  [4/4] TREND FOLLOWING — %d signals", trend_count
        )

        # Sort by EV and deduplicate by market_id; also skip markets we already hold
        held_markets = {p.market_id for p in self._portfolio.open_positions}
        seen_markets: set[str] = set()
        unique_signals = []
        skipped_held = 0
        for s in sorted(all_signals, key=lambda x: x.expected_value_usd, reverse=True):
            if s.market_id in seen_markets:
                continue
            seen_markets.add(s.market_id)
            if s.market_id in held_markets:
                skipped_held += 1
                continue
            unique_signals.append(s)

        self.latest_signals = unique_signals[:20]

        actionable = [s for s in unique_signals if s.is_actionable]
        self.scan_counts["signals_found"] += len(unique_signals)

        # Diagnostic: show why non-actionable signals are blocked
        if unique_signals and not actionable:
            reasons: dict[str, int] = {}
            for s in unique_signals:
                if s.expected_value_usd <= 0:
                    reasons["EV<=0"] = reasons.get("EV<=0", 0) + 1
                if s.edge < 0.02:
                    reasons["edge<2%"] = reasons.get("edge<2%", 0) + 1
                if RiskFlag.FEE_EXCESSIVE in s.risk_flags:
                    reasons["fee_excessive"] = reasons.get("fee_excessive", 0) + 1
                if RiskFlag.AROC_BELOW_MIN in s.risk_flags:
                    reasons["aroc_low"] = reasons.get("aroc_low", 0) + 1
            logger.info(
                "SIGNAL SCAN COMPLETE — %d signals, 0 actionable (reasons: %s), %d skipped (already held)",
                len(unique_signals), reasons, skipped_held,
            )
        else:
            logger.info(
                "SIGNAL SCAN COMPLETE — %d signals, %d actionable, %d skipped (already held)",
                len(unique_signals), len(actionable), skipped_held,
            )

        executed_count = 0
        for signal in unique_signals:
            await self._db.log_signal(signal)
            self._calibration.record_prediction(
                signal_id=signal.signal_id,
                market_id=signal.market_id,
                alpha_type=signal.alpha_type,
                side=signal.side,
                predicted_prob=signal.true_probability,
            )
            if signal.is_actionable:
                # Skip markets on cooldown (rejected/failed recently)
                now_ts = time.monotonic()
                cooldown_until = self._rejected_cooldown.get(signal.market_id, 0)
                if now_ts < cooldown_until:
                    continue

                # Skip markets closed at a loss recently (cooldown)
                loss_cooldown = self._closed_at_loss_cooldown.get(signal.market_id, 0)
                if now_ts < loss_cooldown:
                    logger.debug("Skipping market %s due to loss cooldown", signal.market_id[:24])
                    continue

                market = next((m for m in markets if m.market_id == signal.market_id), None)
                if market:
                    # Apply drawdown-adjusted sizing before execution
                    dd = self._portfolio.drawdown
                    if dd > 0.05:
                        max_dd = self._settings.max_drawdown_pct
                        scale = max(0.25, 1.0 - dd / max_dd)
                        signal = signal.model_copy(update={
                            "recommended_size_usd": signal.recommended_size_usd * scale,
                        })
                    logger.info(
                        "  EXECUTE %s [%s] %s size=$%.2f edge=%.1f%% EV=$%.2f",
                        signal.alpha_type.value, signal.market_id[:24],
                        signal.side.value, signal.recommended_size_usd,
                        signal.edge * 100, signal.expected_value_usd,
                    )
                    order = await self._router.execute_signal(signal, market)
                    if order and order.status.value == "filled":
                        executed_count += 1
                        self._log_execution(
                            signal.alpha_type.value, market.market_id[:24],
                            f"{signal.side.value} ${order.filled_size_usd:.0f} @ {signal.implied_probability:.1%} edge={signal.edge:.1%}",
                            "FILLED",
                        )
                        from src.core.models import Position, Side
                        pos = Position(
                            exchange=market.exchange,
                            market_id=market.market_id,
                            market_title=market.title,
                            side=signal.side,
                            size_usd=order.filled_size_usd,
                            entry_price=order.avg_fill_price or signal.implied_probability,
                            current_price=order.avg_fill_price or signal.implied_probability,
                            expiry=market.expiry,
                            signal_id=signal.signal_id,
                        )
                        self._portfolio.open_position(pos, market)
                    elif order:
                        # Cooldown rejected/cancelled markets for 5 minutes
                        if order.status.value in ("rejected", "cancelled"):
                            self._rejected_cooldown[signal.market_id] = now_ts + 300.0
                        self._log_execution(
                            signal.alpha_type.value, market.market_id[:24],
                            f"{signal.side.value} ${signal.recommended_size_usd:.0f}",
                            order.status.value.upper(),
                        )
        self.scan_counts["signals_executed"] += executed_count

    async def _run_ofi_scan(self) -> None:
        """
        Order flow imbalance scan using public Polymarket trades API.
        Runs every 20s. Targets the top-volume Polymarket markets.
        """
        if self._http_session is None or self._http_session.closed:
            return
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        markets = self._filter_markets(markets)
        self.scan_counts["ofi_scans"] += 1
        poly_markets = [
            m for m in markets
            if m.exchange == Exchange.POLYMARKET and m.yes_outcome
        ]
        # Focus on top 30 by volume — most likely to have informative OFI
        poly_markets = sorted(
            poly_markets,
            key=lambda m: m.yes_outcome.volume_24h if m.yes_outcome else 0,
            reverse=True,
        )[:30]

        nav = self._portfolio.nav
        ofi_signals = await self._ofi_analyzer.scan_universe(
            self._http_session, poly_markets, bankroll_usd=nav
        )

        # Merge into latest_signals (OFI signals supplement, not replace)
        existing_ids = {s.market_id for s in self.latest_signals}
        new_count = 0
        for sig in ofi_signals:
            if sig.market_id not in existing_ids:
                self.latest_signals.append(sig)
                new_count += 1
                self._calibration.record_prediction(
                    signal_id=sig.signal_id,
                    market_id=sig.market_id,
                    alpha_type=sig.alpha_type,
                    side=sig.side,
                    predicted_prob=sig.true_probability,
                )

        # Trim to 20 signals
        self.latest_signals = sorted(
            self.latest_signals,
            key=lambda s: s.expected_value_usd,
            reverse=True,
        )[:20]

        if new_count > 0:
            logger.info(
                "OFI SCAN — %d new signals from %d markets",
                new_count, len(poly_markets),
            )
        elif self.scan_counts["ofi_scans"] % 6 == 1:
            logger.info(
                "OFI SCAN — 0 signals from %d markets (monitoring buy/sell imbalances)",
                len(poly_markets),
            )

    async def _run_near_expiry_scan(self) -> None:
        """
        Focused scan on markets expiring within NEAR_EXPIRY_HOURS (48h).

        Near-expiry markets have the highest theta and most time pressure.
        Any mispricing here resolves quickly — alpha window is short but fat.
        We run this every 5 minutes instead of 30.
        """
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        markets = self._filter_markets(markets)
        near_expiry = [
            m for m in markets
            if m.days_to_expiry is not None
            and 0 < m.days_to_expiry <= (NEAR_EXPIRY_HOURS / 24.0)
            and m.is_active
        ]
        if not near_expiry:
            return

        logger.info(
            "Near-expiry scan: %d markets within %.0fh",
            len(near_expiry), NEAR_EXPIRY_HOURS,
        )

        # Arb scan on near-expiry markets only
        near_opps = self._arb_scanner.scan(near_expiry)
        for opp in near_opps:
            await self._db.log_arb_opportunity(opp)
            if opp.is_actionable:
                market_map = {m.market_id: m for m in near_expiry}
                await self._router.execute_arb(opp, market_map)

        # Time-decay signals on near-expiry markets
        for market in near_expiry:
            if not market.yes_outcome:
                continue
            model = fit_poisson_model(market)
            if model:
                signal = self._td_generator.generate_signal(market, model)
                if signal and signal.is_actionable:
                    await self._db.log_signal(signal)
                    self._calibration.record_prediction(
                        signal_id=signal.signal_id,
                        market_id=signal.market_id,
                        alpha_type=signal.alpha_type,
                        side=signal.side,
                        predicted_prob=signal.true_probability,
                    )
                    await self._router.execute_signal(signal, market)

    async def _update_mtm(self) -> None:
        """Update portfolio mark-to-market using latest prices."""
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        market_map = {m.market_id: m for m in markets}

        for pos in self._portfolio.open_positions:
            market = market_map.get(pos.market_id)
            if market and market.implied_prob_yes_mid:
                price = (
                    market.implied_prob_yes_mid
                    if pos.side.value == "yes"
                    else 1.0 - market.implied_prob_yes_mid
                )
                self._portfolio.mark_to_market(pos.position_id, price)

        # Check UMA dispute expirations
        finalisable = self._portfolio.check_dispute_expirations()
        for pos_id in finalisable:
            logger.info("Finalising UMA position %s after dispute period", pos_id[:8])

        # Check stop losses
        stop_positions = self._portfolio.check_stop_losses(POSITION_STOP_LOSS_PCT)
        for pos_id in stop_positions:
            pos = self._portfolio._positions.get(pos_id)
            if pos:
                self._log_execution(
                    "STOP LOSS", pos.market_title[:24],
                    f"loss={-pos.unrealised_pnl/pos.size_usd:.0%} > {POSITION_STOP_LOSS_PCT:.0%}",
                    "EXIT",
                )
                self._portfolio.close_position(pos_id, pos.current_price, reason="stop_loss")
                # Cooldown for 1 hour after a stop-loss exit to prevent the "death loop"
                self._closed_at_loss_cooldown[pos.market_id] = time.monotonic() + 3600.0

        snapshot = self._portfolio.compute_snapshot()
        await self._state.set_portfolio_snapshot(snapshot)

    # ---------------------------------------------------------------- Portfolio Status

    def _log_portfolio_status(self) -> None:
        """Log portfolio status periodically (for no-dashboard mode)."""
        snap = self._portfolio.compute_snapshot()
        positions = snap.positions
        pos_summary = ""
        for p in positions[:5]:
            pnl_sign = "+" if p.unrealised_pnl >= 0 else ""
            pos_summary += (
                f"\n    {p.side.value.upper():<3} {p.market_title[:35]:<35} "
                f"${p.size_usd:<6.0f} @{p.entry_price:.2f}->{p.current_price:.2f} "
                f"{pnl_sign}${p.unrealised_pnl:.2f}"
            )
        if len(positions) > 5:
            pos_summary += f"\n    ... and {len(positions) - 5} more"

        logger.info(
            "PORTFOLIO — NAV: $%.2f | Cash: $%.2f | Locked: $%.2f | "
            "uPnL: $%.2f | rPnL: $%.2f | DD: %.1f%% | Positions: %d%s",
            snap.total_nav_usd, snap.available_capital_usd, snap.locked_capital_usd,
            snap.unrealised_pnl_usd, snap.realised_pnl_usd,
            snap.current_drawdown_pct * 100, len(positions), pos_summary,
        )

    def get_portfolio_snapshot(self):
        return self._portfolio.compute_snapshot()
