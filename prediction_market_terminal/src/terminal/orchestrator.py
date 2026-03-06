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
from datetime import datetime
from typing import Optional

import aiohttp

from config.settings import get_settings
from src.alpha.arbitrage import ArbitrageScanner
from src.alpha.calibration import get_calibration_tracker
from src.alpha.fundamental import EVEngine
from src.alpha.mean_reversion import MeanReversionDetector
from src.alpha.orderflow import OrderFlowAnalyzer
from src.alpha.time_decay import TimeDecaySignalGenerator, fit_poisson_model
from src.core.constants import NEAR_EXPIRY_HOURS, NEAR_EXPIRY_SCAN_INTERVAL_SEC
from src.core.models import (
    ArbitrageOpportunity,
    DirectionalSignal,
    Exchange,
    Market,
    OracleEstimate,
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
        )
        self._kalshi_adapter = PaperExchangeAdapter(
            exchange=Exchange.KALSHI,
            initial_balance_usd=self._settings.paper_initial_balance_usd / 2,
            db_log_callback=self._db.log_paper_order,
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
        self._mr_detector = MeanReversionDetector()
        self._td_generator = TimeDecaySignalGenerator(min_edge=0.04)
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

        # Live signals / opportunities (for dashboard consumption)
        self.latest_arb_opportunities: list[ArbitrageOpportunity] = []
        self.latest_signals: list[DirectionalSignal] = []

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
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        logger.info("Stopping terminal...")
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await self._poly_feed.stop()
        await self._kalshi_feed.stop()
        await self._oracle_feed.stop()
        await self._db.close()
        logger.info("Terminal stopped")

    def _request_shutdown(self) -> None:
        logger.info("Shutdown requested")
        self._running = False

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
            poly_markets = await self._poly_feed.fetch_active_markets(limit=200)
            await self._state.upsert_markets(poly_markets)
        except Exception as exc:
            logger.warning("Polymarket market refresh failed: %s", exc)

        try:
            kalshi_markets = await self._kalshi_feed.fetch_active_markets(limit=200)
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

        poly_markets = [m for m in markets if m.exchange == Exchange.POLYMARKET]
        kalshi_markets = [m for m in markets if m.exchange == Exchange.KALSHI]
        logger.info(
            "ARB SCAN — checking %d Polymarket × %d Kalshi markets for price discrepancies",
            len(poly_markets), len(kalshi_markets),
        )

        opportunities = self._arb_scanner.scan(markets)
        self.latest_arb_opportunities = opportunities[:20]

        if not opportunities:
            logger.info("ARB SCAN — no opportunities found (markets are efficiently priced)")
        else:
            logger.info("ARB SCAN — found %d opportunities (%d actionable)",
                        len(opportunities), sum(1 for o in opportunities if o.is_actionable))

        for opp in opportunities[:5]:  # log top 5
            await self._db.log_arb_opportunity(opp)
            if opp.is_actionable:
                logger.info(
                    "ARB EXECUTE — type=%s net_edge=$%.2f aroc=%.0f%% flags=%s",
                    opp.alpha_type.value,
                    opp.net_edge_usd,
                    opp.aroc_annual * 100,
                    [f.value for f in opp.risk_flags],
                )
                if self._settings.is_live or True:  # execute in paper mode too
                    market_map = {m.market_id: m for m in markets}
                    await self._router.execute_arb(opp, market_map)

    async def _run_signal_scan(self) -> None:
        """Run all directional signal generators."""
        markets = await self._state.get_all_markets(max_age_sec=120.0)
        if not markets:
            return

        logger.info("SIGNAL SCAN — evaluating %d markets across 3 strategies", len(markets))
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
        eligible = [m for m in markets if m.expiry and m.yes_outcome]
        logger.info(
            "  [2/3] TIME DECAY — checking %d markets with expiry and YES outcome",
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
        logger.info("  [2/3] TIME DECAY — %d signals from %d eligible markets", td_count, len(eligible))

        # 3. Mean-reversion signals (requires price history — builds up over time)
        mr_count = 0
        for market in markets:
            if market.implied_prob_yes_mid is not None:
                self._mr_detector.update(
                    market.market_id,
                    market.implied_prob_yes_mid,
                    market.yes_outcome.volume_24h if market.yes_outcome else 0.0,
                )
            oracle_prob = await self._state.get_consensus_probability(market.market_id)
            mr_signal = self._mr_detector.detect_overreaction(market, oracle_prob)
            if mr_signal:
                all_signals.append(mr_signal)
                mr_count += 1
        logger.info(
            "  [3/3] MEAN REVERSION — %d signals (needs price history; more signals appear over time)",
            mr_count,
        )

        # Sort by EV and deduplicate by market_id
        seen_markets = set()
        unique_signals = []
        for s in sorted(all_signals, key=lambda x: x.expected_value_usd, reverse=True):
            if s.market_id not in seen_markets:
                seen_markets.add(s.market_id)
                unique_signals.append(s)

        self.latest_signals = unique_signals[:20]

        actionable = [s for s in unique_signals if s.is_actionable]
        logger.info(
            "SIGNAL SCAN COMPLETE — %d total signals, %d actionable, executing now",
            len(unique_signals), len(actionable),
        )

        for signal in unique_signals:
            await self._db.log_signal(signal)
            # Record in calibration tracker
            self._calibration.record_prediction(
                signal_id=signal.signal_id,
                market_id=signal.market_id,
                alpha_type=signal.alpha_type,
                side=signal.side,
                predicted_prob=signal.true_probability,
            )
            if signal.is_actionable:
                market = next((m for m in markets if m.market_id == signal.market_id), None)
                if market:
                    # Apply drawdown-adjusted sizing before execution
                    dd = self._portfolio.drawdown
                    if dd > 0.05:  # only scale once DD > 5%
                        max_dd = self._settings.max_drawdown_pct
                        scale = max(0.0, 1.0 - dd / max_dd)
                        signal = signal.model_copy(update={
                            "recommended_size_usd": signal.recommended_size_usd * scale,
                        })
                        logger.info(
                            "  Drawdown %.1f%% — scaling %s size to $%.2f",
                            dd * 100, signal.market_id[:20], signal.recommended_size_usd,
                        )
                    logger.info(
                        "  EXECUTE %s [%s] %s size=$%.2f edge=%.1f%% EV=$%.2f",
                        signal.alpha_type.value, signal.market_id[:24],
                        signal.side.value, signal.recommended_size_usd,
                        signal.edge * 100, signal.expected_value_usd,
                    )
                    await self._router.execute_signal(signal, market)

    async def _run_ofi_scan(self) -> None:
        """
        Order flow imbalance scan using public Polymarket trades API.
        Runs every 20s. Targets the top-volume Polymarket markets.
        """
        if self._http_session is None or self._http_session.closed:
            return
        markets = await self._state.get_all_markets(max_age_sec=120.0)
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

        logger.info(
            "OFI SCAN — watching trade flow on top %d Polymarket markets by volume",
            len(poly_markets),
        )

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

        logger.info(
            "OFI SCAN — %d new signals from %d markets (looks for large buy/sell imbalances)",
            new_count, len(poly_markets),
        )

    async def _run_near_expiry_scan(self) -> None:
        """
        Focused scan on markets expiring within NEAR_EXPIRY_HOURS (48h).

        Near-expiry markets have the highest theta and most time pressure.
        Any mispricing here resolves quickly — alpha window is short but fat.
        We run this every 5 minutes instead of 30.
        """
        markets = await self._state.get_all_markets(max_age_sec=120.0)
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

        snapshot = self._portfolio.compute_snapshot()
        await self._state.set_portfolio_snapshot(snapshot)

    # ---------------------------------------------------------------- Portfolio Access

    def get_portfolio_snapshot(self):
        return self._portfolio.compute_snapshot()
