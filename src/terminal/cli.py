"""
CLI entry point for the Prediction Market Terminal.

Commands:
  pmt run         — Start the terminal (paper mode by default)
  pmt run --live  — Live trading mode
  pmt status      — Show current portfolio status
  pmt backtest    — Run historical simulation (see scripts/backtest.py)
  pmt kelly       — Interactive Kelly sizing calculator
"""
from __future__ import annotations

import asyncio
import sys

import click

from src.utils.logging_config import setup_logging


@click.group()
def main() -> None:
    """Prediction Market Terminal — algorithmic trading for Polymarket & Kalshi."""
    pass


@main.command()
@click.option("--live", is_flag=True, default=False, help="Enable live trading (real money).")
@click.option("--dashboard/--no-dashboard", default=True, help="Show live dashboard.")
@click.option("--log-level", default="INFO", help="Log level (DEBUG/INFO/WARNING).")
@click.option("--log-json", is_flag=True, default=False, help="Output JSON logs.")
def run(live: bool, dashboard: bool, log_level: str, log_json: bool) -> None:
    """Start the trading terminal."""
    import os

    if live:
        os.environ.setdefault("PMT_MODE", "live")
        click.echo("WARNING: Live trading mode enabled.", err=True)
        confirm = click.confirm("Are you sure you want to trade with real money?", default=False)
        if not confirm:
            click.echo("Aborted.")
            sys.exit(0)

    setup_logging(level=log_level, json_output=log_json)

    from config.settings import get_settings
    settings = get_settings()

    click.echo(f"Starting PMT in {settings.pmt_mode.value.upper()} mode...")
    asyncio.run(_run_terminal(dashboard=dashboard))


async def _run_terminal(dashboard: bool) -> None:
    from src.terminal.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator()

    if dashboard:
        from src.terminal.dashboard import run_dashboard
        # Run orchestrator and dashboard concurrently
        await asyncio.gather(
            orchestrator.start(),
            run_dashboard(orchestrator),
            return_exceptions=True,
        )
    else:
        await orchestrator.start()


@main.command()
def status() -> None:
    """Show a quick portfolio status snapshot (reads from local DB)."""
    import asyncio

    async def _show_status() -> None:
        from config.settings import get_settings
        from src.utils.database import Database
        settings = get_settings()
        db = Database(settings.database_url)
        await db.init()

        orders = await db.get_paper_orders(limit=10, status="filled")
        signals = await db.get_recent_signals(limit=10)

        click.echo("\n=== Recent Paper Orders ===")
        for o in orders:
            click.echo(
                f"  [{o['exchange']}] {o['side'].upper()} "
                f"${o['size_usd']:.0f} @ {o['price']:.3f} "
                f"({o['alpha_type'] or 'manual'})"
            )

        click.echo("\n=== Recent Signals ===")
        for s in signals:
            click.echo(
                f"  [{s['exchange']}] {s['alpha_type']} {s['side'].upper()} "
                f"edge={s['edge']:.1%} EV=${s['ev_usd']:.2f} "
                f"{'ACTIONABLE' if s['actionable'] else 'skipped'}"
            )

        await db.close()

    asyncio.run(_show_status())


@main.command(name="check")
def check_env() -> None:
    """Validate environment variables and connectivity before starting."""
    from config.settings import get_settings

    settings = get_settings()
    ok = True

    def check(label: str, value, required: bool = False) -> None:
        nonlocal ok
        if value:
            click.echo(f"  [OK]  {label}")
        elif required:
            click.echo(f"  [!!]  {label} — REQUIRED but not set")
            ok = False
        else:
            click.echo(f"  [--]  {label} — not set (optional)")

    click.echo("\n=== PMT Environment Check ===")
    click.echo(f"\nMode: {settings.pmt_mode.value.upper()}")
    click.echo(f"Database: {settings.database_url}")

    click.echo("\n[Polymarket]")
    check("POLYMARKET_PRIVATE_KEY", settings.polymarket_private_key, required=settings.is_live)
    check("POLYMARKET_API_KEY", settings.polymarket_api_key, required=settings.is_live)
    check("POLYMARKET_API_SECRET", settings.polymarket_api_secret, required=settings.is_live)
    check("POLYMARKET_API_PASSPHRASE", settings.polymarket_api_passphrase, required=settings.is_live)

    click.echo("\n[Kalshi]")
    click.echo(f"  Env:  {settings.kalshi_env.value}  →  {settings.kalshi_base_url}")
    check("KALSHI_API_KEY", settings.kalshi_api_key, required=settings.is_live)
    check("KALSHI_API_SECRET", settings.kalshi_api_secret, required=settings.is_live)
    check("KALSHI_PRIVATE_KEY (RSA)", settings.kalshi_private_key)

    click.echo("\n[Oracles / Data]")
    check("NEWSAPI_KEY", settings.newsapi_key)
    check("FRED_API_KEY", settings.fred_api_key)
    check("TWITTER_BEARER_TOKEN", settings.twitter_bearer_token)
    check("SPORTS_API_KEY", settings.sports_api_key)

    click.echo("\n[Risk Limits]")
    click.echo(f"  Max portfolio exposure:  ${settings.max_portfolio_exposure_usd:,.0f}")
    click.echo(f"  Max single position:     ${settings.max_single_position_usd:,.0f}")
    click.echo(f"  Kelly fraction:          {settings.kelly_fraction:.0%}")
    click.echo(f"  Max drawdown:            {settings.max_drawdown_pct:.0%}")
    click.echo(f"  Slippage tolerance:      {settings.slippage_tolerance_pct:.1%}")

    click.echo()
    if ok:
        click.echo("All required settings are present. Ready to run.")
    else:
        click.echo("Some required settings are missing. See above.")
        sys.exit(1)


@main.command()
@click.option("--p-win", prompt="True probability of winning (e.g. 0.60)", type=float)
@click.option("--price", prompt="Market ask price (e.g. 0.45)", type=float)
@click.option("--bankroll", prompt="Available bankroll in USD", type=float, default=1000.0)
@click.option("--fraction", default=0.25, help="Kelly fraction (default: 0.25 = quarter-Kelly)")
@click.option("--fee", default=0.02, help="Taker fee rate (default: 0.02 = 2%)")
def kelly(p_win: float, price: float, bankroll: float, fraction: float, fee: float) -> None:
    """Interactive Kelly sizing calculator."""
    from src.risk.kelly import sizing_report
    report = sizing_report(
        p_win=p_win,
        market_price=price,
        bankroll_usd=bankroll,
        fraction=fraction,
        taker_fee=fee,
    )
    click.echo("\n=== Kelly Sizing Report ===")
    for key, value in report.items():
        if isinstance(value, float):
            click.echo(f"  {key:<35} {value:.4f}")
        else:
            click.echo(f"  {key:<35} {value}")
