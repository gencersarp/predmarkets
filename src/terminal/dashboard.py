"""
Rich Terminal Dashboard — real-time Bloomberg-style TUI.

Displays:
  - Header: mode, NAV, drawdown, uptime
  - Arb Opportunities: live ticker sorted by net yield
  - Directional Signals: sorted by EV
  - Portfolio: locked capital, factor exposures, cashflow calendar
  - Recent Orders: last N paper trades
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from src.terminal.orchestrator import TradingOrchestrator

console = Console()


def _pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


def _usd(v: float) -> str:
    return f"${v:,.2f}"


def _colour_edge(edge: float) -> str:
    if edge >= 0.08:
        return "bold green"
    if edge >= 0.04:
        return "green"
    if edge >= 0.0:
        return "yellow"
    return "red"


def _colour_pnl(pnl: float) -> str:
    return "green" if pnl >= 0 else "red"


def build_header(orchestrator: "TradingOrchestrator") -> Panel:
    from config.settings import get_settings
    settings = get_settings()

    snap = orchestrator.get_portfolio_snapshot()
    mode = settings.pmt_mode.value.upper()
    mode_colour = "red bold" if settings.is_live else "cyan bold"

    uptime_s = int(time.monotonic())
    h, remainder = divmod(uptime_s, 3600)
    m, s = divmod(remainder, 60)

    text = Text()
    text.append("PMT ", style="bold white")
    text.append(f"[{mode}] ", style=mode_colour)
    text.append(f"NAV: ", style="white")
    text.append(f"{_usd(snap.total_nav_usd)}", style="bold green" if snap.total_nav_usd >= 1000 else "bold red")
    text.append(f"  Available: {_usd(snap.available_capital_usd)}")
    text.append(f"  Locked: {_usd(snap.locked_capital_usd)}")
    text.append(f"  DD: ", style="white")
    dd_colour = "red bold" if snap.current_drawdown_pct > 0.10 else "yellow" if snap.current_drawdown_pct > 0.05 else "green"
    text.append(f"{_pct(snap.current_drawdown_pct)}", style=dd_colour)
    text.append(f"  uPnL: ")
    text.append(f"{_usd(snap.unrealised_pnl_usd)}", style=_colour_pnl(snap.unrealised_pnl_usd))
    text.append(f"  rPnL: ")
    text.append(f"{_usd(snap.realised_pnl_usd)}", style=_colour_pnl(snap.realised_pnl_usd))
    text.append(f"  Positions: {len(snap.positions)}")
    text.append(f"  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}", style="dim")

    return Panel(text, title="Prediction Market Terminal", border_style="blue")


def build_arb_table(opportunities) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Type", style="cyan", width=20)
    table.add_column("Market(s)", width=36)
    table.add_column("Net Edge", justify="right", width=10)
    table.add_column("Net $", justify="right", width=8)
    table.add_column("Capital", justify="right", width=8)
    table.add_column("AROC", justify="right", width=8)
    table.add_column("Expiry", width=10)
    table.add_column("Flags", width=24)

    if not opportunities:
        table.add_row("[dim]No arb opportunities detected[/dim]", "", "", "", "", "", "", "")
    else:
        for opp in opportunities[:10]:
            edge_str = _pct(opp.net_edge_pct)
            edge_colour = _colour_edge(opp.net_edge_pct)
            expiry_str = opp.expiry.strftime("%m/%d") if opp.expiry else "?"
            market_str = ", ".join(mid[:18] for mid in opp.market_ids[:2])
            flags_str = " ".join(f.value[:12] for f in opp.risk_flags[:2])

            table.add_row(
                opp.alpha_type.value.replace("_", " "),
                market_str,
                Text(edge_str, style=edge_colour),
                Text(_usd(opp.net_edge_usd), style=edge_colour),
                _usd(opp.required_capital_usd),
                f"{opp.aroc_annual * 100:.0f}%",
                expiry_str,
                Text(flags_str or "-", style="yellow" if flags_str else "dim"),
            )

    return Panel(table, title="[bold]Arbitrage Opportunities[/bold]", border_style="magenta")


def build_signals_table(signals) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=box.SIMPLE_HEAVY,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Type", width=16)
    table.add_column("Market", width=36)
    table.add_column("Exch", width=6)
    table.add_column("Side", width=5)
    table.add_column("True P", justify="right", width=7)
    table.add_column("Impl P", justify="right", width=7)
    table.add_column("Edge", justify="right", width=6)
    table.add_column("EV $", justify="right", width=8)
    table.add_column("Size $", justify="right", width=7)
    table.add_column("AROC", justify="right", width=7)
    table.add_column("Conf", justify="right", width=5)

    if not signals:
        table.add_row("[dim]No signals — waiting for oracle data[/dim]", *[""] * 10)
    else:
        for sig in signals[:10]:
            edge_colour = _colour_edge(sig.edge)
            side_colour = "green" if sig.side.value == "yes" else "red"
            table.add_row(
                sig.alpha_type.value[:14].replace("_", " "),
                sig.market_id[:34],
                sig.exchange.value[:6],
                Text(sig.side.value.upper(), style=side_colour),
                _pct(sig.true_probability),
                _pct(sig.implied_probability),
                Text(_pct(sig.edge), style=edge_colour),
                Text(_usd(sig.expected_value_usd), style=edge_colour),
                _usd(sig.recommended_size_usd),
                f"{sig.aroc_annual * 100:.0f}%",
                f"{sig.confidence:.0%}",
            )

    return Panel(table, title="[bold]Directional Signals[/bold]", border_style="yellow")


def build_portfolio_panel(orchestrator: "TradingOrchestrator") -> Panel:
    snap = orchestrator.get_portfolio_snapshot()
    cashflows = orchestrator._portfolio.expected_cashflows(weeks=4)
    factors = snap.factor_exposures

    lines = []

    # Factor exposures
    if factors:
        lines.append("[bold]Factor Exposures:[/bold]")
        for factor, pct in sorted(factors.items(), key=lambda x: -x[1]):
            bar_width = int(pct * 30)
            bar = "█" * bar_width
            colour = "red" if pct > 0.35 else "yellow" if pct > 0.20 else "green"
            lines.append(f"  {factor:<24} [{colour}]{bar:<30}[/{colour}] {_pct(pct)}")
    else:
        lines.append("[dim]No factor exposures tracked yet[/dim]")

    lines.append("")

    # Cashflow calendar
    if cashflows:
        lines.append("[bold]Expected Cashflows by Week:[/bold]")
        for week, amount in list(cashflows.items())[:4]:
            lines.append(f"  Week {week}  {_usd(amount)}")

    # AROC breakdown
    aroc_report = orchestrator._portfolio.aroc_report()
    if aroc_report:
        lines.append("")
        lines.append("[bold]Open Positions AROC:[/bold]")
        for pos in aroc_report[:5]:
            colour = _colour_pnl(pos["unrealised_pnl"])
            lines.append(
                f"  {pos['market'][:30]:<30} "
                f"{pos['side'].upper():<4} "
                f"${pos['size_usd']:<7.0f} "
                f"uPnL=[{colour}]{_usd(pos['unrealised_pnl'])}[/{colour}] "
                f"AROC={pos['aroc_annual'] * 100:.0f}%"
            )

    return Panel("\n".join(lines), title="[bold]Portfolio Greeks & Cashflows[/bold]", border_style="green")


async def run_dashboard(orchestrator: "TradingOrchestrator") -> None:
    """Run the live dashboard in a loop, refreshing every 2 seconds."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=16),
    )
    layout["body"].split_row(
        Layout(name="arb"),
        Layout(name="signals"),
    )

    with Live(layout, refresh_per_second=0.5, screen=True, console=console) as live:
        while True:
            try:
                layout["header"].update(build_header(orchestrator))
                layout["arb"].update(build_arb_table(orchestrator.latest_arb_opportunities))
                layout["signals"].update(build_signals_table(orchestrator.latest_signals))
                layout["footer"].update(build_portfolio_panel(orchestrator))
            except Exception as exc:
                layout["header"].update(Panel(f"[red]Dashboard error: {exc}[/red]"))

            await asyncio.sleep(2.0)
