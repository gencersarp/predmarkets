"""
Rich Terminal Dashboard — real-time Bloomberg-style TUI.

Displays:
  - Header: mode, NAV, drawdown, P&L, positions, uptime
  - Arb Opportunities: live ticker sorted by net yield
  - Directional Signals: sorted by EV
  - Portfolio: positions, factor exposures, cashflow calendar, AROC
  - Execution Log: recent actions with reasons and results
  - Strategy Stats: scan counts, signal counts, execution counts
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
_start_time = time.monotonic()


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

    uptime_s = int(time.monotonic() - _start_time)
    h, remainder = divmod(uptime_s, 3600)
    m, s = divmod(remainder, 60)

    text = Text()
    text.append("PMT ", style="bold white")
    text.append(f"[{mode}] ", style=mode_colour)
    text.append(f"NAV: ", style="white")
    nav_colour = "bold green" if snap.total_nav_usd >= settings.paper_initial_balance_usd else "bold red"
    text.append(f"{_usd(snap.total_nav_usd)}", style=nav_colour)
    text.append(f"  Cash: {_usd(snap.available_capital_usd)}", style="dim")
    text.append(f"  Locked: {_usd(snap.locked_capital_usd)}", style="dim")
    text.append(f"  DD: ", style="white")
    dd_colour = "red bold" if snap.current_drawdown_pct > 0.10 else "yellow" if snap.current_drawdown_pct > 0.05 else "green"
    text.append(f"{_pct(snap.current_drawdown_pct)}", style=dd_colour)
    text.append(f"  uPnL: ")
    text.append(f"{_usd(snap.unrealised_pnl_usd)}", style=_colour_pnl(snap.unrealised_pnl_usd))
    text.append(f"  rPnL: ")
    text.append(f"{_usd(snap.realised_pnl_usd)}", style=_colour_pnl(snap.realised_pnl_usd))
    text.append(f"  Pos: {len(snap.positions)}")
    text.append(f"  Up: {h}h{m:02d}m", style="dim")
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
    table.add_column("Type", style="cyan", width=18)
    table.add_column("Market(s)", width=34)
    table.add_column("Net Edge", justify="right", width=8)
    table.add_column("Net $", justify="right", width=8)
    table.add_column("Capital", justify="right", width=8)
    table.add_column("AROC", justify="right", width=7)
    table.add_column("Expiry", width=8)
    table.add_column("Flags", width=20)

    if not opportunities:
        table.add_row("[dim]No arb opportunities — markets efficiently priced[/dim]", *[""] * 7)
    else:
        for opp in opportunities[:8]:
            edge_colour = _colour_edge(opp.net_edge_pct)
            expiry_str = opp.expiry.strftime("%m/%d") if opp.expiry else "?"
            market_str = ", ".join(mid[:16] for mid in opp.market_ids[:2])
            flags_str = " ".join(f.value[:10] for f in opp.risk_flags[:2])

            table.add_row(
                opp.alpha_type.value.replace("_", " "),
                market_str,
                Text(_pct(opp.net_edge_pct), style=edge_colour),
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
    table.add_column("Strategy", width=14)
    table.add_column("Market", width=30)
    table.add_column("Exch", width=5)
    table.add_column("Side", width=4)
    table.add_column("True", justify="right", width=6)
    table.add_column("Impl", justify="right", width=6)
    table.add_column("Edge", justify="right", width=6)
    table.add_column("EV $", justify="right", width=7)
    table.add_column("Size", justify="right", width=6)
    table.add_column("Conf", justify="right", width=5)

    if not signals:
        table.add_row("[dim]No actionable signals yet[/dim]", *[""] * 9)
    else:
        for sig in signals[:8]:
            edge_colour = _colour_edge(sig.edge)
            side_colour = "green" if sig.side.value == "yes" else "red"
            table.add_row(
                sig.alpha_type.value[:12].replace("_", " "),
                sig.market_id[:28],
                sig.exchange.value[:5],
                Text(sig.side.value.upper(), style=side_colour),
                _pct(sig.true_probability),
                _pct(sig.implied_probability),
                Text(_pct(sig.edge), style=edge_colour),
                Text(_usd(sig.expected_value_usd), style=edge_colour),
                f"${sig.recommended_size_usd:.0f}",
                f"{sig.confidence:.0%}",
            )

    return Panel(table, title="[bold]Directional Signals[/bold]", border_style="yellow")


def build_portfolio_panel(orchestrator: "TradingOrchestrator") -> Panel:
    snap = orchestrator.get_portfolio_snapshot()
    lines = []

    # Open positions table
    if snap.positions:
        lines.append("[bold]Open Positions:[/bold]")
        for pos in snap.positions[:6]:
            pnl_colour = _colour_pnl(pos.unrealised_pnl)
            days_str = f"{pos.days_locked:.0f}d" if pos.days_locked else "?"
            side_colour = "green" if pos.side.value == "yes" else "red"
            lines.append(
                f"  [{side_colour}]{pos.side.value.upper():<3}[/{side_colour}] "
                f"{pos.market_title[:28]:<28} "
                f"${pos.size_usd:<6.0f} "
                f"@{pos.entry_price:.2f}->{pos.current_price:.2f} "
                f"[{pnl_colour}]{_usd(pos.unrealised_pnl)}[/{pnl_colour}] "
                f"{days_str}"
            )
        if len(snap.positions) > 6:
            lines.append(f"  ... and {len(snap.positions) - 6} more positions")
    else:
        lines.append("[dim]No open positions[/dim]")

    lines.append("")

    # Factor exposures
    factors = snap.factor_exposures
    if factors:
        lines.append("[bold]Factor Exposures:[/bold]")
        for factor, pct in sorted(factors.items(), key=lambda x: -x[1])[:4]:
            bar_width = int(pct * 25)
            bar = "=" * bar_width
            colour = "red" if pct > 0.35 else "yellow" if pct > 0.20 else "green"
            lines.append(f"  {factor:<20} [{colour}]{bar:<25}[/{colour}] {_pct(pct)}")

    return Panel("\n".join(lines), title="[bold]Portfolio[/bold]", border_style="green")


def build_execution_log(orchestrator: "TradingOrchestrator") -> Panel:
    lines = []

    # Scan stats summary
    sc = orchestrator.scan_counts
    lines.append(
        f"[bold]Scans:[/bold] arb={sc['arb_scans']} "
        f"sig={sc['signal_scans']} "
        f"ofi={sc['ofi_scans']} "
        f"| [bold]Found:[/bold] arb={sc['arb_found']} "
        f"sig={sc['signals_found']} "
        f"ofi={sc['ofi_found']} "
        f"| [bold]Executed:[/bold] {sc['signals_executed']} "
        f"| [bold]Filtered:[/bold] {sc['markets_filtered']}"
    )
    lines.append("")

    # Recent execution log
    log = orchestrator.execution_log
    if log:
        lines.append("[bold]Recent Actions:[/bold]")
        for entry in reversed(log[-12:]):
            result_colour = "green" if entry["result"] == "FILLED" else "yellow" if entry["result"] == "EXECUTING" else "red"
            lines.append(
                f"  [dim]{entry['time']}[/dim] "
                f"[cyan]{entry['action']:<12}[/cyan] "
                f"{entry['market']:<24} "
                f"{entry['detail']:<40} "
                f"[{result_colour}]{entry['result']}[/{result_colour}]"
            )
    else:
        lines.append("[dim]No executions yet — scanning for opportunities...[/dim]")

    return Panel("\n".join(lines), title="[bold]Execution Log & Stats[/bold]", border_style="cyan")


async def run_dashboard(orchestrator: "TradingOrchestrator") -> None:
    """Run the live dashboard in a loop, refreshing every 2 seconds."""
    global _start_time
    _start_time = time.monotonic()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="upper", ratio=3),
        Layout(name="lower", ratio=2),
        Layout(name="footer", ratio=2),
    )
    layout["upper"].split_row(
        Layout(name="arb"),
        Layout(name="signals"),
    )
    layout["lower"].split_row(
        Layout(name="portfolio"),
        Layout(name="exec_log"),
    )

    with Live(layout, refresh_per_second=0.5, screen=True, console=console) as live:
        while True:
            try:
                layout["header"].update(build_header(orchestrator))
                layout["arb"].update(build_arb_table(orchestrator.latest_arb_opportunities))
                layout["signals"].update(build_signals_table(orchestrator.latest_signals))
                layout["portfolio"].update(build_portfolio_panel(orchestrator))
                layout["exec_log"].update(build_execution_log(orchestrator))
                layout["footer"].update(
                    Panel(
                        "[dim]Ctrl+C to stop | Strategies: Arb Scan (10s) | Signal Scan (30s) | OFI (20s) | Near-Expiry (5m)[/dim]",
                        border_style="dim",
                    )
                )
            except Exception as exc:
                layout["header"].update(Panel(f"[red]Dashboard error: {exc}[/red]"))

            await asyncio.sleep(2.0)
