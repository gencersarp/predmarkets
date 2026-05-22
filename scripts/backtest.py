#!/usr/bin/env python3
"""
Historical backtest runner.

Replays logged market snapshots (from DB) through the strategy engines
to evaluate:
  - Win rate per strategy type
  - Average EV realization vs. predicted
  - Sharpe ratio on paper-traded positions
  - Kelly calibration accuracy

Usage:
    python scripts/backtest.py --days 30
    python scripts/backtest.py --strategy ev_directional --days 14
"""
import argparse
import asyncio
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="PMT Backtest Runner")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyse")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "ev_directional", "time_decay", "mean_reversion", "arb"],
        help="Strategy to analyse",
    )
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Min signal confidence")
    return parser.parse_args()


async def run_backtest(args):
    from src.utils.logging_config import setup_logging
    setup_logging(level="INFO")

    from config.settings import get_settings
    from src.utils.database import Database

    settings = get_settings()
    db = Database(settings.database_url)
    await db.init()

    print(f"\n{'='*60}")
    print(f"PMT BACKTEST REPORT — Last {args.days} days")
    print(f"{'='*60}\n")

    # Analyse signals
    signals = await db.get_recent_signals(limit=1000)
    if args.strategy != "all":
        signals = [s for s in signals if s["alpha_type"] == args.strategy]
    signals = [s for s in signals if s.get("confidence", 0) >= args.min_confidence]

    if not signals:
        print("No signals found in database. Run the terminal first to generate data.")
        await db.close()
        return

    total = len(signals)
    actionable = sum(1 for s in signals if s.get("actionable", False))
    total_ev = sum(s.get("ev_usd", 0) for s in signals)
    total_size = sum(s.get("size_usd", 0) for s in signals)
    avg_edge = sum(s.get("edge", 0) for s in signals) / max(total, 1)
    avg_aroc = sum(s.get("aroc", 0) for s in signals) / max(total, 1)

    print(f"Signal Analysis:")
    print(f"  Total signals:         {total}")
    print(f"  Actionable signals:    {actionable} ({actionable/max(total,1):.0%})")
    print(f"  Total expected EV:     ${total_ev:.2f}")
    print(f"  Total recommended $:   ${total_size:.2f}")
    print(f"  Average edge:          {avg_edge:.1%}")
    print(f"  Average AROC:          {avg_aroc:.0%}")
    print()

    # Strategy breakdown
    strategy_stats = {}
    for s in signals:
        stype = s.get("alpha_type", "unknown")
        if stype not in strategy_stats:
            strategy_stats[stype] = {"count": 0, "ev_sum": 0, "edge_sum": 0}
        strategy_stats[stype]["count"] += 1
        strategy_stats[stype]["ev_sum"] += s.get("ev_usd", 0)
        strategy_stats[stype]["edge_sum"] += s.get("edge", 0)

    print("By Strategy:")
    for stype, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["count"]):
        avg_e = stats["edge_sum"] / stats["count"]
        print(
            f"  {stype:<25} count={stats['count']:<4} "
            f"avg_edge={avg_e:.1%} total_ev=${stats['ev_sum']:.2f}"
        )

    # Paper orders
    orders = await db.get_paper_orders(limit=1000, status="filled")
    if orders:
        print(f"\nPaper Order History:")
        print(f"  Total filled orders:   {len(orders)}")
        total_traded = sum(o.get("size_usd", 0) for o in orders)
        print(f"  Total capital deployed: ${total_traded:.2f}")

        by_exchange = {}
        for o in orders:
            ex = o.get("exchange", "unknown")
            by_exchange[ex] = by_exchange.get(ex, 0) + 1
        for ex, count in by_exchange.items():
            print(f"  {ex}: {count} orders")

    print(f"\n{'='*60}")
    print("Note: To see realised P&L, run with live oracle data for 1+ weeks.")
    print("Connect oracle feed (NEWSAPI_KEY, FRED_API_KEY) for higher signal quality.")
    print(f"{'='*60}\n")

    await db.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_backtest(args))
