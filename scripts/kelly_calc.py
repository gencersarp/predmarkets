#!/usr/bin/env python3
"""
Standalone Kelly Criterion calculator with full sizing diagnostics.

Usage:
    python scripts/kelly_calc.py
    python scripts/kelly_calc.py --p-win 0.65 --price 0.50 --bankroll 1000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Kelly Sizing Calculator")
    parser.add_argument("--p-win", type=float, help="True probability of winning (0-1)")
    parser.add_argument("--price", type=float, help="Market ask price per $1 payout (0-1)")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--fraction", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--fee", type=float, default=0.02, help="Taker fee rate")
    return parser.parse_args()


def interactive_mode():
    print("\n=== PMT Kelly Sizing Calculator ===\n")
    p_win = float(input("True probability of winning (e.g. 0.65): "))
    price = float(input("Market ask price per $1 payout (e.g. 0.45): "))
    bankroll = float(input("Available bankroll in USD [1000]: ") or 1000)
    fraction = float(input("Kelly fraction [0.25 = quarter-Kelly]: ") or 0.25)
    fee = float(input("Taker fee rate [0.02 = 2%]: ") or 0.02)
    return p_win, price, bankroll, fraction, fee


def print_report(report: dict):
    print("\n" + "="*55)
    print("KELLY SIZING REPORT")
    print("="*55)

    fields = [
        ("True probability (p_win)", f"{report['p_win']:.2%}"),
        ("Market price (implied prob)", f"{report['implied_prob']:.2%}"),
        ("Edge (true - implied)", f"{report['edge']:+.2%}"),
        ("Decimal odds", f"{report['decimal_odds']:.2f}x"),
        ("Net profit per unit (b)", f"{report['net_profit_b']:.4f}"),
        ("EV per unit staked", f"{report['ev_per_unit']:+.4f}"),
        ("Breakeven probability", f"{report['breakeven_prob']:.2%}"),
        ("Required prob (after fees)", f"{report['required_prob_after_fees']:.2%}"),
        ("", ""),
        ("Full Kelly fraction", f"{report['full_kelly_fraction']:.4f} ({report['full_kelly_fraction']:.2%})"),
        ("Scaled Kelly fraction", f"{report['scaled_kelly_fraction']:.4f} ({report['scaled_kelly_fraction']:.2%})"),
        ("RECOMMENDED BET SIZE", f"${report['recommended_size_usd']:.2f}"),
        ("", ""),
        ("Log growth rate (G)", f"{report['log_growth_rate']:.6f}"),
        ("Ruin prob (100 bets, 50% loss)", f"{report['ruin_prob_100bets']:.2%}"),
    ]

    for label, value in fields:
        if label == "":
            print()
        elif label == "RECOMMENDED BET SIZE":
            print(f"  {'>>> ' + label:<38} {value}")
        else:
            print(f"  {label:<38} {value}")

    print()
    ev = report["ev_per_unit"]
    size = report["recommended_size_usd"]
    if ev > 0 and size > 0:
        print(f"  Expected profit on this bet: ${ev * size:.2f}")
        print()

    if report["edge"] <= 0:
        print("  [!] NEGATIVE EDGE: Do NOT bet on this market.")
    elif report["edge"] < 0.03:
        print("  [!] THIN EDGE: Fees may eat this margin. Reconsider.")
    elif report["ruin_prob_100bets"] > 0.20:
        print("  [!] HIGH RUIN RISK: Consider reducing Kelly fraction.")
    else:
        print("  [+] Signal looks tradeable.")

    print("="*55 + "\n")


def main():
    args = parse_args()

    if args.p_win is None or args.price is None:
        p_win, price, bankroll, fraction, fee = interactive_mode()
    else:
        p_win = args.p_win
        price = args.price
        bankroll = args.bankroll
        fraction = args.fraction
        fee = args.fee

    from src.risk.kelly import sizing_report
    report = sizing_report(
        p_win=p_win,
        market_price=price,
        bankroll_usd=bankroll,
        fraction=fraction,
        taker_fee=fee,
    )
    print_report(report)


if __name__ == "__main__":
    main()
