#!/usr/bin/env python3
"""
Quick-start script for the Prediction Market Terminal.

Usage:
    python scripts/run_terminal.py              # paper mode, with dashboard
    python scripts/run_terminal.py --no-dash    # paper mode, logs only
    python scripts/run_terminal.py --live       # live mode (requires credentials)
    python scripts/run_terminal.py --debug      # verbose logging
"""
import argparse
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction Market Terminal")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--no-dash", action="store_true", help="Disable dashboard (log-only mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


async def main(args):
    from src.utils.logging_config import setup_logging
    setup_logging(level="DEBUG" if args.debug else "INFO")

    if args.live:
        os.environ["PMT_MODE"] = "live"
        print("WARNING: Live trading enabled. Real money will be used.")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    from src.terminal.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator()

    if not args.no_dash:
        from src.terminal.dashboard import run_dashboard
        await asyncio.gather(
            orchestrator.start(),
            run_dashboard(orchestrator),
            return_exceptions=True,
        )
    else:
        await orchestrator.start()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nTerminal stopped.")
