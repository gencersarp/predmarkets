"""Immutable system-wide constants."""
from __future__ import annotations

# ---- Fee structures (as decimal fractions) ---------------------------------
POLYMARKET_TAKER_FEE: float = 0.02        # 2% taker
POLYMARKET_MAKER_FEE: float = 0.00        # 0% maker
KALSHI_TAKER_FEE: float = 0.07            # ~7% of winnings (approximation)
KALSHI_MAKER_FEE: float = 0.03            # ~3% of winnings

# ---- Polygon gas estimate (in USD per transaction) -------------------------
POLYGON_GAS_ESTIMATE_USD: float = 0.01    # typical ~$0.01 on Polygon

# ---- Arbitrage thresholds ---------------------------------------------------
# Minimum net edge (after fees) to consider a trade worth executing
MIN_NET_EDGE_PCT: float = 0.0075          # 0.75%
# Mutually exclusive market arb: sum of asks < this = buy both
INTRA_MARKET_ARB_ASK_THRESHOLD: float = 0.98
# Sum of bids > this = sell both
INTRA_MARKET_ARB_BID_THRESHOLD: float = 1.02

# ---- Probability bounds (avoid near-zero / near-one markets) ---------------
MIN_TRADEABLE_PROB: float = 0.05          # 5¢ floor (avoid illiquid low-prob junk)
MAX_TRADEABLE_PROB: float = 0.95          # 95¢ ceiling

# ---- Data freshness --------------------------------------------------------
# Data freshness: must be > MARKET_REFRESH_INTERVAL (60s) to avoid
# every trade being blocked as stale between refresh cycles.
MAX_MARKET_SNAPSHOT_AGE_SEC: float = 120.0
MAX_ORACLE_ESTIMATE_AGE_SEC: float = 3600.0

# ---- UMA Oracle ------------------------------------------------------------
UMA_DISPUTE_PERIOD_HOURS: float = 48.0    # Standard UMA dispute window

# ---- Kelly ----------------------------------------------------------------
KELLY_MAX_FRACTION: float = 0.25          # Absolute cap (Quarter-Kelly)

# ---- Timing ----------------------------------------------------------------
WEBSOCKET_RECONNECT_DELAY_SEC: float = 5.0
HEARTBEAT_INTERVAL_SEC: float = 30.0
ORDER_POLLING_INTERVAL_SEC: float = 2.0

# ---- Fee / Edge protection -------------------------------------------------
FEE_EDGE_MAX_CONSUMPTION: float = 0.30    # fees may consume at most 30% of gross edge

# ---- Liquidity thresholds --------------------------------------------------
MIN_MARKET_VOLUME_24H_USD: float = 5_000.0     # Reject thin markets < $5k vol
SHARP_MONEY_TRADE_THRESHOLD_USD: float = 500.0  # single trade flagged as "sharp"
ORDER_FLOW_WINDOW_TRADES: int = 50             # trades to look back for OFI
ORDER_FLOW_IMBALANCE_THRESHOLD: float = 0.50   # |OFI_pct| > 50% = strong signal

# ---- Near-expiry -----------------------------------------------------------
NEAR_EXPIRY_HOURS: float = 48.0   # markets within this window scanned more aggressively
NEAR_EXPIRY_SCAN_INTERVAL_SEC: float = 60.0   # 1-minute re-scan for near-expiry markets

# ---- Calibration -----------------------------------------------------------
CALIBRATION_MIN_SAMPLES: int = 10             # need >= 10 resolved signals to trust calibration
CALIBRATION_SHRINKAGE_MIN: float = 0.30       # never shrink calibration factor below 30%

# ---- Correlation factor labels (used across risk module) -------------------
FACTOR_LABELS: dict[str, list[str]] = {
    "US_POLITICS_GOP": ["trump", "republican", "gop", "house", "senate"],
    "US_POLITICS_DEM": ["democrat", "biden", "harris", "dnc"],
    "FED_POLICY": ["fed", "fomc", "rate cut", "rate hike", "powell"],
    "CRYPTO": ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi"],
    "MACRO_RECESSION": ["recession", "gdp", "unemployment", "cpi", "inflation"],
}

# ---- Market filters (blacklist patterns for test/garbage markets) ----------
MARKET_TITLE_BLACKLIST_PATTERNS: list[str] = [
    "1+1",
    "quicksettle",
    "test market",
    "will 2+2",
    "will 1+1",
]

# ---- Position stop loss ---------------------------------------------------
POSITION_STOP_LOSS_PCT: float = 0.20        # exit if loss > 20% of position size
MAX_DAILY_LOSS_PCT: float = 0.05            # Halt trading if NAV drops > 5% in 24h
