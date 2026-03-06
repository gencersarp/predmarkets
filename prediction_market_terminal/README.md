# Prediction Market Terminal (PMT)

A professional-grade algorithmic trading terminal for Polymarket and Kalshi prediction markets. Detects arbitrage opportunities, sizes positions via Kelly Criterion, manages portfolio risk, and executes orders through a paper-first simulation layer before live deployment.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Paper Trading Quickstart](#paper-trading-quickstart)
5. [Live Trading](#live-trading)
6. [AWS Secrets Manager Setup](#aws-secrets-manager-setup)
7. [CLI Reference](#cli-reference)
8. [Strategy Reference](#strategy-reference)
9. [Risk Management](#risk-management)
10. [Mathematical Models](#mathematical-models)
11. [Running Tests](#running-tests)
12. [Directory Structure](#directory-structure)
13. [Runbook: Safe Capital Deployment](#runbook-safe-capital-deployment)

---

## Architecture Overview

```
config/                  # Pydantic-settings configuration
src/
  core/                  # Shared models, exceptions, constants
  data/
    feeds/               # Polymarket WebSocket + REST, Kalshi REST, Oracle feeds
    state.py             # Async in-memory market state (TerminalState)
  alpha/
    arbitrage.py         # Cross-exchange, intra-market, conditional arb detection
    fundamental.py       # EV-based directional signal generation
    time_decay.py        # Poisson time-decay theta harvesting
    mean_reversion.py    # Ornstein-Uhlenbeck mean-reversion detector
  risk/
    kelly.py             # Kelly Criterion sizing (quarter-Kelly default)
    guards.py            # Composable fail-safe risk guards
    portfolio.py         # Portfolio manager: NAV, drawdown, UMA dispute tracking
    correlation.py       # Factor exposure tracker, correlation matrix
  execution/
    base.py              # Abstract ExchangeAdapter interface
    paper.py             # Paper trading adapter (instant-fill simulation)
    router.py            # Order router: risk guards -> adapter dispatch
    polymarket.py        # Live Polymarket adapter (requires py-clob-client)
    kalshi.py            # Live Kalshi adapter (HMAC-authenticated REST)
  terminal/
    orchestrator.py      # Main async trading loop
    dashboard.py         # Rich terminal dashboard (Live layout)
    cli.py               # Click CLI entry points
  utils/
    database.py          # SQLAlchemy async + aiosqlite persistence
    secrets.py           # AWS Secrets Manager injection
    logging_config.py    # structlog setup
scripts/
  run_terminal.py        # Main entry point
  kelly_calc.py          # Standalone Kelly calculator
  backtest.py            # Post-hoc signal/order analysis
tests/
  unit/                  # 200+ unit tests
  integration/           # Async state integration tests
```

### Data Flow

```
Market Feeds (Polymarket WS / Kalshi REST)
          |
          v
    TerminalState (async in-memory store)
          |
    +-----+-----+
    |           |
    v           v
 ArbitrageScanner   FundamentalEVEngine
 TimeDecaySignals    MeanReversionDetector
    |           |
    +-----+-----+
          |
          v
    RiskGuardRunner (fail-fast composable checks)
          |
          v
    OrderRouter --> PaperAdapter (paper mode)
                --> PolymarketAdapter (live)
                --> KalshiAdapter (live)
          |
          v
    SQLite Audit Log + Rich Dashboard
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip or uv

### Install

```bash
git clone <repo>
cd prediction_market_terminal
pip install -e ".[dev]"
```

Or with uv (faster):

```bash
uv pip install -e ".[dev]"
```

### Dependencies installed

| Package | Purpose |
|---------|---------|
| `pydantic-settings` | Typed configuration from environment |
| `aiohttp` | Async HTTP + WebSocket feeds |
| `sqlalchemy[asyncio]` | Async ORM for audit persistence |
| `aiosqlite` | SQLite async driver |
| `rich` | Terminal dashboard |
| `click` | CLI framework |
| `structlog` | Structured logging |
| `numpy` | Numerical computation (OU calibration, Kelly) |
| `boto3` | AWS Secrets Manager (optional) |
| `pytest-asyncio` | Async test runner |

---

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Trading mode: "paper" (safe default) or "live"
PMT_MODE=paper

# Database (SQLite for local, PostgreSQL for production)
DATABASE_URL=sqlite+aiosqlite:///./pmt.db

# Kalshi REST API
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_PRIVATE_KEY=your_kalshi_private_key_pem

# Polymarket (only needed for live trading)
POLYMARKET_PRIVATE_KEY=your_polygon_wallet_private_key
POLYMARKET_API_KEY=your_clob_api_key
POLYMARKET_API_SECRET=your_clob_api_secret
POLYMARKET_API_PASSPHRASE=your_clob_passphrase

# Risk limits
MAX_SINGLE_POSITION_USD=150.0
MAX_DRAWDOWN_PCT=0.20
MAX_PORTFOLIO_CONCENTRATION=0.25
AROC_MINIMUM_ANNUAL=0.30

# Data sources (optional, enhances oracle signals)
NEWSAPI_KEY=your_newsapi_key
FRED_API_KEY=your_fred_key
```

### Key settings reference

| Setting | Default | Description |
|---------|---------|-------------|
| `PMT_MODE` | `paper` | `paper` or `live` |
| `MAX_SINGLE_POSITION_USD` | `150.0` | Hard cap per position |
| `MAX_DRAWDOWN_PCT` | `0.20` | Halt trading at 20% drawdown from peak NAV |
| `MAX_PORTFOLIO_CONCENTRATION` | `0.25` | Max fraction of NAV in a single factor |
| `AROC_MINIMUM_ANNUAL` | `0.30` | Reject opportunities with < 30% annualised AROC |
| `MIN_ARB_EDGE_PCT` | `0.02` | Minimum net edge to act on arb |
| `STALE_DATA_SECONDS` | `60` | Reject markets with data older than 60s |
| `GAS_PRICE_MAX_GWEI` | `150.0` | Abort if gas > 150 gwei |
| `MAX_SLIPPAGE_PCT` | `0.03` | Abort if estimated slippage > 3% |
| `FEE_MAX_CONSUMPTION_PCT` | `0.40` | Abort if fees > 40% of gross edge |

---

## Paper Trading Quickstart

Paper mode is the default. No real money is at risk. All orders are simulated with instant-fill logic respecting limit prices.

### Start the terminal

```bash
python scripts/run_terminal.py
```

This launches:
- The Rich live dashboard (refreshes every 2s)
- Market data feeds polling Polymarket and Kalshi
- Arbitrage scanner (runs every 10s)
- Signal generator (runs every 30s)
- Portfolio mark-to-market (runs every 15s)

### Dashboard layout

```
+------------------------------------------------------------------+
| PMT | NAV: $1,000.00 | P&L: +$0.00 | DD: 0.00% | Mode: PAPER   |
+------------------------------------------------------------------+
| ARBITRAGE OPPORTUNITIES          | DIRECTIONAL SIGNALS           |
| Edge  Capital  AROC   Exchange   | Edge  Size  AROC  Confidence  |
| 3.2%  $100    145%   KAL/POLY   | 8.1%  $95   120%  0.75       |
| 2.1%  $85     98%    POLY intra | 5.3%  $60   88%   0.72       |
+----------------------------------+-------------------------------+
| PORTFOLIO                                                        |
| Position    Size    Entry  Current  P&L    Expiry  Status       |
| Will X...   $95    0.52   0.55    +$5.7   14d    OPEN         |
+------------------------------------------------------------------+
```

### Kelly calculator

Use the standalone calculator to evaluate any trade before committing capital:

```bash
python scripts/kelly_calc.py --p-win 0.65 --price 0.50 --bankroll 1000
```

Interactive mode (no args):

```bash
python scripts/kelly_calc.py
```

Example output:
```
=======================================================
KELLY SIZING REPORT
=======================================================
  True probability (p_win)           65.00%
  Market price (implied prob)        50.00%
  Edge (true - implied)             +15.00%
  Decimal odds                        2.00x
  EV per unit staked                 +0.3000
  Breakeven probability              50.00%
  Required prob (after fees)         51.02%

  Full Kelly fraction                0.3000 (30.00%)
  Scaled Kelly fraction              0.0750 (7.50%)
  >>> RECOMMENDED BET SIZE           $75.00

  Log growth rate (G)                0.009065
  Ruin prob (100 bets, 50% loss)     2.34%

  Expected profit on this bet: $22.50

  [+] Signal looks tradeable.
=======================================================
```

### View historical signals and orders

```bash
python scripts/backtest.py
```

---

## Live Trading

**Read this entire section before enabling live mode.**

### Step 1: Paper trade for at least two weeks

Run paper mode continuously. Review:
- Drawdown: did it ever breach 20%?
- AROC: are realized opportunities achieving the projected AROC?
- Fill rate: are signals actionable by the time orders would execute?

```bash
python scripts/backtest.py  # review paper P&L
```

### Step 2: Configure credentials

Fill in all live credentials in `.env`. For Polymarket you need:
- A Polygon wallet private key with USDC deposited
- CLOB API credentials from https://docs.polymarket.com

For Kalshi:
- API key and RSA private key from https://kalshi.com/api

### Step 3: Enable live mode

```bash
PMT_MODE=live python scripts/run_terminal.py --live
```

The `--live` flag is required in addition to `PMT_MODE=live` as a double confirmation.

### Step 4: Start small

Set conservative limits for your first live session:

```bash
MAX_SINGLE_POSITION_USD=25.0 PMT_MODE=live python scripts/run_terminal.py --live
```

Increase limits only after verifying execution quality over multiple days.

---

## AWS Secrets Manager Setup

For production deployments, store credentials in AWS Secrets Manager rather than `.env` files.

### Create secrets

```bash
# Polymarket secret
aws secretsmanager create-secret \
  --name pmt/polymarket \
  --secret-string '{"private_key":"0x...","api_key":"...","api_secret":"...","api_passphrase":"..."}'

# Kalshi secret
aws secretsmanager create-secret \
  --name pmt/kalshi \
  --secret-string '{"api_key":"...","private_key":"-----BEGIN RSA..."}'

# Data API keys
aws secretsmanager create-secret \
  --name pmt/data \
  --secret-string '{"newsapi_key":"...","fred_api_key":"..."}'
```

### Configure PMT to use AWS

```bash
# .env (no sensitive values here)
PMT_MODE=live
AWS_REGION=us-east-1
AWS_SECRET_NAME_POLYMARKET=pmt/polymarket
AWS_SECRET_NAME_KALSHI=pmt/kalshi
AWS_SECRET_NAME_DATA=pmt/data
```

The terminal will automatically inject secrets from AWS on startup via `inject_aws_secrets()`.

### IAM permissions required

```json
{
  "Effect": "Allow",
  "Action": ["secretsmanager:GetSecretValue"],
  "Resource": [
    "arn:aws:secretsmanager:us-east-1:*:secret:pmt/*"
  ]
}
```

---

## CLI Reference

### `pmt run`

```bash
pmt run [--live] [--no-dash] [--debug]
```

| Flag | Description |
|------|-------------|
| `--live` | Enable live order execution (requires `PMT_MODE=live`) |
| `--no-dash` | Disable Rich dashboard (log-only mode, good for headless servers) |
| `--debug` | Enable DEBUG log level |

### `pmt status`

Print current portfolio snapshot (NAV, positions, drawdown) and exit.

```bash
pmt status
```

### `pmt kelly`

Interactive Kelly calculator (same as `scripts/kelly_calc.py`).

```bash
pmt kelly
```

---

## Strategy Reference

### 1. Cross-Exchange Arbitrage

Detects pricing discrepancies between Polymarket and Kalshi for equivalent markets.

**Signal**: The same real-world event trades at different prices on two exchanges.

**Entry condition**:
```
net_edge = (1 - ask_poly_yes - ask_kalshi_no) - fees
net_edge >= MIN_ARB_EDGE_PCT (default 2%)
```

**Execution**: Simultaneous buy on both legs. If either leg fails, the other is cancelled.

**Key risk**: Resolution risk — markets may resolve differently despite similar titles. The `ResolutionRiskAssessor` uses string similarity, resolution source comparison (UMA oracle vs Kalshi internal), and expiry delta to flag incompatible pairs.

### 2. Intra-Market Arbitrage

When `ask_yes + ask_no < 1.0` on a single market, buying both sides guarantees a profit.

**Entry condition**:
```
gap = 1.0 - (ask_yes + ask_no) - fees
gap > 0
```

**Note**: This opportunity is rare and disappears quickly. AMM slippage is simulated using the CPMM constant-product formula before committing capital.

### 3. Conditional Arbitrage

Exploits violations of the Fréchet bound: `P(A and B) <= min(P(A), P(B))`.

If two correlated markets are priced such that their joint probability implied by the CLOB exceeds the theoretical bound, a structured position can capture the mispricing.

### 4. EV Directional Signals

When a proprietary probability estimate (from oracle aggregation, news analysis, or base rate models) differs from the market-implied probability by more than the fee-adjusted breakeven:

```
edge = true_prob - implied_prob
edge >= required_edge_after_fees(market_price, taker_fee)
```

Position sized by quarter-Kelly on the estimated bankroll.

### 5. Poisson Time-Decay (Theta Harvesting)

For binary events with a known event rate `lambda`, the implied probability follows:

```
P(event by T | currently at t) = 1 - exp(-lambda * (T - t))
```

As time passes without the event occurring, the probability decays. If the market is slow to reprice, a mean-reverting short position captures the theta.

**Best for**: Weather events, economic data releases, sports outcomes with known timing.

### 6. Mean-Reversion (Ornstein-Uhlenbeck)

Calibrates an OU process to the market's price history:

```
dP_t = kappa * (mu - P_t) * dt + sigma * dW_t
```

Where `kappa` is mean-reversion speed, `mu` is long-run mean, `sigma` is volatility.

**Signal**: When price deviates more than `z_score_threshold` standard deviations from the OU mean, a reversion trade is opened.

**Best for**: Liquid markets with stable base rates (e.g., "Will the Fed raise rates?").

---

## Risk Management

### Risk Guards (fail-fast pipeline)

Every signal passes through a composable guard chain before execution. Any guard failure raises an exception that halts the trade:

| Guard | Check | Exception |
|-------|-------|-----------|
| `guard_paper_mode` | Live order in paper mode | `PaperModeViolation` |
| `guard_drawdown` | Portfolio drawdown > 20% | `DrawdownLimitBreached` |
| `guard_position_size` | Order > `MAX_SINGLE_POSITION_USD` | `PositionSizeTooLarge` |
| `guard_stale_data` | Market data > 60s old | `StaleDataError` |
| `guard_probability_bounds` | Price outside [0.03, 0.97] | `RiskLimitBreached` |
| `guard_gas_price` | Gas > 150 gwei | `GasLimitExceeded` |
| `guard_slippage` | Slippage > 3% | `SlippageLimitExceeded` |
| `guard_fee_consumption` | Fees > 40% of gross edge | `RiskLimitBreached` |
| `guard_aroc` | AROC < 30% annual | `RiskLimitBreached` |
| `guard_resolution_risk` | Resolution risk score > threshold | `RiskLimitBreached` |
| `guard_correlation` | Factor exposure > 25% of NAV | `CorrelationLimitBreached` |

### Portfolio-Level Controls

- **Drawdown halt**: Trading halts automatically when NAV drops 20% from its peak. Resume requires manual restart after reviewing causes.
- **Factor concentration**: Positions are tagged with political/macro/crypto factors. No single factor can exceed 25% of NAV.
- **UMA dispute buffer**: Polymarket winnings on UMA-resolved markets are held in `pending_cashflows` for 48 hours before being credited to NAV, preventing phantom profits during the dispute window.

### Guard preview (pre-trade dry run)

```python
from src.risk.guards import RiskGuardRunner

runner = RiskGuardRunner()
checks = runner.preview(signal, market, portfolio)
# Returns {"guard_drawdown": "PASS", "guard_position_size": "PASS", ...}
```

---

## Mathematical Models

### Kelly Criterion (Binary Markets)

For a binary market with price `p` (implied probability):

```
b = (1 / p) - 1          # net profit per dollar staked
q = 1 - p_win            # loss probability

f* = (b * p_win - q) / b # full Kelly fraction
f_scaled = f* * fraction  # quarter-Kelly: fraction=0.25
```

Position size: `size_usd = f_scaled * bankroll_usd`

**Why quarter-Kelly?** Full Kelly maximizes long-run log-wealth but requires perfect probability estimates. Quarter-Kelly is more robust to estimation error and reduces variance significantly at modest cost to growth rate.

### Portfolio Kelly Adjustment

When adding a position correlated with existing holdings:

```
existing_exposure_fraction = existing_correlated_usd / bankroll_usd
reduction = (existing_exposure_fraction / max_factor_exposure) * correlation
adjusted_kelly = base_kelly * (1 - reduction)
```

This prevents over-concentration in correlated bets (e.g., multiple Democratic-outcome markets during an election).

### AROC (Annualised Return on Capital)

```
aroc = (net_edge_usd / required_capital_usd) * (365 / days_to_expiry)
```

A 3% edge on a $100 position expiring in 14 days:

```
aroc = (3 / 100) * (365 / 14) = 0.782 = 78.2% annual
```

The 30% annual AROC minimum rejects opportunities with too long a time horizon relative to their edge.

### AMM Slippage (Polymarket CPMM)

Polymarket uses a Constant Product Market Maker:

```
k = R_yes * R_no           # invariant
spot_price = R_no / (R_yes + R_no)

# After buying delta_yes tokens:
R_yes_new = R_yes + delta_yes
R_no_new = k / R_yes_new
cost = R_no - R_no_new     # USDC cost

slippage = (cost / delta_yes - spot_price) / spot_price
```

Slippage is computed before any order to verify it stays within the 3% limit.

### Poisson Time-Decay

```
lambda = -ln(1 - p_base) / T_total   # calibrated from base rate

P(t) = 1 - exp(-lambda * (T - t))    # current fair value

theta = dP/dt = lambda * exp(-lambda * (T - t))  # daily decay rate
```

A market priced above the Poisson fair value is a sell candidate; below is a buy candidate.

### Ornstein-Uhlenbeck Calibration

Parameters estimated via OLS on first-differenced price series:

```
delta_P[t] = kappa * (mu - P[t-1]) * dt + epsilon[t]

OLS: delta_P ~ alpha + beta * P[t-1]
  -> beta = -kappa * dt
  -> alpha = kappa * mu * dt
  -> sigma = std(epsilon) / sqrt(dt)
```

Z-score for a current price `P`:
```
z = (P - mu) / (sigma / sqrt(2 * kappa))
```

Signal fires when `|z| > z_threshold` (default: 2.0).

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific module
pytest tests/unit/test_guards.py -v
```

All 225 tests should pass in under 1 second.

### Test structure

```
tests/
  conftest.py              # make_market(), make_arb_opportunity(), make_directional_signal()
  unit/
    test_kelly.py          # 32 tests: Kelly fractions, sizing, growth, ruin probability
    test_arbitrage.py      # 21 tests: cross-exchange, intra-market, conditional arb
    test_time_decay.py     # 19 tests: Poisson model, calibration, signal generation
    test_fundamental.py    # 21 tests: EV engine, oracle aggregation, signal filtering
    test_guards.py         # 26 tests: all risk guards, RiskGuardRunner
    test_portfolio.py      # 24 tests: NAV tracking, drawdown, UMA dispute, AROC report
    test_correlation.py    # 22 tests: factor mapping, correlation matrix, exposure tracker
    test_mean_reversion.py # 18 tests: OU calibration, detector, z-score signals
    test_paper_execution.py # 14 tests: fill simulation, IOC/GTC/FOK, balance tracking
  integration/
    test_state.py          # 14 tests: async TerminalState CRUD
```

---

## Directory Structure

```
prediction_market_terminal/
├── .env.example               # Environment variable template
├── pyproject.toml             # Project metadata + dependencies
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py            # Pydantic-settings; get_settings()
├── scripts/
│   ├── run_terminal.py        # Main entry point
│   ├── kelly_calc.py          # Standalone Kelly calculator
│   └── backtest.py            # Historical signal analysis
├── src/
│   ├── core/
│   │   ├── constants.py       # KELLY_MAX_FRACTION, FEE_EDGE_MAX_CONSUMPTION, etc.
│   │   ├── exceptions.py      # All custom exceptions
│   │   └── models.py          # Market, Order, Position, Signal, ArbitrageOpportunity
│   ├── alpha/
│   │   ├── arbitrage.py       # ArbitrageScanner + ResolutionRiskAssessor
│   │   ├── fundamental.py     # FundamentalEVEngine + OracleAggregator
│   │   ├── mean_reversion.py  # MeanReversionDetector + OUCalibrator
│   │   └── time_decay.py      # PoissonDecayModel + TimeDecaySignalGenerator
│   ├── data/
│   │   ├── feeds/
│   │   │   ├── kalshi.py      # Kalshi REST feed (HMAC auth)
│   │   │   ├── oracles.py     # NewsAPI, RSS, FRED, whale activity
│   │   │   └── polymarket.py  # Polymarket WebSocket + AMM math
│   │   └── state.py           # TerminalState (async in-memory store)
│   ├── execution/
│   │   ├── base.py            # ExchangeAdapter abstract interface
│   │   ├── kalshi.py          # Live Kalshi adapter
│   │   ├── paper.py           # Paper trading adapter
│   │   ├── polymarket.py      # Live Polymarket adapter
│   │   └── router.py          # OrderRouter (guards -> adapter dispatch)
│   ├── risk/
│   │   ├── correlation.py     # FactorMapper + CorrelationMatrix
│   │   ├── guards.py          # RiskGuardRunner + all guards
│   │   ├── kelly.py           # Kelly sizing functions
│   │   └── portfolio.py       # PortfolioManager
│   ├── terminal/
│   │   ├── cli.py             # Click CLI
│   │   ├── dashboard.py       # Rich Live dashboard
│   │   └── orchestrator.py    # TradingOrchestrator async loop
│   └── utils/
│       ├── database.py        # SQLAlchemy async + aiosqlite
│       ├── logging_config.py  # structlog setup
│       └── secrets.py         # AWS Secrets Manager injection
└── tests/
    ├── conftest.py
    ├── integration/
    └── unit/
```

---

## Runbook: Safe Capital Deployment

This runbook describes how to progress from first install to live capital deployment safely.

### Phase 1: Environment verification (Day 1)

```bash
# 1. Install and verify tests pass
pip install -e ".[dev]"
pytest

# 2. Verify configuration loads
python -c "from config.settings import get_settings; s = get_settings(); print('Mode:', s.pmt_mode)"

# 3. Run Kelly calculator to verify math
python scripts/kelly_calc.py --p-win 0.60 --price 0.50 --bankroll 1000
```

Expected: 225 tests pass, mode is "paper", Kelly output shows ~$20-30 recommended.

### Phase 2: Paper trading observation (Days 2-14)

```bash
# Start terminal in paper mode
python scripts/run_terminal.py

# In a second terminal, monitor the database
python scripts/backtest.py
```

**Checkpoints at end of paper period:**
- [ ] At least 10 arb opportunities detected and simulated
- [ ] At least 20 directional signals generated
- [ ] No drawdown > 5% in paper NAV
- [ ] AROC of realized opportunities aligns with projected (within 20%)
- [ ] No repeated `StaleDataError` (if so, investigate feed connectivity)
- [ ] No repeated `GasLimitExceeded` (if so, adjust `GAS_PRICE_MAX_GWEI`)

### Phase 3: Live connectivity test (Day 14)

Without enabling live execution, verify API credentials work:

```bash
# Test Kalshi auth
python -c "
import asyncio
from src.data.feeds.kalshi import KalshiFeed
from config.settings import get_settings
s = get_settings()
feed = KalshiFeed(api_key=s.kalshi_api_key.get_secret_value(),
                  private_key=s.kalshi_private_key.get_secret_value())
asyncio.run(feed.fetch_markets())
print('Kalshi OK')
"
```

### Phase 4: Micro-capital live test (Days 15-21)

Start with very small position limits:

```bash
MAX_SINGLE_POSITION_USD=10.0 PMT_MODE=live python scripts/run_terminal.py --live
```

**Checkpoints:**
- [ ] Orders reach the exchange (verify in Kalshi/Polymarket dashboard)
- [ ] Fill prices match paper simulation within 1%
- [ ] No `PaperModeViolation` exceptions (would indicate a code bug)
- [ ] SQLite audit log records every order

### Phase 5: Scale up (Week 4+)

Increase position limits gradually, doubling no faster than weekly:

```
Week 4: MAX_SINGLE_POSITION_USD=25.0
Week 5: MAX_SINGLE_POSITION_USD=50.0
Week 6: MAX_SINGLE_POSITION_USD=100.0
Week 7: MAX_SINGLE_POSITION_USD=150.0 (default maximum)
```

**Never skip position limit verification between increases.**

### Emergency procedures

**Halt all trading immediately:**
```bash
# Kill the terminal process
pkill -f run_terminal.py

# Or set drawdown limit to 0 to force halt on next cycle
MAX_DRAWDOWN_PCT=0.0 python scripts/run_terminal.py
```

**Cancel all open orders (paper):**
```python
import asyncio
from src.execution.paper import PaperExchangeAdapter
from src.core.models import Exchange

async def cancel_all():
    adapter = PaperExchangeAdapter(Exchange.POLYMARKET)
    orders = await adapter.get_open_orders()
    for o in orders:
        await adapter.cancel_order(o.order_id)
    print(f"Cancelled {len(orders)} orders")

asyncio.run(cancel_all())
```

**Review audit log:**
```bash
sqlite3 pmt.db "SELECT * FROM paper_orders ORDER BY created_at DESC LIMIT 20;"
sqlite3 pmt.db "SELECT * FROM arb_log ORDER BY detected_at DESC LIMIT 10;"
sqlite3 pmt.db "SELECT * FROM signal_log ORDER BY generated_at DESC LIMIT 10;"
```

---

## Known Limitations

1. **Polymarket live signing**: The live Polymarket adapter requires `py-clob-client` for EIP-712 order signing. Install separately and implement `PolymarketAdapter.place_order` with the client library.

2. **Whale tracking**: `fetch_polymarket_whale_activity` returns an empty list. Production implementation requires Alchemy or Dune Analytics API access.

3. **Market matching**: Cross-exchange arb uses O(n²) string similarity (`SequenceMatcher`). For universes > 500 markets, replace with sentence-transformer embeddings for speed and accuracy.

4. **No partial fills**: The paper adapter uses an instant-fill model with no queue priority simulation. Actual fill rates on GTC limit orders may be lower than simulated.

5. **Kalshi rate limits**: Kalshi's REST API has rate limits. The feed polls on configurable intervals; reduce `KALSHI_POLL_INTERVAL_SECONDS` with caution.

---

## License

For personal and research use. Ensure compliance with Polymarket and Kalshi terms of service before deploying with live capital.
