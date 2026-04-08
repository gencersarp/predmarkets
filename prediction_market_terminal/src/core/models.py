"""
Core Pydantic data models — the canonical representation of every entity
in the terminal. All exchange-specific data is normalised into these types.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Exchange(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    INTERNAL = "internal"   # synthetic / computed markets


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"
    FOK = "fok"
    GTC = "gtc"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(str, Enum):
    OPEN = "open"
    LOCKED = "locked"      # awaiting resolution
    RESOLVED_WIN = "resolved_win"
    RESOLVED_LOSS = "resolved_loss"
    DISPUTED = "disputed"  # UMA oracle dispute active


class ResolutionSource(str, Enum):
    UMA_ORACLE = "uma_oracle"
    KALSHI_INTERNAL = "kalshi_internal"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class AlphaType(str, Enum):
    CROSS_EXCHANGE_ARB = "cross_exchange_arb"
    INTRA_MARKET_ARB = "intra_market_arb"
    CONDITIONAL_ARB = "conditional_arb"
    EV_DIRECTIONAL = "ev_directional"
    TIME_DECAY = "time_decay"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ORDER_FLOW = "order_flow"
          # order-flow imbalance / sharp money


class RiskFlag(str, Enum):
    RESOLUTION_RISK = "resolution_risk"
    UMA_DISPUTE_RISK = "uma_dispute_risk"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_CORRELATION = "high_correlation"
    FEE_EXCESSIVE = "fee_excessive"
    AROC_BELOW_MIN = "aroc_below_min"
    CAPITAL_LOCK_RISK = "capital_lock_risk"
    SHARP_MONEY_AGAINST = "sharp_money_against"  # large informed trades opposing our side
    LOW_VOLUME = "low_volume"                     # 24h volume below liquidity threshold


# ---------------------------------------------------------------------------
# Market Snapshot — normalised across exchanges
# ---------------------------------------------------------------------------

class OrderBook(BaseModel):
    """Level-2 order book snapshot."""
    timestamp: datetime
    bids: list[tuple[float, float]] = Field(default_factory=list)  # (price, size)
    asks: list[tuple[float, float]] = Field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class MarketOutcome(BaseModel):
    """One binary outcome within a market (YES or NO leg)."""
    outcome_id: str
    side: Side
    implied_prob_bid: float = Field(ge=0.0, le=1.0)
    implied_prob_ask: float = Field(ge=0.0, le=1.0)
    order_book: Optional[OrderBook] = None
    volume_24h: float = 0.0
    open_interest: float = 0.0

    # AMM-specific (Polymarket)
    amm_token_address: Optional[str] = None
    amm_reserve_yes: Optional[float] = None
    amm_reserve_no: Optional[float] = None


class Market(BaseModel):
    """Normalised market representation — canonical across all exchanges."""
    market_id: str
    exchange: Exchange
    title: str
    description: str
    category: str = ""
    resolution_source: ResolutionSource = ResolutionSource.UNKNOWN
    resolution_criteria: str = ""
    expiry: Optional[datetime] = None

    outcomes: list[MarketOutcome] = Field(default_factory=list)

    # Fees (as fraction, e.g. 0.02 = 2%)
    taker_fee: float = 0.0
    maker_fee: float = 0.0

    # Metadata
    raw_data: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    @property
    def yes_outcome(self) -> Optional[MarketOutcome]:
        for o in self.outcomes:
            if o.side == Side.YES:
                return o
        return None

    @property
    def no_outcome(self) -> Optional[MarketOutcome]:
        for o in self.outcomes:
            if o.side == Side.NO:
                return o
        return None

    @property
    def implied_prob_yes_mid(self) -> Optional[float]:
        o = self.yes_outcome
        if o:
            return (o.implied_prob_bid + o.implied_prob_ask) / 2
        return None

    @property
    def days_to_expiry(self) -> Optional[float]:
        if self.expiry:
            delta = self.expiry - datetime.now(timezone.utc)
            return max(0.0, delta.total_seconds() / 86400)
        return None


# ---------------------------------------------------------------------------
# Alpha Signals
# ---------------------------------------------------------------------------

class ResolutionRiskAssessment(BaseModel):
    """Assess whether two markets have different resolution rules."""
    flagged: bool
    reason: str = ""
    risk_level: float = 0.0    # 0.0 = identical rules, 1.0 = completely different


class ArbitrageOpportunity(BaseModel):
    """A detected arbitrage opportunity."""
    opp_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alpha_type: AlphaType
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Markets involved
    market_ids: list[str]
    exchanges: list[Exchange]

    # Economics
    gross_edge_pct: float         # before fees / gas
    net_edge_pct: float           # after all costs
    gross_edge_usd: float
    net_edge_usd: float
    required_capital_usd: float
    fee_cost_usd: float
    gas_cost_usd: float = 0.0

    # Timing
    expiry: Optional[datetime] = None
    aroc_annual: float = 0.0      # annualised return on capital

    # Risk
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    resolution_risk: Optional[ResolutionRiskAssessment] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    # Execution plan
    legs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return (
            self.net_edge_pct > 0
            and RiskFlag.FEE_EXCESSIVE not in self.risk_flags
            and RiskFlag.AROC_BELOW_MIN not in self.risk_flags
        )


class DirectionalSignal(BaseModel):
    """A fundamental / directional alpha signal."""
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alpha_type: AlphaType
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    market_id: str
    exchange: Exchange
    side: Side

    # Probabilities
    true_probability: float = Field(ge=0.0, le=1.0)   # model estimate
    implied_probability: float = Field(ge=0.0, le=1.0) # market price
    edge: float                                         # true_prob - implied_prob

    # Payoff
    decimal_odds: float            # payout per unit bet (e.g., 2.5)
    kelly_fraction_suggested: float
    recommended_size_usd: float
    expected_value_usd: float

    # Timing
    expiry: Optional[datetime] = None
    aroc_annual: float = 0.0

    # Risk
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    oracle_sources: list[str] = Field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return (
            self.expected_value_usd > 0
            and self.edge >= 0.02
            and RiskFlag.FEE_EXCESSIVE not in self.risk_flags
            and RiskFlag.AROC_BELOW_MIN not in self.risk_flags
        )


# ---------------------------------------------------------------------------
# Orders & Positions
# ---------------------------------------------------------------------------

class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exchange: Exchange
    market_id: str
    side: Side
    order_side: OrderSide
    order_type: OrderType
    price: float
    size_usd: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size_usd: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exchange_order_id: Optional[str] = None
    is_paper: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exchange: Exchange
    market_id: str
    market_title: str
    side: Side
    size_usd: float
    entry_price: float
    current_price: float
    unrealised_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    expiry: Optional[datetime] = None
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    is_paper: bool = True
    signal_id: Optional[str] = None
    uma_dispute_deadline: Optional[datetime] = None

    @property
    def days_locked(self) -> Optional[float]:
        if self.expiry:
            delta = self.expiry - datetime.now(timezone.utc)
            return max(0.0, delta.total_seconds() / 86400)
        return None

    @property
    def cost_basis(self) -> float:
        return self.size_usd


# ---------------------------------------------------------------------------
# Portfolio Snapshot
# ---------------------------------------------------------------------------

class PortfolioSnapshot(BaseModel):
    snapshot_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_nav_usd: float
    available_capital_usd: float
    locked_capital_usd: float
    unrealised_pnl_usd: float
    realised_pnl_usd: float
    peak_nav_usd: float
    current_drawdown_pct: float
    positions: list[Position] = Field(default_factory=list)
    factor_exposures: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Oracle / Fundamental Data
# ---------------------------------------------------------------------------

class OracleEstimate(BaseModel):
    """External model estimate for a market outcome."""
    source: str
    market_id: str
    true_probability: float = Field(ge=0.0, le=1.0)
    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 1.0
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)


class NewsEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    headline: str
    source: str
    published_at: datetime
    url: str = ""
    sentiment_score: float = 0.0    # -1 to +1
    relevant_market_ids: list[str] = Field(default_factory=list)
    probability_impact: float = 0.0  # estimated change in prob
