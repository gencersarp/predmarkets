"""Custom exception hierarchy for the trading terminal."""
from __future__ import annotations


class PMTError(Exception):
    """Base exception for all terminal errors."""


# ---- Data / Feed Errors ---------------------------------------------------

class FeedError(PMTError):
    """Error fetching or parsing market data."""


class StaleDataError(FeedError):
    """Market snapshot is too old to trade on."""
    def __init__(self, market_id: str, age_seconds: float) -> None:
        super().__init__(f"Stale data for {market_id}: {age_seconds:.1f}s old")
        self.market_id = market_id
        self.age_seconds = age_seconds


class MarketNotFoundError(FeedError):
    def __init__(self, market_id: str) -> None:
        super().__init__(f"Market not found: {market_id}")
        self.market_id = market_id


# ---- Alpha / Strategy Errors -----------------------------------------------

class AlphaError(PMTError):
    """Error in signal generation."""


class InsufficientLiquidityError(AlphaError):
    def __init__(self, market_id: str, required: float, available: float) -> None:
        super().__init__(
            f"Insufficient liquidity in {market_id}: "
            f"required=${required:.2f}, available=${available:.2f}"
        )


# ---- Risk Errors -----------------------------------------------------------

class RiskLimitBreached(PMTError):
    """A risk limit has been hit — execution must be blocked."""
    def __init__(self, limit_name: str, current: float, limit: float) -> None:
        super().__init__(
            f"Risk limit '{limit_name}' breached: "
            f"current={current:.4f}, limit={limit:.4f}"
        )
        self.limit_name = limit_name
        self.current = current
        self.limit = limit


class DrawdownLimitBreached(RiskLimitBreached):
    """Max drawdown circuit breaker triggered."""


class CorrelationLimitBreached(RiskLimitBreached):
    """Factor correlation cap hit."""


class PositionSizeTooLarge(RiskLimitBreached):
    """Single position limit exceeded."""


# ---- Execution Errors ------------------------------------------------------

class ExecutionError(PMTError):
    """Error during order placement or management."""


class OrderRejected(ExecutionError):
    def __init__(self, reason: str, order_id: str = "") -> None:
        super().__init__(f"Order rejected [{order_id}]: {reason}")
        self.reason = reason
        self.order_id = order_id


class PaperModeViolation(ExecutionError):
    """Attempt to sign/broadcast transaction while in paper mode."""
    def __init__(self) -> None:
        super().__init__(
            "Attempted real transaction in PAPER mode. "
            "Set PMT_MODE=live to enable live trading."
        )


class GasLimitExceeded(ExecutionError):
    def __init__(self, current_gwei: float, max_gwei: float) -> None:
        super().__init__(f"Gas {current_gwei:.1f} gwei > max {max_gwei:.1f} gwei")


class SlippageLimitExceeded(ExecutionError):
    def __init__(self, actual: float, limit: float) -> None:
        super().__init__(f"Slippage {actual:.4f} > tolerance {limit:.4f}")


# ---- Configuration Errors --------------------------------------------------

class ConfigurationError(PMTError):
    """Missing or invalid configuration."""


class SecretsError(ConfigurationError):
    """Failed to retrieve secrets."""
