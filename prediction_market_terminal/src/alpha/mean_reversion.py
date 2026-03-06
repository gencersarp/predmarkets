"""
Overreaction / Mean-Reversion Signal Generator.

Hypothesis:
  Retail participants heavily overreact to breaking news, causing
  probability spikes that are not supported by fundamental base rates.
  Once the volume surge subsides, prices revert.

Detection algorithm:
  1. Detect a probability spike: |ΔP| > threshold within a short window
  2. Monitor volume decay: volume_now < volume_peak * decay_threshold
  3. If base-rate model says spike is excessive: fade the move

Statistical approach:
  Model P_t as a mean-reverting Ornstein-Uhlenbeck process:
      dP_t = κ(μ - P_t)dt + σ dW_t
  where:
      μ = long-run mean (from oracle / historical average)
      κ = mean-reversion speed (calibrated empirically)
      σ = volatility

  The expected reversion magnitude at time t:
      E[P_t | P_0] = μ + (P_0 - μ) * exp(-κ * t)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.core.models import (
    AlphaType,
    DirectionalSignal,
    Market,
    RiskFlag,
    Side,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PriceHistory:
    """Rolling price history for a single market outcome."""
    timestamps: list[datetime] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    max_window: int = 500  # max data points to retain

    def append(self, timestamp: datetime, price: float, volume: float = 0.0) -> None:
        self.timestamps.append(timestamp)
        self.prices.append(price)
        self.volumes.append(volume)
        if len(self.prices) > self.max_window:
            self.timestamps = self.timestamps[-self.max_window:]
            self.prices = self.prices[-self.max_window:]
            self.volumes = self.volumes[-self.max_window:]

    def recent(self, minutes: int = 60) -> tuple[list[datetime], list[float], list[float]]:
        """Return data within the last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        result_t, result_p, result_v = [], [], []
        for t, p, v in zip(self.timestamps, self.prices, self.volumes):
            if t >= cutoff:
                result_t.append(t)
                result_p.append(p)
                result_v.append(v)
        return result_t, result_p, result_v

    def peak_price(self, minutes: int = 60) -> Optional[float]:
        _, prices, _ = self.recent(minutes)
        return max(prices) if prices else None

    def trough_price(self, minutes: int = 60) -> Optional[float]:
        _, prices, _ = self.recent(minutes)
        return min(prices) if prices else None

    def peak_volume(self, minutes: int = 60) -> Optional[float]:
        _, _, vols = self.recent(minutes)
        return max(vols) if vols else None

    def current_price(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None

    def current_volume(self) -> Optional[float]:
        return self.volumes[-1] if self.volumes else None

    def rolling_mean(self, minutes: int = 1440) -> Optional[float]:
        """Rolling mean over the last N minutes (default: 24h)."""
        _, prices, _ = self.recent(minutes)
        if not prices:
            return None
        return sum(prices) / len(prices)

    def rolling_std(self, minutes: int = 1440) -> Optional[float]:
        """Rolling standard deviation."""
        _, prices, _ = self.recent(minutes)
        if len(prices) < 2:
            return None
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / (len(prices) - 1)
        return math.sqrt(variance)


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""
    mu: float           # long-run mean
    kappa: float        # mean-reversion speed (per day)
    sigma: float        # volatility (per sqrt-day)

    def expected_price(self, current_price: float, days: float) -> float:
        """E[P(t)] = μ + (P(0) - μ) * exp(-κt)"""
        return self.mu + (current_price - self.mu) * math.exp(-self.kappa * days)

    def reversion_halflife_days(self) -> float:
        """Days until half the deviation from mean is expected to close."""
        if self.kappa <= 0:
            return float("inf")
        return math.log(2) / self.kappa


class OUCalibrator:
    """
    Calibrate OU parameters from historical price data using OLS regression.
    Regression: ΔP_t = α + β * P_{t-1} + ε
    where κ = -β/Δt, μ = α / (-β)
    """

    @staticmethod
    def calibrate(history: PriceHistory) -> Optional[OUParameters]:
        prices = history.prices
        n = len(prices)
        if n < 20:
            return None

        # Compute returns: ΔP_t = P_t - P_{t-1}
        delta = [prices[i] - prices[i - 1] for i in range(1, n)]
        lagged = prices[:-1]

        # OLS: ΔP = α + β * P_{t-1}
        n_reg = len(delta)
        sum_x = sum(lagged)
        sum_y = sum(delta)
        sum_xx = sum(x ** 2 for x in lagged)
        sum_xy = sum(x * y for x, y in zip(lagged, delta))

        denom = n_reg * sum_xx - sum_x ** 2
        if abs(denom) < 1e-10:
            return None

        beta = (n_reg * sum_xy - sum_x * sum_y) / denom
        alpha = (sum_y - beta * sum_x) / n_reg

        if beta >= 0:
            return None  # No mean reversion

        # Assume ticks are ~1 minute apart = 1/1440 days
        dt = 1.0 / 1440.0
        kappa = -beta / dt
        mu = alpha / (-beta)

        # Estimate sigma from residuals
        residuals = [y - (alpha + beta * x) for x, y in zip(lagged, delta)]
        var_resid = sum(r ** 2 for r in residuals) / max(n_reg - 2, 1)
        sigma = math.sqrt(max(0.0, var_resid) / dt)

        return OUParameters(mu=max(0.0, min(1.0, mu)), kappa=kappa, sigma=sigma)


class MeanReversionDetector:
    """
    Detects overreaction events and generates mean-reversion trade signals.
    """

    def __init__(
        self,
        spike_threshold: float = 0.10,         # price move > 10% in 30min = spike
        volume_decay_threshold: float = 0.30,   # volume falls to <30% of peak
        min_edge: float = 0.04,
    ) -> None:
        self._spike_threshold = spike_threshold
        self._volume_decay_threshold = volume_decay_threshold
        self._min_edge = min_edge
        self._price_histories: dict[str, PriceHistory] = {}
        self._ou_params: dict[str, OUParameters] = {}

    def update(self, market_id: str, price: float, volume: float) -> None:
        """Feed a new price/volume tick into the history buffer."""
        if market_id not in self._price_histories:
            self._price_histories[market_id] = PriceHistory()
        self._price_histories[market_id].append(datetime.now(timezone.utc), price, volume)

        # Recalibrate OU params every 100 ticks
        history = self._price_histories[market_id]
        if len(history.prices) % 100 == 0:
            params = OUCalibrator.calibrate(history)
            if params:
                self._ou_params[market_id] = params
                logger.debug(
                    "OU calibrated for %s: μ=%.3f κ=%.2f σ=%.4f",
                    market_id, params.mu, params.kappa, params.sigma,
                )

    def detect_overreaction(
        self, market: Market, oracle_true_prob: Optional[float] = None
    ) -> Optional[DirectionalSignal]:
        """
        Detect if the market has over-reacted and generate a fade signal.
        """
        market_id = market.market_id
        history = self._price_histories.get(market_id)
        if not history or len(history.prices) < 10:
            return None

        current = history.current_price()
        if current is None:
            return None

        # 1. Detect spike: large move in recent 30-min window
        _, recent_prices, recent_vols = history.recent(minutes=30)
        if len(recent_prices) < 3:
            return None

        price_30m_ago = recent_prices[0]
        price_move = current - price_30m_ago

        if abs(price_move) < self._spike_threshold:
            return None  # No significant spike

        # 2. Check volume decay
        peak_vol = history.peak_volume(minutes=30)
        current_vol = history.current_volume()
        if peak_vol and current_vol:
            vol_ratio = current_vol / max(peak_vol, 0.01)
            volume_decaying = vol_ratio < self._volume_decay_threshold
        else:
            volume_decaying = False

        # 3. Determine fair value
        if oracle_true_prob is not None:
            fair_value = oracle_true_prob
        elif market_id in self._ou_params:
            fair_value = self._ou_params[market_id].mu
        else:
            fair_value = history.rolling_mean(minutes=1440)

        if fair_value is None:
            return None

        # 4. Edge calculation
        edge = fair_value - current   # positive = current price is below fair value

        if abs(edge) < self._min_edge:
            return None

        # 5. Only fade if volume is decaying (confirmation of overreaction subsiding)
        if not volume_decaying and abs(price_move) < 0.20:
            logger.debug(
                "Spike detected for %s (move=%.2f) but volume not yet decaying",
                market_id, price_move,
            )
            return None

        settings = get_settings()

        if edge > 0:
            # Price spiked DOWN (overreaction to bad news): BUY YES
            side = Side.YES
            trade_price = market.yes_outcome.implied_prob_ask if market.yes_outcome else current
            true_prob_for_side = fair_value
            implied_for_side = current
        else:
            # Price spiked UP (overreaction to good news): BUY NO
            side = Side.NO
            yes = market.yes_outcome
            trade_price = (1.0 - yes.implied_prob_bid) if yes else (1.0 - current)
            true_prob_for_side = 1.0 - fair_value
            implied_for_side = 1.0 - current

        # OU-based expected reversion target
        ou = self._ou_params.get(market_id)
        if ou:
            expected_1d = ou.expected_price(current, days=1.0)
            expected_edge = abs(expected_1d - current)
        else:
            expected_edge = abs(edge) * 0.5  # assume 50% reversion

        from src.risk.kelly import kelly_fraction
        b = (1.0 / max(trade_price, 0.01)) - 1.0
        kf = kelly_fraction(
            p_win=true_prob_for_side,
            b=b,
            fraction=settings.kelly_fraction,
        )
        size_usd = min(settings.max_single_position_usd, settings.max_portfolio_exposure_usd * kf)
        ev_usd = size_usd * (true_prob_for_side * b - (1 - true_prob_for_side))

        risk_flags: list[RiskFlag] = []
        if not volume_decaying:
            risk_flags.append(RiskFlag.CAPITAL_LOCK_RISK)

        days = market.days_to_expiry or 7.0
        aroc = (ev_usd / max(size_usd, 1.0)) * (365.0 / max(days, 1.0))
        if aroc < settings.aroc_minimum_annual:
            risk_flags.append(RiskFlag.AROC_BELOW_MIN)

        confidence = 0.55 + (0.25 if volume_decaying else 0.0) + (0.10 if ou else 0.0)

        return DirectionalSignal(
            alpha_type=AlphaType.MEAN_REVERSION,
            market_id=market_id,
            exchange=market.exchange,
            side=side,
            true_probability=true_prob_for_side,
            implied_probability=implied_for_side,
            edge=abs(edge),
            decimal_odds=(1.0 / max(trade_price, 0.01)),
            kelly_fraction_suggested=kf,
            recommended_size_usd=size_usd,
            expected_value_usd=ev_usd,
            expiry=market.expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            confidence=min(0.90, confidence),
            oracle_sources=["ou_mean_reversion", "historical_rolling_mean"],
        )

    @property
    def tracked_markets(self) -> set[str]:
        return set(self._price_histories.keys())
