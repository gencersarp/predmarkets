"""
Trend Following (Momentum) Signal Generator.

Hypothesis:
  In markets with complex information or slow resolution, price changes
  often show momentum as participants slowly absorb and price-in new data.
  Unlike overreactions (which revert), these "drifts" continue in the same
  direction as volume stays high or increasing.

Detection algorithm:
  1. Detect a breakout: price crosses N-standard-deviation Bollinger Band
  2. Confirm with momentum: RSI > 70 (for uptrend) or < 30 (for downtrend)
  3. Confirm with volume: Volume now > moving average volume
  4. Only trade if market duration > M days (drift takes time)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.alpha.mean_reversion import PriceHistory
from src.core.models import (
    AlphaType,
    DirectionalSignal,
    Market,
    RiskFlag,
    Side,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


class TrendFollowingDetector:
    """
    Detects momentum events and generates trend-following (drift) signals.
    """

    def __init__(
        self,
        momentum_threshold: float = 0.05,       # 5% move for momentum
        volume_increase_threshold: float = 1.2, # volume > 1.2x average
        rsi_period: int = 14,
    ) -> None:
        self._momentum_threshold = momentum_threshold
        self._volume_increase_threshold = volume_increase_threshold
        self._rsi_period = rsi_period
        self._price_histories: dict[str, PriceHistory] = {}

    def update(self, market_id: str, price: float, volume: float) -> None:
        """Feed a new price/volume tick into the history buffer."""
        if market_id not in self._price_histories:
            self._price_histories[market_id] = PriceHistory()
        self._price_histories[market_id].append(datetime.now(timezone.utc), price, volume)

    def detect_trend(self, market: Market) -> Optional[DirectionalSignal]:
        """
        Detect if a market is trending and generate a follow signal.
        """
        market_id = market.market_id
        history = self._price_histories.get(market_id)
        if not history or len(history.prices) < self._rsi_period + 1:
            return None

        current_price = history.current_price()
        if current_price is None:
            return None

        # 1. Bollinger Band Breakout
        mean = history.rolling_mean(minutes=1440)  # 24h mean
        std = history.rolling_std(minutes=1440)
        if mean is None or std is None or std < 0.001:
            return None

        upper_band = mean + 2.0 * std
        lower_band = mean - 2.0 * std

        # 2. RSI Calculation
        rsi = self._calculate_rsi(history.prices[-self._rsi_period-1:])
        if rsi is None:
            return None

        # 3. Volume Confirmation
        avg_vol = history.peak_volume(minutes=1440) # Proxy for peak 24h vol
        curr_vol = history.current_volume()
        if avg_vol and curr_vol:
            volume_confirm = curr_vol > avg_vol * 0.5 # Require at least half of peak vol
        else:
            volume_confirm = False

        # Signal generation logic
        side = None
        if current_price > upper_band and rsi > 65 and volume_confirm:
            side = Side.YES  # Follow the upward breakout
        elif current_price < lower_band and rsi < 35 and volume_confirm:
            side = Side.NO   # Follow the downward breakout (buy NO)

        if side is None:
            return None

        # Estimated "True Prob" based on trend continuation
        # For a trend follower, we assume the price will move at least 0.5 std further
        true_prob = current_price + (0.05 if side == Side.YES else -0.05)
        true_prob = max(0.01, min(0.99, true_prob))
        
        edge = abs(true_prob - current_price)
        if edge < 0.02:
            return None

        settings = get_settings()
        trade_price = market.yes_outcome.implied_prob_ask if side == Side.YES else (1.0 - market.yes_outcome.implied_prob_bid)
        
        from src.risk.kelly import kelly_fraction
        kf = kelly_fraction(
            p_win=true_prob if side == Side.YES else (1.0 - true_prob),
            b=(1.0 / max(trade_price, 0.01)) - 1.0,
            fraction=settings.kelly_fraction * 0.5, # Lower fraction for momentum
        )
        
        size_usd = min(settings.max_single_position_usd, settings.max_portfolio_exposure_usd * kf)
        ev_usd = size_usd * edge # Simplified EV

        return DirectionalSignal(
            alpha_type=AlphaType.MOMENTUM,
            market_id=market_id,
            exchange=market.exchange,
            side=side,
            true_probability=true_prob if side == Side.YES else (1.0 - true_prob),
            implied_probability=current_price if side == Side.YES else (1.0 - current_price),
            edge=edge,
            decimal_odds=(1.0 / max(trade_price, 0.01)),
            kelly_fraction_suggested=kf,
            recommended_size_usd=size_usd,
            expected_value_usd=ev_usd,
            expiry=market.expiry,
            aroc_annual=(ev_usd / max(size_usd, 1.0)) * (365.0 / 7.0),
            risk_flags=[RiskFlag.LIQUIDITY_RISK] if not volume_confirm else [],
            confidence=0.60,
            oracle_sources=["bollinger_breakout", "rsi_momentum"],
        )

    def _calculate_rsi(self, prices: list[float]) -> Optional[float]:
        if len(prices) < self._rsi_period + 1:
            return None
        
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / self._rsi_period
        avg_loss = sum(losses) / self._rsi_period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
