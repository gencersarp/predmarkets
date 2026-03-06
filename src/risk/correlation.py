"""
Correlation Matrix & Factor Exposure Tracking.

Prediction markets cluster around underlying macro/political factors.
Positions in correlated markets add up to concentrated factor exposure.

Architecture:
  1. Factor Mapper: maps market titles/categories to a factor label vector
  2. Exposure Tracker: maintains current USD exposure per factor
  3. Correlation Checker: flags when new positions would breach factor caps
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from src.core.constants import FACTOR_LABELS
from src.core.models import Market, Position, Side

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor Mapper
# ---------------------------------------------------------------------------

class FactorMapper:
    """
    Maps a market to its underlying risk factors using keyword matching.
    Returns a dict: factor_label → exposure_weight (0.0 to 1.0)

    In production, replace with an embedding-based classifier (sentence-transformers
    + k-NN against a labelled factor taxonomy).
    """

    def map_market(self, market: Market) -> dict[str, float]:
        """Return factor exposures for a market (factor → weight)."""
        text = f"{market.title} {market.description} {market.category}".lower()
        exposures: dict[str, float] = {}

        for factor, keywords in FACTOR_LABELS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                # Weight proportional to keyword density, capped at 1.0
                weight = min(1.0, hits / max(len(keywords) * 0.5, 1.0))
                exposures[factor] = weight

        # Normalise so weights sum to 1 (avoid double-counting)
        total = sum(exposures.values())
        if total > 0:
            exposures = {k: v / total for k, v in exposures.items()}

        return exposures

    def build_factor_vector(
        self, market: Market, factor_labels: list[str]
    ) -> np.ndarray:
        """Return a numpy array of factor weights in the order of `factor_labels`."""
        exposures = self.map_market(market)
        return np.array([exposures.get(f, 0.0) for f in factor_labels])


# ---------------------------------------------------------------------------
# Correlation Matrix Builder
# ---------------------------------------------------------------------------

class CorrelationMatrix:
    """
    Builds and maintains a pairwise correlation matrix across open positions.

    For prediction markets, "correlation" ≈ shared factor exposure.
    Two positions are perfectly correlated if they load on the same factor.

    In production, supplement with:
    - Historical price correlations (rolling 30-day)
    - Canonical market pair correlations (e.g. state → national election)
    """

    def __init__(self) -> None:
        self._mapper = FactorMapper()
        self._factor_labels = list(FACTOR_LABELS.keys())

    def compute_pairwise_correlation(
        self, market_a: Market, market_b: Market
    ) -> float:
        """
        Cosine similarity between factor vectors as a proxy for correlation.
        Range: 0.0 (uncorrelated) to 1.0 (same factor).
        """
        vec_a = self._mapper.build_factor_vector(market_a, self._factor_labels)
        vec_b = self._mapper.build_factor_vector(market_b, self._factor_labels)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def build_matrix(self, markets: list[Market]) -> np.ndarray:
        """
        Build full n×n correlation matrix for a list of markets.
        Used for Markowitz-style portfolio optimisation.
        """
        n = len(markets)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.compute_pairwise_correlation(markets[i], markets[j])
                matrix[i, j] = corr
                matrix[j, i] = corr
        return matrix


# ---------------------------------------------------------------------------
# Factor Exposure Tracker
# ---------------------------------------------------------------------------

class FactorExposureTracker:
    """
    Tracks current USD exposure per risk factor across all open positions.
    Enforces limits before new positions can be opened.
    """

    def __init__(
        self,
        max_factor_exposure_pct: float = 0.40,
    ) -> None:
        """
        max_factor_exposure_pct: maximum fraction of NAV that can be
        concentrated in any single risk factor.
        """
        self._max_exposure_pct = max_factor_exposure_pct
        self._mapper = FactorMapper()
        self._factor_exposures: dict[str, float] = {}    # factor → USD
        self._position_factors: dict[str, dict[str, float]] = {}  # pos_id → {factor: usd}

    def add_position(self, position: Position, market: Market) -> None:
        """Record the factor exposure from a newly opened position."""
        factor_weights = self._mapper.map_market(market)
        pos_exposure: dict[str, float] = {}

        for factor, weight in factor_weights.items():
            usd = position.size_usd * weight
            pos_exposure[factor] = usd
            self._factor_exposures[factor] = self._factor_exposures.get(factor, 0.0) + usd

        self._position_factors[position.position_id] = pos_exposure

    def remove_position(self, position_id: str) -> None:
        """Remove a closed/resolved position's factor exposure."""
        if position_id not in self._position_factors:
            return
        for factor, usd in self._position_factors[position_id].items():
            self._factor_exposures[factor] = max(
                0.0, self._factor_exposures.get(factor, 0.0) - usd
            )
        del self._position_factors[position_id]

    def check_new_position(
        self,
        market: Market,
        size_usd: float,
        nav_usd: float,
    ) -> tuple[bool, str]:
        """
        Check whether opening a new position of `size_usd` in `market`
        would breach factor concentration limits.

        Returns: (is_allowed, reason_if_blocked)
        """
        factor_weights = self._mapper.map_market(market)
        for factor, weight in factor_weights.items():
            new_exposure = size_usd * weight
            current = self._factor_exposures.get(factor, 0.0)
            total_if_added = current + new_exposure
            if total_if_added / max(nav_usd, 1.0) > self._max_exposure_pct:
                return (
                    False,
                    f"Factor '{factor}' would exceed {self._max_exposure_pct:.0%} NAV: "
                    f"current=${current:.0f}, new=${new_exposure:.0f}, "
                    f"NAV=${nav_usd:.0f}",
                )
        return True, ""

    def get_exposures(self) -> dict[str, float]:
        """Return current factor exposures in USD."""
        return dict(self._factor_exposures)

    def get_exposure_pct(self, nav_usd: float) -> dict[str, float]:
        """Return factor exposures as fraction of NAV."""
        return {k: v / max(nav_usd, 1.0) for k, v in self._factor_exposures.items()}

    def largest_factor(self) -> Optional[tuple[str, float]]:
        """Return (factor, usd) for the largest single factor exposure."""
        if not self._factor_exposures:
            return None
        return max(self._factor_exposures.items(), key=lambda x: x[1])
