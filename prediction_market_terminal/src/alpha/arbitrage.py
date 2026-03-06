"""
Arbitrage Detection Engine.

Implements four structural arb types:

1. Cross-Exchange Arb:    Same event on Kalshi + Polymarket (buy YES-A, NO-B)
2. Complement Pair Arb:   Mutually exclusive events — buy YES-A + YES-B < $1
3. Intra-Market Arb:      Sum(P_ask) < 1 OR Sum(P_bid) > 1 within one market
4. Conditional Arb:       Logical impossibilities across related markets

Complement pair arb (the main cross-site inefficiency):
  Market A: "Will Trump win?" (Poly)  ask=0.55
  Market B: "Will Harris win?" (Kalshi) ask=0.38
  Cost = 0.55 + 0.38 = 0.93 → Profit = $0.07 per $1 when one resolves

Near-complement arb with risk:
  A and B may not be perfect complements (a third outcome exists).
  E[payout] = P(exactly one resolves YES) = 1 - P(both NO) - P(both YES)
  Risk flag raised when P(neither) is non-trivial (implied by price sum).

Key mathematical invariants:
  - For a binary market: P(YES) + P(NO) = 1
  - True complements:    P(YES-A) + P(YES-B) = 1
  - Cross-market arb:    if ask(YES, A) + ask(NO, B) < 1 — buy both
  - Complement arb:      if ask(YES, A) + ask(YES, B) < 1 — buy both YES
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from src.core.constants import (
    FEE_EDGE_MAX_CONSUMPTION,
    INTRA_MARKET_ARB_ASK_THRESHOLD,
    INTRA_MARKET_ARB_BID_THRESHOLD,
    MIN_NET_EDGE_PCT,
    POLYGON_GAS_ESTIMATE_USD,
)
from src.core.models import (
    AlphaType,
    ArbitrageOpportunity,
    Exchange,
    Market,
    ResolutionRiskAssessment,
    ResolutionSource,
    RiskFlag,
    Side,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Stopwords removed before token similarity comparison.
# Only pure function words — DO NOT add content words (election, presidential, etc.)
# because those are meaningful discriminators between market topics.
_STOPWORDS = frozenset([
    "will", "the", "a", "an", "in", "of", "for", "is", "are", "be",
    "to", "at", "by", "or", "and", "on", "this", "that", "with",
    "it", "its", "their", "they", "if", "when", "who", "what",
])

# Phrases that strongly signal a negated / opposite outcome
_NEGATION_PATTERNS = re.compile(
    r"\b(not|no|won'?t|doesn'?t|fail|fails|lose|loses|below|under|fall|drop|against|defeat)\b",
    re.IGNORECASE,
)


def _normalise_tokens(title: str) -> frozenset[str]:
    """Lowercase, strip punctuation, remove stopwords → frozenset of tokens."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def _token_overlap(title_a: str, title_b: str) -> float:
    """
    Jaccard similarity on normalised token sets.
    Much better than SequenceMatcher for cross-exchange title pairs
    where word order and phrasing differ significantly.
    Returns 0.0–1.0.
    """
    ta = _normalise_tokens(title_a)
    tb = _normalise_tokens(title_b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


def _title_match_score(title_a: str, title_b: str) -> float:
    """
    Combined score: max(token-overlap Jaccard, SequenceMatcher ratio).
    Using the max lets either approach succeed.
    """
    return max(
        _token_overlap(title_a, title_b),
        SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio(),
    )


def _has_negation(title: str) -> bool:
    """Return True if the title contains a negation or opposite-outcome word."""
    return bool(_NEGATION_PATTERNS.search(title))


def _are_complement_titles(title_a: str, title_b: str) -> bool:
    """
    Heuristic: do these two titles describe mutually exclusive outcomes?

    Two cases:
      1. Same subject, one negated (e.g. "Will X win?" / "Will X lose?")
      2. Same competition, different winners (e.g. "Trump wins" / "Harris wins")

    We detect this by:
      - Significant keyword overlap on the non-negation tokens
      - Exactly one of the two has a negation word, OR
      - Both are positive but share a competition keyword (election, match,
        race, championship, cup, series, title, super bowl, world cup, …)
    """
    ta = _normalise_tokens(title_a)
    tb = _normalise_tokens(title_b)
    if not ta or not tb:
        return False
    shared = ta & tb
    if not shared:
        return False
    # Need at least some shared context (shared / smaller set ≥ 40%)
    overlap = len(shared) / min(len(ta), len(tb))
    if overlap < 0.30:
        return False
    # Case 1: one title negated, the other positive
    if _has_negation(title_a) != _has_negation(title_b):
        return True
    # Case 2: competition keywords present — different participants
    competition_words = frozenset([
        "election", "race", "match", "championship", "final", "cup",
        "series", "superbowl", "super", "bowl", "title", "primary",
        "runoff", "debate", "vote", "referendum",
    ])
    if shared & competition_words:
        return True
    return False


# ---------------------------------------------------------------------------
# Resolution Risk Assessor
# ---------------------------------------------------------------------------

class ResolutionRiskAssessor:
    """
    Flag pseudo-arbs where two markets describe the same event but resolve
    differently (e.g. UMA oracle vs Kalshi internal).

    Risk levels:
        0.0  = identical resolution mechanism and rules
        0.3  = same source, slight wording differences
        0.7  = different resolution mechanisms
        1.0  = fundamentally incompatible resolution criteria
    """

    def assess(self, market_a: Market, market_b: Market) -> ResolutionRiskAssessment:
        flags = []
        risk = 0.0

        # Different resolution mechanisms
        if market_a.resolution_source != market_b.resolution_source:
            risk += 0.4
            flags.append(
                f"Different resolution sources: "
                f"{market_a.resolution_source.value} vs "
                f"{market_b.resolution_source.value}"
            )

        # UMA oracle has a dispute period — capital could be locked longer
        uma_involved = (
            market_a.resolution_source == ResolutionSource.UMA_ORACLE
            or market_b.resolution_source == ResolutionSource.UMA_ORACLE
        )
        if uma_involved:
            risk += 0.2
            flags.append("UMA dispute period risk: capital locked up to 48h post-event")

        # Compare resolution criteria text similarity
        sim = SequenceMatcher(
            None,
            market_a.resolution_criteria.lower(),
            market_b.resolution_criteria.lower(),
        ).ratio()
        if sim < 0.5:
            risk += 0.3
            flags.append(
                f"Low resolution criteria similarity ({sim:.0%}): "
                "markets may resolve differently on identical outcomes"
            )
        elif sim < 0.8:
            risk += 0.1
            flags.append(f"Moderate criteria similarity ({sim:.0%}): verify resolution rules")

        # Different expiry dates → temporal mismatch
        if market_a.expiry and market_b.expiry:
            diff_days = abs((market_a.expiry - market_b.expiry).total_seconds()) / 86400
            if diff_days > 7:
                risk += 0.2
                flags.append(
                    f"Expiry mismatch: {diff_days:.0f} days apart — "
                    "positions may not resolve simultaneously"
                )

        risk = min(1.0, risk)
        return ResolutionRiskAssessment(
            flagged=risk > 0.1,
            reason="; ".join(flags) if flags else "No resolution risk detected",
            risk_level=risk,
        )


# ---------------------------------------------------------------------------
# Intra-Market Arbitrage (single exchange, mutually exclusive outcomes)
# ---------------------------------------------------------------------------

def detect_intra_market_arb(
    market: Market,
    min_capital_usd: float = 10.0,
    max_capital_usd: float = 500.0,
) -> Optional[ArbitrageOpportunity]:
    """
    For a binary market, detect when the ask-side book allows:
        ask(YES) + ask(NO) < 1  →  buy BOTH sides for guaranteed profit

    Or the bid-side allows:
        bid(YES) + bid(NO) > 1  →  sell BOTH sides for guaranteed profit

    Net edge after fees:
        gross_edge = 1 - (ask_yes + ask_no)
        fee_cost   = size * (taker_fee_yes + taker_fee_no)
        net_edge   = gross_edge - fee_cost / size
    """
    yes = market.yes_outcome
    no = market.no_outcome
    if not yes or not no:
        return None

    settings = get_settings()

    # --- Buy-both scenario (ask sum < 1)
    ask_yes = yes.implied_prob_ask
    ask_no = no.implied_prob_ask
    ask_sum = ask_yes + ask_no

    if ask_sum < INTRA_MARKET_ARB_ASK_THRESHOLD:
        gross_edge_per_unit = 1.0 - ask_sum
        # Cost per dollar of payout
        cost_per_payout = ask_sum  # spend ask_yes + ask_no, receive $1
        # Fee: taker fee applied to each leg
        total_fee_rate = market.taker_fee * 2
        # Net edge as fraction of required capital
        net_edge_pct = gross_edge_per_unit - total_fee_rate
        if net_edge_pct <= MIN_NET_EDGE_PCT:
            return None

        # Size: invest enough that net profit exceeds gas cost
        optimal_size_usd = min(max_capital_usd, max(min_capital_usd, 50.0))
        fee_cost_usd = optimal_size_usd * total_fee_rate
        gas_cost_usd = POLYGON_GAS_ESTIMATE_USD if market.exchange == Exchange.POLYMARKET else 0.0
        net_edge_usd = optimal_size_usd * net_edge_pct - gas_cost_usd

        if net_edge_usd <= 0:
            return None

        # Fee consumption check
        gross_edge_usd = optimal_size_usd * gross_edge_per_unit
        if (fee_cost_usd + gas_cost_usd) / gross_edge_usd > settings.fee_edge_max_consumption:
            risk_flags = [RiskFlag.FEE_EXCESSIVE]
        else:
            risk_flags = []

        # AROC
        days = market.days_to_expiry or 30.0
        aroc = (net_edge_usd / optimal_size_usd) * (365.0 / max(days, 1.0))
        if aroc < settings.aroc_minimum_annual:
            risk_flags.append(RiskFlag.AROC_BELOW_MIN)

        return ArbitrageOpportunity(
            alpha_type=AlphaType.INTRA_MARKET_ARB,
            market_ids=[market.market_id, market.market_id],
            exchanges=[market.exchange, market.exchange],
            gross_edge_pct=gross_edge_per_unit,
            net_edge_pct=net_edge_pct,
            gross_edge_usd=gross_edge_usd,
            net_edge_usd=net_edge_usd,
            required_capital_usd=optimal_size_usd,
            fee_cost_usd=fee_cost_usd,
            gas_cost_usd=gas_cost_usd,
            expiry=market.expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            confidence=0.95,  # structural arb — high confidence
            legs=[
                {
                    "side": Side.YES.value,
                    "action": "buy",
                    "price": ask_yes,
                    "size_usd": optimal_size_usd / 2,
                    "market_id": market.market_id,
                    "exchange": market.exchange.value,
                },
                {
                    "side": Side.NO.value,
                    "action": "buy",
                    "price": ask_no,
                    "size_usd": optimal_size_usd / 2,
                    "market_id": market.market_id,
                    "exchange": market.exchange.value,
                },
            ],
        )

    return None


# ---------------------------------------------------------------------------
# Complement-Pair Arbitrage
# ---------------------------------------------------------------------------

def detect_complement_arb(
    market_a: Market,
    market_b: Market,
    min_keyword_overlap: float = 0.25,
    max_capital_usd: float = 300.0,
) -> Optional[ArbitrageOpportunity]:
    """
    Detect complement-pair arbitrage: buy YES on BOTH markets where
    the two markets describe mutually exclusive outcomes.

    Example:
        Market A (Poly): "Will Trump win the 2024 election?"  ask(YES) = 0.55
        Market B (Kalshi): "Will Harris win the 2024 election?" ask(YES) = 0.38
        Cost = 0.93 < $1.00  →  guaranteed $0.07 profit (one must win)

    Near-complement risk:
        If the events are not perfectly exclusive (e.g., a third candidate
        could win), P(neither) > 0 and expected payout < $1.
        E[payout] = 1 - P(both lose) = (ask_a + ask_b) prices imply a spread.

        We estimate P(neither wins) = max(0, 1 - bid(YES-A) - bid(YES-B))
        and reduce confidence when this is > 5%.

    Resolution risk:
        Both markets must resolve consistently — same event, same date.
        The ResolutionRiskAssessor flags discrepancies.
    """
    if market_a.exchange == market_b.exchange:
        return None

    yes_a = market_a.yes_outcome
    yes_b = market_b.yes_outcome
    if not yes_a or not yes_b:
        return None

    # Title must share meaningful keywords (same competition / subject)
    overlap = _token_overlap(market_a.title, market_b.title)
    complement = _are_complement_titles(market_a.title, market_b.title)

    # Accept if overlap ≥ threshold OR explicit complement detection passes
    if overlap < min_keyword_overlap and not complement:
        return None

    # The arb: cost of buying YES on both sides
    ask_a = yes_a.implied_prob_ask
    ask_b = yes_b.implied_prob_ask
    cost = ask_a + ask_b

    if cost >= 1.0:
        return None  # No complement arb

    gross_edge = 1.0 - cost

    # Near-complement risk: estimate P(neither resolves YES) from mid prices.
    # For a true complement (one MUST win), P(neither) = 0 and E[payout] = $1.
    # For a near-complement (e.g., small 3rd-candidate probability), P(neither) > 0.
    # We estimate this from mid prices — not from bids (which just reflect the spread).
    mid_a = (yes_a.implied_prob_bid + yes_a.implied_prob_ask) / 2.0
    mid_b = (yes_b.implied_prob_bid + yes_b.implied_prob_ask) / 2.0
    # If mid prices sum to > 1, they can't both be complements — divergence risk.
    # If they sum to < 1, P(neither) is implied by the gap.
    p_neither = max(0.0, 1.0 - mid_a - mid_b)

    # For confirmed complement titles, E[payout] = $1 (one MUST resolve YES).
    # P(neither) becomes a resolution risk flag, not a mathematical edge deduction.
    # We only reduce confidence and flag risk — we do NOT deduct from gross_edge
    # because that would double-count with the resolution risk assessment.
    effective_edge = gross_edge  # = 1 - cost; E[payout] = $1 for true complement

    settings = get_settings()
    if effective_edge <= MIN_NET_EDGE_PCT:
        return None

    # Fees on both legs
    fee_rate_a = market_a.taker_fee
    fee_rate_b = market_b.taker_fee
    capital = min(max_capital_usd, 200.0)
    fee_cost = capital * (fee_rate_a + fee_rate_b) / 2
    gas = sum(
        POLYGON_GAS_ESTIMATE_USD
        for m in [market_a, market_b]
        if m.exchange == Exchange.POLYMARKET
    )
    net_edge_pct = effective_edge - (fee_cost / capital) - (gas / capital)
    net_edge_usd = capital * net_edge_pct

    if net_edge_pct <= MIN_NET_EDGE_PCT:
        return None

    risk_flags: list[RiskFlag] = []
    assessor = ResolutionRiskAssessor()
    res_risk = assessor.assess(market_a, market_b)

    if res_risk.flagged:
        risk_flags.append(RiskFlag.RESOLUTION_RISK)
    if res_risk.risk_level >= 0.5:
        risk_flags.append(RiskFlag.UMA_DISPUTE_RISK)

    # Near-complement risk flag when P(neither) is material
    if p_neither > 0.05:
        risk_flags.append(RiskFlag.RESOLUTION_RISK)

    gross_edge_usd = capital * gross_edge
    if (fee_cost + gas) / max(gross_edge_usd, 0.01) > settings.fee_edge_max_consumption:
        risk_flags.append(RiskFlag.FEE_EXCESSIVE)

    expiry = max(
        filter(None, [market_a.expiry, market_b.expiry]),
        default=None,
    )
    days = 30.0
    if expiry:
        days = max(1.0, (expiry - datetime.now(timezone.utc)).total_seconds() / 86400)
    aroc = (net_edge_usd / capital) * (365.0 / days)

    if aroc < settings.aroc_minimum_annual:
        risk_flags.append(RiskFlag.AROC_BELOW_MIN)

    # Confidence: lower when near-complement risk exists
    confidence = max(0.50, 0.92 - res_risk.risk_level * 0.4 - p_neither * 0.5)

    return ArbitrageOpportunity(
        alpha_type=AlphaType.CROSS_EXCHANGE_ARB,
        market_ids=[market_a.market_id, market_b.market_id],
        exchanges=[market_a.exchange, market_b.exchange],
        gross_edge_pct=gross_edge,
        net_edge_pct=net_edge_pct,
        gross_edge_usd=gross_edge_usd,
        net_edge_usd=net_edge_usd,
        required_capital_usd=capital,
        fee_cost_usd=fee_cost,
        gas_cost_usd=gas,
        expiry=expiry,
        aroc_annual=aroc,
        risk_flags=risk_flags,
        resolution_risk=res_risk,
        confidence=confidence,
        legs=[
            {
                "action": "buy",
                "side": Side.YES.value,
                "price": ask_a,
                "size_usd": capital / 2,
                "market_id": market_a.market_id,
                "exchange": market_a.exchange.value,
                "note": f"Complement YES-A (p_neither={p_neither:.2f})",
            },
            {
                "action": "buy",
                "side": Side.YES.value,
                "price": ask_b,
                "size_usd": capital / 2,
                "market_id": market_b.market_id,
                "exchange": market_b.exchange.value,
                "note": "Complement YES-B",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Cross-Exchange Arbitrage
# ---------------------------------------------------------------------------

class CrossExchangeArbDetector:
    """
    Detects arb between economically equivalent markets on different exchanges.

    Strategy:
        If ask(YES, Kalshi) + ask(NO, Polymarket) < 1 (or vice versa):
            Buy YES on exchange with lower ask
            Buy NO on other exchange
            Guaranteed $1 payout per combined share pair

    Critical constraint: must assess resolution risk BEFORE flagging as true arb.
    """

    def __init__(self) -> None:
        self._risk_assessor = ResolutionRiskAssessor()

    def detect(
        self,
        market_a: Market,
        market_b: Market,
        min_similarity: float = 0.60,
        max_capital_usd: float = 500.0,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect cross-exchange arb between market_a and market_b.
        Returns None if no actionable opportunity exists.
        """
        # Must be on different exchanges
        if market_a.exchange == market_b.exchange:
            return None

        # Title similarity: combined token-overlap + SequenceMatcher
        # Token overlap handles cross-exchange naming differences better
        sim = _title_match_score(market_a.title, market_b.title)
        if sim < min_similarity:
            return None

        yes_a = market_a.yes_outcome
        yes_b = market_b.yes_outcome
        no_a = market_a.no_outcome
        no_b = market_b.no_outcome
        if not all([yes_a, yes_b, no_a, no_b]):
            return None

        settings = get_settings()
        res_risk = self._risk_assessor.assess(market_a, market_b)

        # --- Scenario 1: Buy YES on A, Buy NO on B
        cost_1 = yes_a.implied_prob_ask + no_b.implied_prob_ask  # type: ignore[union-attr]
        # --- Scenario 2: Buy YES on B, Buy NO on A
        cost_2 = yes_b.implied_prob_ask + no_a.implied_prob_ask  # type: ignore[union-attr]

        best_cost = min(cost_1, cost_2)
        if best_cost >= 1.0:
            return None  # No arb

        if cost_1 <= cost_2:
            # Buy YES on A, NO on B
            buy_yes_market, buy_yes_ask = market_a, yes_a.implied_prob_ask  # type: ignore
            buy_no_market, buy_no_ask = market_b, no_b.implied_prob_ask     # type: ignore
            buy_yes_side, buy_no_side = Side.YES, Side.NO
        else:
            buy_yes_market, buy_yes_ask = market_b, yes_b.implied_prob_ask  # type: ignore
            buy_no_market, buy_no_ask = market_a, no_a.implied_prob_ask     # type: ignore
            buy_yes_side, buy_no_side = Side.YES, Side.NO

        gross_edge = 1.0 - best_cost
        # Fees: taker fee on each exchange
        fee_rate_yes = buy_yes_market.taker_fee
        fee_rate_no = buy_no_market.taker_fee
        # Gas: Polymarket legs require on-chain tx
        gas = sum(
            POLYGON_GAS_ESTIMATE_USD
            for m in [buy_yes_market, buy_no_market]
            if m.exchange == Exchange.POLYMARKET
        )

        # Optimal position size
        capital = min(max_capital_usd, 200.0)
        fee_cost = capital * (fee_rate_yes + fee_rate_no) / 2
        net_edge_pct = gross_edge - (fee_cost / capital) - (gas / capital)
        net_edge_usd = capital * net_edge_pct

        if net_edge_pct <= MIN_NET_EDGE_PCT:
            return None

        risk_flags: list[RiskFlag] = []
        if res_risk.flagged:
            risk_flags.append(RiskFlag.RESOLUTION_RISK)
        if res_risk.risk_level >= 0.5:
            risk_flags.append(RiskFlag.UMA_DISPUTE_RISK)

        gross_edge_usd = capital * gross_edge
        if (fee_cost + gas) / max(gross_edge_usd, 0.01) > settings.fee_edge_max_consumption:
            risk_flags.append(RiskFlag.FEE_EXCESSIVE)

        # AROC — use the LATER expiry (conservative)
        expiry = max(
            filter(None, [market_a.expiry, market_b.expiry]),
            default=None,
        )
        days = 30.0
        if expiry:
            days = max(1.0, (expiry - datetime.now(timezone.utc)).total_seconds() / 86400)
        aroc = (net_edge_usd / capital) * (365.0 / days)

        if aroc < settings.aroc_minimum_annual:
            risk_flags.append(RiskFlag.AROC_BELOW_MIN)

        return ArbitrageOpportunity(
            alpha_type=AlphaType.CROSS_EXCHANGE_ARB,
            market_ids=[market_a.market_id, market_b.market_id],
            exchanges=[market_a.exchange, market_b.exchange],
            gross_edge_pct=gross_edge,
            net_edge_pct=net_edge_pct,
            gross_edge_usd=gross_edge_usd,
            net_edge_usd=net_edge_usd,
            required_capital_usd=capital,
            fee_cost_usd=fee_cost,
            gas_cost_usd=gas,
            expiry=expiry,
            aroc_annual=aroc,
            risk_flags=risk_flags,
            resolution_risk=res_risk,
            confidence=max(0.5, 0.95 - res_risk.risk_level * 0.5),
            legs=[
                {
                    "action": "buy",
                    "side": buy_yes_side.value,
                    "price": buy_yes_ask,
                    "size_usd": capital / 2,
                    "market_id": buy_yes_market.market_id,
                    "exchange": buy_yes_market.exchange.value,
                },
                {
                    "action": "buy",
                    "side": buy_no_side.value,
                    "price": buy_no_ask,
                    "size_usd": capital / 2,
                    "market_id": buy_no_market.market_id,
                    "exchange": buy_no_market.exchange.value,
                },
            ],
        )

    def scan_universe(
        self,
        all_markets: list[Market],
        min_similarity: float = 0.35,
    ) -> list[ArbitrageOpportunity]:
        """
        O(n²) scan of market universe for cross-exchange arb.

        Detects two patterns for every Poly × Kalshi pair:
          1. Same-event arb:    ask(YES-A) + ask(NO-B) < $1
          2. Complement arb:    ask(YES-A) + ask(YES-B) < $1

        min_similarity lowered to 0.35 (from 0.60) because token overlap
        handles cross-exchange naming differences; false-positive pairs are
        filtered by the price check (cost < $1) which is objective.

        For production with 1000s of markets, use embedding similarity search.
        """
        poly_markets = [m for m in all_markets if m.exchange == Exchange.POLYMARKET]
        kalshi_markets = [m for m in all_markets if m.exchange == Exchange.KALSHI]

        opps: list[ArbitrageOpportunity] = []
        seen: set[tuple[str, str]] = set()

        for pm in poly_markets:
            for km in kalshi_markets:
                pair_key = (pm.market_id, km.market_id)
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # Pattern 1: same-event arb (buy YES on one, NO on other)
                opp = self.detect(pm, km, min_similarity=min_similarity)
                if opp:
                    opps.append(opp)
                    continue  # found same-event arb; complement unlikely too

                # Pattern 2: complement pair arb (buy YES-A + YES-B)
                comp_opp = detect_complement_arb(pm, km)
                if comp_opp:
                    opps.append(comp_opp)

        logger.info(
            "Cross-exchange scan: %d poly × %d kalshi = %d opportunities",
            len(poly_markets), len(kalshi_markets), len(opps),
        )
        return sorted(opps, key=lambda o: o.net_edge_usd, reverse=True)


# ---------------------------------------------------------------------------
# Conditional / Combinatorial Arbitrage
# ---------------------------------------------------------------------------

def detect_conditional_arb(
    market_a: Market,   # "Trump wins PA"
    market_b: Market,   # "Trump wins Election"
    market_c: Market,   # "Trump wins PA AND Election" (joint)
) -> Optional[ArbitrageOpportunity]:
    """
    Detect logical impossibilities in conditional markets.

    Mathematical constraint:
        P(A ∩ B) ≤ min(P(A), P(B))
        P(A ∪ B) ≥ max(P(A), P(B))
        P(A ∩ B) + P(A ∩ ¬B) + P(¬A ∩ B) + P(¬A ∩ ¬B) = 1

    If market_c's price (joint event) exceeds min(market_a, market_b),
    there is a definite mispricing.
    """
    prob_a = market_a.implied_prob_yes_mid
    prob_b = market_b.implied_prob_yes_mid
    prob_c = market_c.implied_prob_yes_mid  # P(A ∩ B)

    if prob_a is None or prob_b is None or prob_c is None:
        return None

    # Fréchet upper bound: P(A ∩ B) ≤ min(P(A), P(B))
    upper_bound = min(prob_a, prob_b)

    if prob_c <= upper_bound + 0.02:  # allow 2% noise
        return None  # No violation

    # Violation detected: joint probability exceeds marginal
    violation_magnitude = prob_c - upper_bound

    # Execution: sell joint market (too expensive), buy component market
    # Short joint market: sell YES at current price (collect prob_c per share)
    # Buy the cheaper component: spend prob_a or prob_b
    edge = violation_magnitude

    # Approximate sizing
    capital = 100.0  # small size for conditional arb (complex settlement)
    fee_est = capital * 0.04  # assume ~4% fees for 2 legs
    net_edge_usd = capital * edge - fee_est

    if net_edge_usd <= 0:
        return None

    return ArbitrageOpportunity(
        alpha_type=AlphaType.CONDITIONAL_ARB,
        market_ids=[market_a.market_id, market_b.market_id, market_c.market_id],
        exchanges=[market_a.exchange, market_b.exchange, market_c.exchange],
        gross_edge_pct=edge,
        net_edge_pct=edge - (fee_est / capital),
        gross_edge_usd=capital * edge,
        net_edge_usd=net_edge_usd,
        required_capital_usd=capital,
        fee_cost_usd=fee_est,
        gas_cost_usd=0.0,
        confidence=0.80,
        risk_flags=[RiskFlag.RESOLUTION_RISK],
        resolution_risk=ResolutionRiskAssessment(
            flagged=True,
            reason="Conditional markets may resolve on different schedules",
            risk_level=0.5,
        ),
        legs=[
            {
                "action": "sell",
                "side": Side.YES.value,
                "price": prob_c,
                "size_usd": capital,
                "market_id": market_c.market_id,
                "exchange": market_c.exchange.value,
                "note": "Sell overpriced joint probability",
            },
            {
                "action": "buy",
                "side": Side.YES.value,
                "price": min(prob_a, prob_b),
                "size_usd": capital,
                "market_id": (market_a if prob_a < prob_b else market_b).market_id,
                "exchange": (market_a if prob_a < prob_b else market_b).exchange.value,
                "note": "Buy cheaper marginal probability as hedge",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Universe Scanner — runs all arb strategies
# ---------------------------------------------------------------------------

class ArbitrageScanner:
    """
    Top-level scanner. Call `scan()` with a list of markets;
    returns all actionable arb opportunities sorted by net USD edge.
    """

    def __init__(self) -> None:
        self._cross_ex_detector = CrossExchangeArbDetector()

    def scan(self, markets: list[Market]) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        # 1. Intra-market arb (single market, both sides)
        for market in markets:
            opp = detect_intra_market_arb(market)
            if opp:
                opportunities.append(opp)

        # 2. Cross-exchange arb
        cross_opps = self._cross_ex_detector.scan_universe(markets)
        opportunities.extend(cross_opps)

        # 3. Conditional arb — requires explicit market triplets
        # (caller must pass pre-identified triplets)

        actionable = [o for o in opportunities if o.is_actionable]
        logger.info(
            "Arb scan: %d total opportunities, %d actionable",
            len(opportunities), len(actionable),
        )
        # Return actionable only — callers log all, execute actionales
        return sorted(actionable, key=lambda o: o.net_edge_usd, reverse=True)
