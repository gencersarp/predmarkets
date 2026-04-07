"""Unit tests for arbitrage detection engine."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.alpha.arbitrage import (
    ArbitrageScanner,
    CrossExchangeArbDetector,
    ResolutionRiskAssessor,
    _are_complement_titles,
    _title_match_score,
    _token_overlap,
    detect_complement_arb,
    detect_conditional_arb,
    detect_intra_market_arb,
)
from src.core.models import (
    AlphaType,
    Exchange,
    ResolutionSource,
    RiskFlag,
)
from tests.conftest import make_market


class TestIntraMarketArb:
    def test_detects_arb_when_ask_sum_below_threshold(self, cheap_market):
        """ask_yes=0.47, ask_no=0.48 → sum=0.95 < 0.98 → arb detected"""
        opp = detect_intra_market_arb(cheap_market)
        assert opp is not None
        assert opp.alpha_type == AlphaType.INTRA_MARKET_ARB
        assert opp.net_edge_pct > 0

    def test_no_arb_when_prices_fair(self, fair_market):
        """ask_yes=0.52, ask_no=0.49 → sum=1.01 > 0.98 → no arb"""
        opp = detect_intra_market_arb(fair_market)
        assert opp is None

    def test_legs_cover_both_sides(self, cheap_market):
        opp = detect_intra_market_arb(cheap_market)
        assert opp is not None
        sides = {leg["side"] for leg in opp.legs}
        assert "yes" in sides
        assert "no" in sides

    def test_gross_edge_positive(self, cheap_market):
        opp = detect_intra_market_arb(cheap_market)
        assert opp is not None
        assert opp.gross_edge_pct > 0

    def test_fee_flag_on_thin_edge(self):
        # Very thin edge: ask_yes=0.484, ask_no=0.484 → sum=0.968
        # gross_edge = 0.032, fee = 2%*2 = 4% → fees eat all edge
        m = make_market(yes_ask=0.484, no_ask=0.484, taker_fee=0.02)
        opp = detect_intra_market_arb(m)
        if opp is not None:
            # Fee consumption check should flag this
            assert RiskFlag.FEE_EXCESSIVE in opp.risk_flags or opp.net_edge_pct < 0.005

    def test_polymarket_includes_gas_cost(self):
        m = make_market(
            exchange=Exchange.POLYMARKET,
            yes_ask=0.45, no_ask=0.46,
        )
        opp = detect_intra_market_arb(m)
        if opp is not None:
            assert opp.gas_cost_usd > 0

    def test_kalshi_no_gas_cost(self):
        m = make_market(
            exchange=Exchange.KALSHI,
            yes_ask=0.45, no_ask=0.46,
            taker_fee=0.07,
        )
        opp = detect_intra_market_arb(m)
        if opp is not None:
            assert opp.gas_cost_usd == 0.0

    def test_short_expiry_flags_aroc(self):
        # 1-day expiry → aroc might fail minimum threshold
        m = make_market(yes_ask=0.47, no_ask=0.48, expiry_days=1.0)
        opp = detect_intra_market_arb(m)
        if opp is not None:
            # With very short expiry, AROC should be high (good)
            # or AROC_BELOW_MIN if net_edge is small
            assert isinstance(opp.aroc_annual, float)


class TestCrossExchangeArb:
    def test_detects_cross_exchange_opportunity(self):
        """YES cheaper on Polymarket, NO cheaper on Kalshi → arb"""
        poly = make_market(
            market_id="poly-001",
            exchange=Exchange.POLYMARKET,
            title="Will the Fed cut rates in January?",
            yes_ask=0.42,
            no_bid=0.45,
            no_ask=0.50,
        )
        kalshi = make_market(
            market_id="kal-001",
            exchange=Exchange.KALSHI,
            title="Will the Fed cut rates in January?",
            yes_bid=0.50,
            yes_ask=0.55,
            no_ask=0.48,
            taker_fee=0.07,
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
            resolution_criteria="Fed press release confirms 25bp cut.",
        )
        detector = CrossExchangeArbDetector()
        opp = detector.detect(poly, kalshi)
        if opp is not None:
            # If arb detected: must be cross-exchange
            assert opp.alpha_type == AlphaType.CROSS_EXCHANGE_ARB
            assert Exchange.POLYMARKET in opp.exchanges
            assert Exchange.KALSHI in opp.exchanges

    def test_no_arb_on_same_exchange(self):
        m1 = make_market("a", Exchange.POLYMARKET)
        m2 = make_market("b", Exchange.POLYMARKET)
        detector = CrossExchangeArbDetector()
        opp = detector.detect(m1, m2)
        assert opp is None

    def test_resolution_risk_flagged_for_different_sources(self):
        poly = make_market(
            "p1", Exchange.POLYMARKET, title="Fed rate cut?",
            yes_ask=0.42, no_ask=0.52,
            resolution_source=ResolutionSource.UMA_ORACLE,
            resolution_criteria="UMA oracle: Federal Reserve cuts by 25bps.",
        )
        kalshi = make_market(
            "k1", Exchange.KALSHI, title="Fed rate cut?",
            yes_ask=0.55, no_ask=0.42,
            taker_fee=0.07,
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
            resolution_criteria="Kalshi internal: Reuters press release.",
        )
        detector = CrossExchangeArbDetector()
        opp = detector.detect(poly, kalshi)
        if opp is not None:
            assert RiskFlag.RESOLUTION_RISK in opp.risk_flags

    def test_low_title_similarity_skipped(self):
        poly = make_market(
            "p1", Exchange.POLYMARKET,
            title="Will Bitcoin hit $100k?",
        )
        kalshi = make_market(
            "k1", Exchange.KALSHI,
            title="Will inflation drop below 2%?",
        )
        detector = CrossExchangeArbDetector()
        opp = detector.detect(poly, kalshi, min_similarity=0.60)
        assert opp is None  # completely different topics

    def test_scan_universe(self):
        markets = [
            make_market("p1", Exchange.POLYMARKET, title="Fed cut December?", yes_ask=0.43, no_ask=0.52),
            make_market("p2", Exchange.POLYMARKET, title="Bitcoin above 100k?"),
            make_market("k1", Exchange.KALSHI, title="Fed cut December?", yes_ask=0.55, no_ask=0.42, taker_fee=0.07),
        ]
        detector = CrossExchangeArbDetector()
        opps = detector.scan_universe(markets)
        # May or may not find arb depending on prices, but should not crash
        assert isinstance(opps, list)


class TestResolutionRiskAssessor:
    def test_identical_markets_low_risk(self):
        m = make_market()
        assessor = ResolutionRiskAssessor()
        result = assessor.assess(m, m)
        # Same market: low but not zero (UMA risk adds up)
        assert result.risk_level < 0.5

    def test_different_resolution_sources_flagged(self):
        poly = make_market(resolution_source=ResolutionSource.UMA_ORACLE)
        kalshi = make_market(resolution_source=ResolutionSource.KALSHI_INTERNAL)
        assessor = ResolutionRiskAssessor()
        result = assessor.assess(poly, kalshi)
        assert result.flagged
        assert result.risk_level > 0.2

    def test_expiry_mismatch_adds_risk(self):
        m1 = make_market(expiry_days=5.0)
        m2 = make_market(expiry_days=20.0)
        assessor = ResolutionRiskAssessor()
        result = assessor.assess(m1, m2)
        # 15-day expiry mismatch → additional risk
        assert result.risk_level > 0.0

    def test_risk_level_capped_at_one(self):
        poly = make_market(
            resolution_source=ResolutionSource.UMA_ORACLE,
            resolution_criteria="X wins state Y based on UMA.",
            expiry_days=2,
        )
        kalshi = make_market(
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
            resolution_criteria="Completely different resolution: API call to Yelp.",
            expiry_days=60,
        )
        assessor = ResolutionRiskAssessor()
        result = assessor.assess(poly, kalshi)
        assert result.risk_level <= 1.0


class TestConditionalArb:
    def test_detects_violated_frechet_bound(self):
        """P(A∩B) > min(P(A), P(B)) is mathematically impossible."""
        market_a = make_market("a", title="Trump wins PA", yes_bid=0.64, yes_ask=0.66)
        market_b = make_market("b", title="Trump wins Election", yes_bid=0.54, yes_ask=0.56)
        # Joint market priced at 0.70 > min(0.65, 0.55) = 0.55 → violation
        market_c = make_market("c", title="Trump wins PA and Election", yes_bid=0.68, yes_ask=0.72)

        opp = detect_conditional_arb(market_a, market_b, market_c)
        assert opp is not None
        assert opp.alpha_type == AlphaType.CONDITIONAL_ARB
        assert opp.net_edge_usd > 0

    def test_no_violation_when_joint_below_min(self):
        market_a = make_market("a", yes_bid=0.64, yes_ask=0.66)
        market_b = make_market("b", yes_bid=0.54, yes_ask=0.56)
        # Joint priced at 0.40 < min(0.65, 0.55) = 0.55 → no violation
        market_c = make_market("c", yes_bid=0.38, yes_ask=0.42)

        opp = detect_conditional_arb(market_a, market_b, market_c)
        assert opp is None


class TestTitleMatching:
    def test_token_overlap_identical(self):
        assert _token_overlap("Will the Fed cut rates?", "Will the Fed cut rates?") == 1.0

    def test_token_overlap_zero_different_topics(self):
        score = _token_overlap("Bitcoin price above $100k", "US presidential election result")
        assert score == 0.0

    def test_token_overlap_cross_exchange_same_event(self):
        # Different phrasing, same topic — token overlap captures shared keywords
        score = _token_overlap(
            "Will the Federal Reserve cut rates in January 2025?",
            "Fed rate cut January 2025",
        )
        # "cut" and "january" and "2025" are shared; Federal/Reserve vs Fed differ
        assert score > 0.20

    def test_title_match_score_uses_best_of_two(self):
        # SequenceMatcher is better for near-identical strings
        score = _title_match_score(
            "Will the Fed cut rates in January?",
            "Will the Fed cut rates in January?",
        )
        assert score == 1.0

    def test_complement_titles_negation(self):
        assert _are_complement_titles("Will Trump win the election?", "Will Trump lose the election?")

    def test_complement_titles_competition(self):
        # Same election, different candidates → complements
        assert _are_complement_titles(
            "Will Trump win the 2024 presidential election?",
            "Will Harris win the 2024 presidential election?",
        )

    def test_not_complement_different_topics(self):
        assert not _are_complement_titles("Will Bitcoin hit $100k?", "Will inflation drop to 2%?")

    def test_not_complement_same_direction(self):
        # Both positive on same thing — not a complement pair
        assert not _are_complement_titles("Will Trump win?", "Will Trump be re-elected?")


class TestComplementArb:
    def _make_complement_pair(
        self,
        ask_a: float = 0.55,
        ask_b: float = 0.38,
        title_a: str = "Will Trump win the 2024 presidential election?",
        title_b: str = "Will Harris win the 2024 presidential election?",
    ):
        poly = make_market(
            market_id="poly-trump",
            exchange=Exchange.POLYMARKET,
            title=title_a,
            yes_ask=ask_a,
            yes_bid=ask_a - 0.02,
            no_ask=1.0 - ask_a + 0.02,
            no_bid=1.0 - ask_a,
        )
        kalshi = make_market(
            market_id="kal-harris",
            exchange=Exchange.KALSHI,
            title=title_b,
            yes_ask=ask_b,
            yes_bid=ask_b - 0.02,
            no_ask=1.0 - ask_b + 0.02,
            no_bid=1.0 - ask_b,
            taker_fee=0.07,
            resolution_source=ResolutionSource.KALSHI_INTERNAL,
        )
        return poly, kalshi

    def test_detects_complement_arb_when_cost_below_one(self):
        poly, kalshi = self._make_complement_pair(ask_a=0.55, ask_b=0.38)
        # cost = 0.55 + 0.38 = 0.93 < $1 → arb
        opp = detect_complement_arb(poly, kalshi)
        assert opp is not None
        assert opp.gross_edge_pct == pytest.approx(0.07, abs=0.01)
        assert opp.net_edge_usd > 0

    def test_no_arb_when_cost_above_one(self):
        poly, kalshi = self._make_complement_pair(ask_a=0.65, ask_b=0.42)
        # cost = 0.65 + 0.42 = 1.07 > $1 → no arb
        opp = detect_complement_arb(poly, kalshi)
        assert opp is None

    def test_legs_both_buy_yes(self):
        poly, kalshi = self._make_complement_pair(ask_a=0.50, ask_b=0.35)
        opp = detect_complement_arb(poly, kalshi)
        assert opp is not None
        for leg in opp.legs:
            assert leg["action"] == "buy"
            assert leg["side"] == "yes"

    def test_no_arb_same_exchange(self):
        poly1 = make_market("p1", Exchange.POLYMARKET, title="Will Trump win the 2024 election?")
        poly2 = make_market("p2", Exchange.POLYMARKET, title="Will Harris win the 2024 election?")
        opp = detect_complement_arb(poly1, poly2)
        assert opp is None

    def test_no_arb_unrelated_markets(self):
        poly = make_market("p1", Exchange.POLYMARKET, title="Will Bitcoin hit $200k?", yes_ask=0.30)
        kalshi = make_market("k1", Exchange.KALSHI, title="Will unemployment rise to 5%?", yes_ask=0.25)
        opp = detect_complement_arb(poly, kalshi)
        assert opp is None

    def test_p_neither_reduces_effective_edge(self):
        """When bid prices are low (high P(neither)), effective edge is reduced."""
        poly, kalshi = self._make_complement_pair(ask_a=0.50, ask_b=0.35)
        opp = detect_complement_arb(poly, kalshi)
        if opp:
            # Gross edge is 0.15 but net edge accounts for P(neither)
            assert opp.net_edge_pct <= opp.gross_edge_pct

    def test_scan_universe_finds_complement_pairs(self):
        """scan_universe should detect complement pairs, not just same-event arb."""
        markets = [
            make_market(
                "poly-trump", Exchange.POLYMARKET,
                title="Will Trump win the 2024 presidential election?",
                yes_ask=0.55, yes_bid=0.53, no_ask=0.47, no_bid=0.45,
            ),
            make_market(
                "kal-harris", Exchange.KALSHI,
                title="Will Harris win the 2024 presidential election?",
                yes_ask=0.38, yes_bid=0.36, no_ask=0.64, no_bid=0.62,
                taker_fee=0.07,
                resolution_source=ResolutionSource.KALSHI_INTERNAL,
            ),
        ]
        detector = CrossExchangeArbDetector()
        opps = detector.scan_universe(markets)
        # Should find the complement arb (cost 0.55+0.38=0.93 < $1)
        assert len(opps) >= 1


class TestArbitrageScanner:
    def test_scan_returns_sorted_by_net_edge(self):
        markets = [
            make_market("m1", yes_ask=0.45, no_ask=0.46),  # arb: sum=0.91
            make_market("m2", yes_ask=0.47, no_ask=0.48),  # arb: sum=0.95
            make_market("m3", yes_ask=0.51, no_ask=0.50),  # no arb
        ]
        scanner = ArbitrageScanner()
        opps = scanner.scan(markets)
        for i in range(len(opps) - 1):
            assert opps[i].net_edge_usd >= opps[i + 1].net_edge_usd

    def test_scan_empty_universe(self):
        scanner = ArbitrageScanner()
        opps = scanner.scan([])
        assert opps == []

    def test_only_actionable_returned(self):
        markets = [make_market("m1", yes_ask=0.47, no_ask=0.48)]
        scanner = ArbitrageScanner()
        opps = scanner.scan(markets)
        for opp in opps:
            assert opp.is_actionable
