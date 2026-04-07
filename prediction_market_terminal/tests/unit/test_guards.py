"""Unit tests for risk guard system."""
from __future__ import annotations

import pytest

from src.core.exceptions import (
    CorrelationLimitBreached,
    DrawdownLimitBreached,
    GasLimitExceeded,
    PaperModeViolation,
    PositionSizeTooLarge,
    RiskLimitBreached,
    SlippageLimitExceeded,
    StaleDataError,
)
from src.core.models import Exchange, RiskFlag
from src.risk.guards import (
    GuardResult,
    RiskGuardRunner,
    guard_aroc,
    guard_drawdown,
    guard_fee_consumption,
    guard_gas_price,
    guard_liquidity,
    guard_paper_mode,
    guard_position_size,
    guard_probability_bounds,
    guard_resolution_risk,
    guard_slippage,
    guard_stale_data,
)
from src.risk.portfolio import PortfolioManager
from tests.conftest import make_arb_opportunity, make_directional_signal, make_market


@pytest.fixture
def healthy_portfolio():
    pm = PortfolioManager(initial_nav_usd=1000.0)
    return pm


class TestGuardPaperMode:
    def test_passes_in_paper_mode(self):
        result = guard_paper_mode(is_live_order=False)
        assert result.passed

    def test_raises_in_paper_mode_with_live_order(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.get_settings(), "pmt_mode", "paper", raising=False)
        # settings.is_paper should be True in test env
        # We force it by having is_live_order=True while mode is paper
        with pytest.raises(PaperModeViolation):
            guard_paper_mode(is_live_order=True)


class TestGuardDrawdown:
    def test_passes_with_no_drawdown(self, healthy_portfolio):
        result = guard_drawdown(healthy_portfolio)
        assert result.passed

    def test_raises_when_drawdown_exceeds_limit(self, monkeypatch):
        pm = PortfolioManager(initial_nav_usd=1000.0)
        # Simulate a loss by opening a position that goes to zero
        from tests.conftest import make_market, make_position
        from datetime import datetime, timedelta, timezone

        # Force drawdown by manipulating peak_nav
        pm._peak_nav = 1000.0
        pm._cash_usd = 750.0  # 25% loss

        with pytest.raises(DrawdownLimitBreached):
            guard_drawdown(pm)


class TestGuardPositionSize:
    def test_passes_within_limit(self):
        result = guard_position_size(100.0)
        assert result.passed

    def test_raises_when_too_large(self):
        with pytest.raises(PositionSizeTooLarge):
            guard_position_size(999999.0)

    def test_exactly_at_limit_passes(self):
        from config.settings import get_settings
        limit = get_settings().max_single_position_usd
        result = guard_position_size(limit)
        assert result.passed


class TestGuardStaleData:
    def test_fresh_market_passes(self, market):
        result = guard_stale_data(market)
        assert result.passed

    def test_stale_market_raises(self):
        from datetime import datetime, timedelta, timezone
        m = make_market()
        m.fetched_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        with pytest.raises(StaleDataError):
            guard_stale_data(m)


class TestGuardProbabilityBounds:
    def test_normal_price_passes(self):
        result = guard_probability_bounds(0.50)
        assert result.passed

    def test_near_zero_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_probability_bounds(0.005)  # below 0.01 floor

    def test_near_one_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_probability_bounds(0.98)  # above 0.97 ceiling

    def test_boundary_values_pass(self):
        assert guard_probability_bounds(0.01).passed
        assert guard_probability_bounds(0.97).passed


class TestGuardLiquidity:
    def test_liquid_market_passes(self):
        result = guard_liquidity(50_000.0)
        assert result.passed

    def test_illiquid_market_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_liquidity(500.0)  # below $1,000 minimum

    def test_exactly_at_limit_passes(self):
        from src.core.constants import MIN_MARKET_VOLUME_24H_USD
        result = guard_liquidity(MIN_MARKET_VOLUME_24H_USD)
        assert result.passed

    def test_custom_limit(self):
        with pytest.raises(RiskLimitBreached):
            guard_liquidity(5_000.0, min_volume=10_000.0)


class TestGuardGasPrice:
    def test_normal_gas_passes(self):
        result = guard_gas_price(30.0)
        assert result.passed

    def test_high_gas_raises(self):
        with pytest.raises(GasLimitExceeded):
            guard_gas_price(9999.0)


class TestGuardSlippage:
    def test_low_slippage_passes(self):
        result = guard_slippage(0.001)
        assert result.passed

    def test_high_slippage_raises(self):
        with pytest.raises(SlippageLimitExceeded):
            guard_slippage(0.50)


class TestGuardFeeConsumption:
    def test_low_fee_passes(self):
        result = guard_fee_consumption(gross_edge_usd=100.0, total_costs_usd=20.0)
        assert result.passed  # 20% < 40%

    def test_high_fee_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_fee_consumption(gross_edge_usd=10.0, total_costs_usd=9.0)  # 90%

    def test_zero_gross_edge_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_fee_consumption(gross_edge_usd=0.0, total_costs_usd=5.0)

    def test_exactly_at_limit_passes(self):
        result = guard_fee_consumption(gross_edge_usd=100.0, total_costs_usd=40.0)
        assert result.passed  # exactly 40%


class TestGuardAroc:
    def test_high_aroc_passes(self):
        result = guard_aroc(aroc_annual=1.50)
        assert result.passed

    def test_low_aroc_raises(self):
        with pytest.raises(RiskLimitBreached):
            guard_aroc(aroc_annual=0.05)  # 5% < 30% minimum

    def test_at_minimum_passes(self):
        from config.settings import get_settings
        min_aroc = get_settings().aroc_minimum_annual
        result = guard_aroc(aroc_annual=min_aroc)
        assert result.passed


class TestGuardResolutionRisk:
    def test_low_risk_passes(self):
        opp = make_arb_opportunity()
        result = guard_resolution_risk(opp, max_resolution_risk=0.70)
        assert result.passed

    def test_high_risk_raises(self):
        from src.core.models import ResolutionRiskAssessment
        opp = make_arb_opportunity()
        opp.resolution_risk = ResolutionRiskAssessment(
            flagged=True,
            reason="Completely incompatible resolution",
            risk_level=0.95,
        )
        with pytest.raises(RiskLimitBreached):
            guard_resolution_risk(opp, max_resolution_risk=0.70)


class TestRiskGuardRunner:
    def test_directional_all_pass(self):
        market = make_market()
        signal = make_directional_signal()
        pm = PortfolioManager(initial_nav_usd=1000.0)
        runner = RiskGuardRunner()
        results = runner.run_directional(
            signal=signal,
            market=market,
            portfolio=pm,
            is_live_order=False,
        )
        assert all(r.passed for r in results)

    def test_arb_all_pass(self):
        market = make_market()
        # Use capital within the per-position limit ($150)
        opp = make_arb_opportunity()  # defaults are now sized correctly
        pm = PortfolioManager(initial_nav_usd=1000.0)
        runner = RiskGuardRunner()
        results = runner.run_arbitrage(
            opportunity=opp,
            markets=[market],
            portfolio=pm,
            is_live_order=False,
        )
        assert all(r.passed for r in results)

    def test_preview_returns_dict(self):
        market = make_market()
        signal = make_directional_signal()
        pm = PortfolioManager(initial_nav_usd=1000.0)
        runner = RiskGuardRunner()
        checks = runner.preview(signal, market, pm)
        assert isinstance(checks, dict)
        assert all(v in ("PASS",) or v.startswith("FAIL") for v in checks.values())

    def test_directional_blocked_by_size(self):
        market = make_market()
        from tests.conftest import make_directional_signal
        signal = make_directional_signal(size_usd=999_999.0)  # way too large
        pm = PortfolioManager(initial_nav_usd=1000.0)
        runner = RiskGuardRunner()
        with pytest.raises(PositionSizeTooLarge):
            runner.run_directional(signal=signal, market=market, portfolio=pm)

    def test_guard_result_bool(self):
        g = GuardResult(passed=True, guard_name="test")
        assert bool(g) is True
        g2 = GuardResult(passed=False, guard_name="test")
        assert bool(g2) is False
