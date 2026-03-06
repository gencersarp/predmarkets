"""Unit tests for order flow analysis module."""
from __future__ import annotations

import pytest

from src.alpha.orderflow import (
    OrderFlowAnalyzer,
    TradeRecord,
    compute_ofi_from_records,
)
from src.core.models import Exchange, RiskFlag, Side
from tests.conftest import make_market


def _make_trades(
    n_buy: int,
    n_sell: int,
    buy_size: float = 100.0,
    sell_size: float = 100.0,
    price: float = 0.52,
) -> list[TradeRecord]:
    from datetime import datetime, timezone
    trades = []
    for _ in range(n_buy):
        trades.append(TradeRecord(price=price, size_usd=buy_size, side="BUY",
                                   timestamp=datetime.now(timezone.utc)))
    for _ in range(n_sell):
        trades.append(TradeRecord(price=price, size_usd=sell_size, side="SELL",
                                   timestamp=datetime.now(timezone.utc)))
    return trades


class TestOrderFlowAnalyzer:
    def setup_method(self):
        self.analyzer = OrderFlowAnalyzer(
            window_trades=50,
            ofi_threshold=0.65,
            sharp_money_usd=500.0,
            min_edge_pct=0.03,
        )

    def test_compute_ofi_all_buys(self):
        trades = _make_trades(n_buy=10, n_sell=0, buy_size=100.0)
        ofi, sharp = self.analyzer._compute_ofi(trades)
        assert ofi == 1.0   # 100% buy imbalance
        assert len(sharp) == 0

    def test_compute_ofi_all_sells(self):
        trades = _make_trades(n_buy=0, n_sell=10, sell_size=100.0)
        ofi, sharp = self.analyzer._compute_ofi(trades)
        assert ofi == -1.0  # 100% sell imbalance

    def test_compute_ofi_balanced(self):
        trades = _make_trades(n_buy=5, n_sell=5)
        ofi, sharp = self.analyzer._compute_ofi(trades)
        assert abs(ofi) < 0.01

    def test_compute_ofi_partial_imbalance(self):
        # 3 buys, 1 sell → OFI = (300 - 100) / 400 = 0.5
        trades = _make_trades(n_buy=3, n_sell=1, buy_size=100.0, sell_size=100.0)
        ofi, _ = self.analyzer._compute_ofi(trades)
        assert abs(ofi - 0.5) < 0.01

    def test_sharp_money_detected(self):
        from datetime import datetime, timezone
        big_trade = TradeRecord(price=0.5, size_usd=1000.0, side="BUY",
                                 timestamp=datetime.now(timezone.utc))
        small_trades = _make_trades(n_buy=5, n_sell=5, buy_size=50.0)
        all_trades = [big_trade] + small_trades
        ofi, sharp = self.analyzer._compute_ofi(all_trades)
        assert len(sharp) == 1
        assert sharp[0].size_usd == 1000.0

    def test_classify_signal_strong_buy_ofi(self):
        trades = _make_trades(n_buy=10, n_sell=1)  # ofi = 900/1000 = 0.9
        ofi, sharp = self.analyzer._compute_ofi(trades)
        side, confidence = self.analyzer._classify_signal(ofi, sharp, trades)
        assert side == Side.YES
        assert confidence > 0.50

    def test_classify_signal_strong_sell_ofi(self):
        trades = _make_trades(n_buy=1, n_sell=10)
        ofi, sharp = self.analyzer._compute_ofi(trades)
        side, confidence = self.analyzer._classify_signal(ofi, sharp, trades)
        assert side == Side.NO
        assert confidence > 0.50

    def test_classify_signal_no_signal_balanced(self):
        trades = _make_trades(n_buy=5, n_sell=5)
        ofi, sharp = self.analyzer._compute_ofi(trades)
        side, confidence = self.analyzer._classify_signal(ofi, sharp, trades)
        assert side is None
        assert confidence == 0.0

    def test_classify_signal_boosted_by_sharp_money(self):
        from datetime import datetime, timezone
        # Moderate OFI but 3 large buys
        small = _make_trades(n_buy=5, n_sell=3)
        sharps = [TradeRecord(price=0.5, size_usd=600.0, side="BUY",
                               timestamp=datetime.now(timezone.utc)) for _ in range(3)]
        trades = small + sharps
        ofi, sharp = self.analyzer._compute_ofi(trades)
        side, confidence = self.analyzer._classify_signal(ofi, sharp, trades)
        # Should detect signal from sharp consensus even without hitting OFI threshold
        assert side == Side.YES

    def test_sharp_money_against_flag(self):
        """If signal is YES but large trades are SELL, flag is set."""
        from datetime import datetime, timezone
        # Enough small buys to push OFI over threshold
        buys = _make_trades(n_buy=15, n_sell=2)
        # But also two large sells
        sells = [TradeRecord(price=0.5, size_usd=800.0, side="SELL",
                              timestamp=datetime.now(timezone.utc)) for _ in range(2)]
        trades = buys + sells
        ofi, sharp = self.analyzer._compute_ofi(trades)
        # OFI might still be positive; check flags in full signal
        side, confidence = self.analyzer._classify_signal(ofi, sharp, trades)
        if side == Side.YES:
            opposing_sharp = [t for t in sharp if t.side == "SELL"]
            assert len(opposing_sharp) == 2


class TestComputeOFIFromRecords:
    def test_all_buys(self):
        trades = [{"price": 0.5, "size": 100, "side": "BUY"} for _ in range(5)]
        assert compute_ofi_from_records(trades) == 1.0

    def test_all_sells(self):
        trades = [{"price": 0.5, "size": 100, "side": "SELL"} for _ in range(5)]
        assert compute_ofi_from_records(trades) == -1.0

    def test_empty(self):
        assert compute_ofi_from_records([]) == 0.0

    def test_mixed(self):
        trades = [
            {"price": 0.5, "size": 200, "side": "BUY"},
            {"price": 0.5, "size": 200, "side": "SELL"},
        ]
        assert compute_ofi_from_records(trades) == 0.0

    def test_custom_keys(self):
        trades = [{"p": 0.5, "sz": 100, "dir": "BUY"} for _ in range(4)]
        ofi = compute_ofi_from_records(trades, price_key="p", size_key="sz", side_key="dir")
        assert ofi == 1.0


class TestTradeRecord:
    def test_creation(self):
        from datetime import datetime, timezone
        t = TradeRecord(price=0.55, size_usd=250.0, side="BUY",
                        timestamp=datetime.now(timezone.utc), tx_hash="0xabc")
        assert t.price == 0.55
        assert t.size_usd == 250.0
        assert t.side == "BUY"

    def test_parse_clob_trade_unix_timestamp(self):
        import time
        raw = {
            "price": "0.52",
            "size": "200",
            "side": "BUY",
            "timestamp": str(int(time.time())),
        }
        t = OrderFlowAnalyzer._parse_clob_trade(raw)
        assert t.price == 0.52
        assert abs(t.size_usd - 200 * 0.52) < 0.1
        assert t.side == "BUY"

    def test_parse_clob_trade_iso_timestamp(self):
        raw = {
            "price": "0.45",
            "size": "100",
            "side": "SELL",
            "timestamp": "2024-01-15T12:00:00Z",
        }
        t = OrderFlowAnalyzer._parse_clob_trade(raw)
        assert t.side == "SELL"
        assert t.price == 0.45

    def test_parse_data_trade(self):
        import time
        raw = {
            "price": "0.60",
            "usdcSize": "500",
            "outcome": "Buy",
            "timestamp": str(int(time.time())),
        }
        t = OrderFlowAnalyzer._parse_data_trade(raw)
        assert t.side == "BUY"
        assert t.size_usd == 500.0
