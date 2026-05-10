"""单元测试：trend_calc。无外部依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.trend_calc import (  # noqa: E402
    classify_trend,
    compute_change,
    compute_max_drawdown,
    compute_volatility,
    summarize,
)


def _make_df(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "date": [f"2024-01-{i+1:02d}" for i in range(n)],
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * n,
    })


def test_compute_change_positive():
    df = _make_df([100, 105, 110, 115])
    assert compute_change(df) == pytest.approx(15.0)


def test_compute_change_negative():
    df = _make_df([100, 90, 85, 80])
    assert compute_change(df) == pytest.approx(-20.0)


def test_classify_trend_up():
    df = _make_df([100, 102, 104, 106, 108, 110])
    assert classify_trend(df) == "上涨"


def test_classify_trend_down():
    df = _make_df([100, 98, 96, 94, 92, 90])
    assert classify_trend(df) == "下跌"


def test_classify_trend_sideways():
    df = _make_df([100, 101, 99, 100.5, 99.5, 100])
    assert classify_trend(df) == "震荡"


def test_max_drawdown():
    df = _make_df([100, 120, 90, 110])
    # peak=120, trough=90 → -25%
    assert compute_max_drawdown(df) == pytest.approx(-25.0, rel=1e-3)


def test_volatility_nonneg():
    df = _make_df([100, 105, 95, 110, 90])
    assert compute_volatility(df) > 0


def test_summarize_round_trip():
    df = _make_df([100, 105, 110])
    s = summarize(df)
    d = s.to_dict()
    assert d["start_price"] == 100.0
    assert d["end_price"] == 110.0
    assert d["change_pct"] == pytest.approx(10.0)
    assert d["trend"] == "上涨"
