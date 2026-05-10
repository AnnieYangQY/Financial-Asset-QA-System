"""单元测试：market_agent 的窗口推断。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.market_agent import _detect_window  # noqa: E402


def test_default_window():
    assert _detect_window("阿里巴巴当前股价是多少？") == 30


def test_explicit_7_days():
    assert _detect_window("BABA 最近 7 天涨跌情况如何？") == 7


def test_explicit_30_days():
    assert _detect_window("特斯拉最近 30 天表现") == 30


def test_one_week_synonym():
    assert _detect_window("BABA 近一周走势") == 7


def test_one_month_synonym():
    assert _detect_window("贵州茅台近一月情况") == 30


def test_three_month_synonym():
    assert _detect_window("茅台近三月走势") == 90
