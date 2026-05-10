"""单元测试：router 的启发式兜底，无需 LLM。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.router import _heuristic_route  # noqa: E402


def test_heuristic_market_query():
    decision = _heuristic_route("阿里巴巴当前股价是多少？")
    assert decision.intent == "market"
    assert "阿里巴巴" in decision.symbols


def test_heuristic_knowledge_query():
    decision = _heuristic_route("什么是市盈率？")
    assert decision.intent == "knowledge"


def test_heuristic_hybrid_query():
    decision = _heuristic_route("阿里巴巴最近为什么大涨？")
    assert decision.intent == "hybrid"
    assert "阿里巴巴" in decision.symbols


def test_heuristic_baba_7d():
    decision = _heuristic_route("BABA 最近 7 天涨跌情况如何？")
    assert decision.intent == "market"
    assert "BABA" in decision.symbols


def test_heuristic_tesla_trend():
    decision = _heuristic_route("特斯拉近期走势如何？")
    assert decision.intent == "market"
    assert "特斯拉" in decision.symbols
