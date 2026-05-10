"""
端到端集成测试 — 默认 skip，因为依赖：
  1. AKShare 网络访问
  2. LLM API key
  3. 已构建好的 Chroma 向量库

要跑通需要：
  cp .env.example .env  # 然后填好 LLM_API_KEY
  python -m rag.ingest
  pytest tests/test_integration.py -m integration -s

文档要求覆盖的查询：
  - 阿里巴巴当前股价是多少？     (market)
  - BABA 最近 7 天涨跌情况如何？  (market)
  - 阿里巴巴最近为何 1 月 15 日大涨？  (hybrid)
  - 特斯拉近期走势如何？        (market)
  - 什么是市盈率？               (knowledge)
  - 收入和净利润的区别是什么？     (knowledge)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.integration


def _have_llm() -> bool:
    key = os.getenv("LLM_API_KEY", "")
    return bool(key) and key != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def _have_chroma() -> bool:
    return (ROOT / ".chroma_db").exists()


@pytest.mark.skipif(not _have_llm(), reason="未配置 LLM_API_KEY")
def test_market_baba_current():
    from agents.graph import answer
    qa = answer("阿里巴巴当前股价是多少？")
    assert qa.intent == "market"
    assert qa.facts, "应当返回客观数据列表"


@pytest.mark.skipif(not _have_llm(), reason="未配置 LLM_API_KEY")
def test_market_baba_7d():
    from agents.graph import answer
    qa = answer("BABA 最近 7 天涨跌情况如何？")
    assert qa.intent == "market"
    # 7 天窗口 → 应该包含涨跌幅 fact
    labels = " ".join(f.label for f in qa.facts)
    assert "涨跌" in labels or "trend" in labels.lower() or qa.analysis


@pytest.mark.skipif(not _have_llm(), reason="未配置 LLM_API_KEY")
def test_hybrid_baba_jan15():
    from agents.graph import answer
    qa = answer("阿里巴巴最近为何 1 月 15 日大涨？")
    assert qa.intent in ("hybrid", "market")  # router 也可能判 market


@pytest.mark.skipif(not _have_llm(), reason="未配置 LLM_API_KEY")
def test_market_tsla_recent():
    from agents.graph import answer
    qa = answer("特斯拉近期走势如何？")
    assert qa.intent == "market"


@pytest.mark.skipif(not (_have_llm() and _have_chroma()), reason="未配置 LLM 或未构建向量库")
def test_knowledge_pe():
    from agents.graph import answer
    qa = answer("什么是市盈率？")
    assert qa.intent == "knowledge"
    assert qa.citations, "RAG 应返回引用"


@pytest.mark.skipif(not (_have_llm() and _have_chroma()), reason="未配置 LLM 或未构建向量库")
def test_knowledge_revenue_vs_profit():
    from agents.graph import answer
    qa = answer("收入和净利润的区别是什么？")
    assert qa.intent == "knowledge"
    assert qa.citations
