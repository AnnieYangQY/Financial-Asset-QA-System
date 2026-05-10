"""单元测试：合规过滤闸（关键词闸）。无外部依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.guard import append_disclaimer, sanitize  # noqa: E402


def test_sanitize_blocks_buy_advice():
    text = "建议买入该股票，未来一周将会上涨"
    cleaned, hits = sanitize(text)
    assert "建议买入" not in cleaned
    assert "未来一周将会上涨" not in cleaned
    assert len(hits) >= 2


def test_sanitize_blocks_must_rise():
    text = "这只股票必涨，稳赚不赔"
    cleaned, hits = sanitize(text)
    assert "必涨" not in cleaned
    assert "稳赚" not in cleaned
    assert len(hits) >= 2


def test_sanitize_clean_text_unchanged():
    text = "过去 7 个交易日累计上涨 5.2%，区间最高价为 110."
    cleaned, hits = sanitize(text)
    assert cleaned == text
    assert hits == []


def test_append_disclaimer_idempotent():
    text = "区间结束价 100 元。"
    once = append_disclaimer(text)
    twice = append_disclaimer(once)
    assert once == twice
    assert "不构成投资建议" in once
