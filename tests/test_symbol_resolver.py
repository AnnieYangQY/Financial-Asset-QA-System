"""单元测试：symbol_resolver。无外部依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.symbol_resolver import extract_candidates, resolve  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected_market, expected_symbol",
    [
        ("阿里巴巴", "US", "BABA"),
        ("Alibaba", "US", "BABA"),
        ("BABA", "US", "BABA"),
        ("特斯拉", "US", "TSLA"),
        ("TSLA", "US", "TSLA"),
        ("贵州茅台", "A", "600519"),
        ("600519", "A", "600519"),
        ("000001", "A", "000001"),
        ("00700", "HK", "00700"),
        ("腾讯", "HK", "00700"),
    ],
)
def test_resolve_basic(raw, expected_market, expected_symbol):
    rs = resolve(raw)
    assert rs.market == expected_market
    assert rs.symbol == expected_symbol


def test_resolve_invalid():
    with pytest.raises(ValueError):
        resolve("不存在的公司XYZ")


def test_extract_candidates_mixed():
    text = "BABA 和 阿里巴巴 都涨了，对比 特斯拉 (TSLA) 和 600519"
    cands = extract_candidates(text)
    assert "阿里巴巴" in cands
    assert "BABA" in cands
    assert "特斯拉" in cands
    assert "TSLA" in cands
    assert "600519" in cands


def test_extract_candidates_filters_common_words():
    text = "AI 和 ETF 是什么意思"
    cands = extract_candidates(text)
    assert "AI" not in cands
    assert "ETF" not in cands
