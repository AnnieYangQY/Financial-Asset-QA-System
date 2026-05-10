"""
关键词闸（Guard）：
- 后处理 LLM 输出，过滤"预测未来 / 投资建议"类违禁词
- 命中后用 [已隐去] 替换，并在末尾追加合规免责声明
"""
from __future__ import annotations

import re

# 强禁词：直接命中即触发
HARD_BLOCKLIST = [
    r"建议(买入|卖出|加仓|减仓|清仓|抄底)",
    r"必(涨|跌|赚)",
    r"稳赚",
    r"包赚",
    r"一定会(涨|跌)",
]

# 软禁词：未来预测类
SOFT_BLOCKLIST = [
    r"未来(几天|一周|一月|半年|一年)?(将|会|有望|可能)?(上涨|下跌|大涨|大跌|继续上行|继续下行)",
    r"预计(明|后|下)?(天|周|月|日|年)?会?(涨|跌)",
    r"将会(上涨|下跌|大涨|大跌)",
]

DISCLAIMER = "\n\n> 本回答仅基于已发生的公开数据，不构成投资建议，不预测未来走势。"


def sanitize(text: str) -> tuple[str, list[str]]:
    """
    返回 (净化后文本, 命中词列表)。
    所有命中位置替换为 [已隐去：合规过滤]。
    """
    if not text:
        return text, []

    hits: list[str] = []
    cleaned = text

    for pat in HARD_BLOCKLIST + SOFT_BLOCKLIST:
        for m in re.finditer(pat, cleaned):
            hits.append(m.group(0))
        cleaned = re.sub(pat, "[已隐去：合规过滤]", cleaned)

    return cleaned, hits


def append_disclaimer(text: str) -> str:
    if DISCLAIMER.strip() in text:
        return text
    return text.rstrip() + DISCLAIMER
