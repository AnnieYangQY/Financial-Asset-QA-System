"""
股票符号解析器：把用户输入的「公司名 / 代码」统一解析为
(market, normalized_symbol, display_name) 三元组。

规则：
- 6 位纯数字           → A 股
- 5 位纯数字           → 港股
- 字母（含 . 后缀）      → 美股 / 中概股
- 中文 / 英文公司名      → 查映射表，否则报错
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Market = Literal["A", "HK", "US"]


@dataclass
class ResolvedSymbol:
    market: Market
    symbol: str        # 用于 AKShare 接口的代码（如 "600519" / "00700" / "BABA"）
    display: str       # 展示用名称（如 "贵州茅台 (600519)"）
    yf_symbol: str     # yfinance 兜底用的代码（如 "600519.SS" / "0700.HK" / "BABA"）


# 文档示例 + 常见高频公司，覆盖 vibe coding 演示场景
NAME_MAP: dict[str, tuple[Market, str, str]] = {
    # 中文名 / 英文名 -> (market, ak_symbol, display_name)
    "阿里巴巴": ("US", "BABA", "阿里巴巴"),
    "alibaba": ("US", "BABA", "阿里巴巴"),
    "baba": ("US", "BABA", "阿里巴巴"),
    "特斯拉": ("US", "TSLA", "特斯拉"),
    "tesla": ("US", "TSLA", "特斯拉"),
    "tsla": ("US", "TSLA", "特斯拉"),
    "苹果": ("US", "AAPL", "苹果"),
    "apple": ("US", "AAPL", "苹果"),
    "aapl": ("US", "AAPL", "苹果"),
    "微软": ("US", "MSFT", "微软"),
    "microsoft": ("US", "MSFT", "微软"),
    "msft": ("US", "MSFT", "微软"),
    "英伟达": ("US", "NVDA", "英伟达"),
    "nvidia": ("US", "NVDA", "英伟达"),
    "nvda": ("US", "NVDA", "英伟达"),
    "谷歌": ("US", "GOOGL", "谷歌"),
    "google": ("US", "GOOGL", "谷歌"),
    "亚马逊": ("US", "AMZN", "亚马逊"),
    "amazon": ("US", "AMZN", "亚马逊"),
    "京东": ("US", "JD", "京东"),
    "拼多多": ("US", "PDD", "拼多多"),
    "网易": ("US", "NTES", "网易"),
    "百度": ("US", "BIDU", "百度"),
    # A 股
    "贵州茅台": ("A", "600519", "贵州茅台"),
    "宁德时代": ("A", "300750", "宁德时代"),
    "比亚迪": ("A", "002594", "比亚迪"),
    "招商银行": ("A", "600036", "招商银行"),
    "平安银行": ("A", "000001", "平安银行"),
    "中国平安": ("A", "601318", "中国平安"),
    # 港股
    "腾讯": ("HK", "00700", "腾讯控股"),
    "腾讯控股": ("HK", "00700", "腾讯控股"),
    "tencent": ("HK", "00700", "腾讯控股"),
    "美团": ("HK", "03690", "美团"),
    "小米": ("HK", "01810", "小米集团"),
    "小米集团": ("HK", "01810", "小米集团"),
}


_RE_DIGITS = re.compile(r"^\d+$")
_RE_LETTERS = re.compile(r"^[A-Z][A-Z0-9.\-]*$")


def _to_yf(market: Market, symbol: str) -> str:
    """生成 yfinance 兜底所需的 ticker 写法。"""
    if market == "US":
        return symbol
    if market == "HK":
        # AKShare 港股是 5 位（00700），yfinance 用 4 位 + .HK（0700.HK）
        return f"{symbol.lstrip('0').zfill(4)}.HK"
    # A 股
    if symbol.startswith(("60", "68", "5", "90")):
        return f"{symbol}.SS"  # 上交所
    return f"{symbol}.SZ"      # 深交所


def resolve(query: str) -> ResolvedSymbol:
    """把任意用户输入解析为标准 ResolvedSymbol，无法解析则抛 ValueError。"""
    q = query.strip()
    if not q:
        raise ValueError("空输入")

    # 1) 中文名 / 英文名映射
    key = q.lower()
    if key in NAME_MAP:
        market, sym, display = NAME_MAP[key]
        return ResolvedSymbol(market, sym, f"{display} ({sym})", _to_yf(market, sym))

    # 2) 数字代码
    if _RE_DIGITS.match(q):
        if len(q) == 6:
            return ResolvedSymbol("A", q, q, _to_yf("A", q))
        if len(q) == 5:
            return ResolvedSymbol("HK", q, q, _to_yf("HK", q))
        if len(q) == 4:
            # 港股有时被简写为 4 位
            sym = q.zfill(5)
            return ResolvedSymbol("HK", sym, sym, _to_yf("HK", sym))

    # 3) 字母代码（美股）
    upper = q.upper()
    if _RE_LETTERS.match(upper):
        return ResolvedSymbol("US", upper, upper, upper)

    raise ValueError(
        f"无法解析符号 '{query}'。请使用公司中/英文名（如 阿里巴巴/Alibaba/BABA）"
        f"或直接输入代码（A股 6 位、港股 5 位、美股字母）。"
    )


def extract_candidates(text: str) -> list[str]:
    """
    从一段自然语言里挖掘可能的股票符号 / 公司名。
    用于 router / agent 的兜底，避免依赖 LLM 100% 抽取。
    """
    text_low = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    # 公司名直接子串匹配（按长度倒序，优先匹配长名）
    for name in sorted(NAME_MAP.keys(), key=len, reverse=True):
        if name in text_low and name not in seen:
            found.append(name)
            seen.add(name)

    # 字母 ticker（≥2 个大写字母，避免误抓"AI"等通用词）
    for m in re.finditer(r"\b([A-Z]{2,5})\b", text):
        token = m.group(1)
        if token not in seen and token not in {"AI", "API", "GDP", "CPI", "PPI", "ETF", "IPO", "ROE", "ROA", "EPS"}:
            found.append(token)
            seen.add(token)

    # 6 位数字（A 股）/ 5 位数字（港股）
    for m in re.finditer(r"\b(\d{5,6})\b", text):
        token = m.group(1)
        if token not in seen:
            found.append(token)
            seen.add(token)

    return found
