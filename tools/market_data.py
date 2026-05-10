"""
统一行情数据接口：AKShare 主源 + yfinance 兜底。

对外暴露 4 个主函数：
- get_quote(symbol)              → 当前/最近一日报价
- get_history(symbol, days)      → 最近 N 天 OHLCV
- get_history_range(symbol, ...) → 区间历史
- search_company_news(query, k)  → 财经新闻（可选，需 Tavily key）

所有函数返回的 DataFrame 列名统一为：
['date', 'open', 'high', 'low', 'close', 'volume']
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
from loguru import logger

from .symbol_resolver import ResolvedSymbol, resolve


@dataclass
class Quote:
    symbol: str
    name: str
    market: str
    price: float
    change_pct: float | None
    timestamp: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 简单 30 秒内存缓存，避免演示时反复打 AKShare
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 30.0


def _cached(key: str, builder):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return val
    val = builder()
    _CACHE[key] = (now, val)
    return val


# ----------------------- 历史 K 线 -----------------------

def _ak_history(rs: ResolvedSymbol, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    if rs.market == "A":
        df = ak.stock_zh_a_hist(
            symbol=rs.symbol, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
    elif rs.market == "HK":
        df = ak.stock_hk_hist(
            symbol=rs.symbol, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
    else:  # US
        df = ak.stock_us_hist(
            symbol=rs.symbol, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )

    if df is None or df.empty:
        raise RuntimeError(f"AKShare 返回空数据: {rs.symbol}")

    # AKShare 中文列名 → 标准英文列名
    rename_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _yf_history(rs: ResolvedSymbol, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        rs.yf_symbol,
        start=datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d"),
        end=(datetime.strptime(end, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 返回空数据: {rs.yf_symbol}")

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def get_history_range(symbol: str, start: str, end: str) -> tuple[ResolvedSymbol, pd.DataFrame]:
    """
    返回区间内的 OHLCV。AKShare 失败自动 fallback 到 yfinance。
    start / end 格式：YYYYMMDD
    """
    rs = resolve(symbol)
    cache_key = f"hist:{rs.market}:{rs.symbol}:{start}:{end}"

    def _fetch() -> pd.DataFrame:
        try:
            return _ak_history(rs, start, end)
        except Exception as e:
            logger.warning(f"AKShare 失败 {rs.symbol}: {e}; 切换 yfinance")
            return _yf_history(rs, start, end)

    return rs, _cached(cache_key, _fetch)


def get_history(symbol: str, days: int = 30) -> tuple[ResolvedSymbol, pd.DataFrame]:
    """最近 N 天历史（按自然日）。"""
    end = datetime.now()
    # 多取 30 天保险（节假日、停牌），后面再裁剪
    start = end - timedelta(days=days + 30)
    rs, df = get_history_range(symbol, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    # 取最后 N 个交易日
    if len(df) > days:
        df = df.tail(days).reset_index(drop=True)
    return rs, df


# ----------------------- 当前报价 -----------------------

def get_quote(symbol: str) -> Quote:
    """返回最近一个交易日的收盘报价；演示场景下视为"当前价"。"""
    rs, df = get_history(symbol, days=2)
    if df.empty:
        raise RuntimeError(f"无法获取 {rs.symbol} 的报价")

    last = df.iloc[-1]
    prev_close = df.iloc[-2]["close"] if len(df) >= 2 else last["close"]
    change_pct = (last["close"] - prev_close) / prev_close * 100 if prev_close else None

    return Quote(
        symbol=rs.symbol,
        name=rs.display,
        market=rs.market,
        price=float(last["close"]),
        change_pct=float(change_pct) if change_pct is not None else None,
        timestamp=str(last["date"]),
        source="AKShare/yfinance",
    )
