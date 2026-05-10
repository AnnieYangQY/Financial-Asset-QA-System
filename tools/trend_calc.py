"""
基于历史 OHLCV 的派生指标：
- compute_change(df, window)  → 某个窗口的涨跌幅
- compute_volatility(df)      → 波动率（收盘价标准差 / 均值）
- classify_trend(df)          → 上涨 / 下跌 / 震荡（基于线性回归斜率 + 振幅）
- summarize(df)               → 一次性产出适合喂给 LLM 的结构化字典

注意：所有函数只描述「已发生」的事实，不外推、不预测。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrendSummary:
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    high: float
    low: float
    change_pct: float           # (end - start) / start * 100
    max_drawdown_pct: float     # 区间内最大回撤
    volatility_pct: float       # 收盘价标准差 / 均值，百分制
    trend: str                  # "上涨" / "下跌" / "震荡"
    avg_volume: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_change(df: pd.DataFrame, window: int | None = None) -> float:
    """
    计算窗口涨跌幅（%）。window 为 None 时使用整个 df。
    """
    if df.empty:
        return 0.0
    sub = df.tail(window) if window else df
    start = float(sub["close"].iloc[0])
    end = float(sub["close"].iloc[-1])
    if start == 0:
        return 0.0
    return (end - start) / start * 100


def compute_volatility(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    close = df["close"].astype(float)
    if close.mean() == 0:
        return 0.0
    return float(close.std() / close.mean() * 100)


def compute_max_drawdown(df: pd.DataFrame) -> float:
    """区间内最大回撤（负数百分比，单位 %）。"""
    if df.empty:
        return 0.0
    close = df["close"].astype(float).values
    peak = np.maximum.accumulate(close)
    drawdown = (close - peak) / peak
    return float(drawdown.min() * 100)


def classify_trend(df: pd.DataFrame, threshold_pct: float = 3.0) -> str:
    """
    基于：
      1) 区间总涨跌幅是否超过阈值
      2) 收盘价线性回归斜率方向
    判定为 上涨 / 下跌 / 震荡。
    """
    if len(df) < 2:
        return "数据不足"

    change = compute_change(df)
    close = df["close"].astype(float).values
    x = np.arange(len(close))
    slope = float(np.polyfit(x, close, 1)[0])

    if change >= threshold_pct and slope > 0:
        return "上涨"
    if change <= -threshold_pct and slope < 0:
        return "下跌"
    return "震荡"


def summarize(df: pd.DataFrame) -> TrendSummary:
    """生成一份适合直接注入 prompt `<data>` 标签的结构化摘要。"""
    if df.empty:
        raise ValueError("空 DataFrame 无法生成摘要")

    return TrendSummary(
        start_date=str(df["date"].iloc[0]),
        end_date=str(df["date"].iloc[-1]),
        start_price=round(float(df["close"].iloc[0]), 4),
        end_price=round(float(df["close"].iloc[-1]), 4),
        high=round(float(df["high"].max()), 4),
        low=round(float(df["low"].min()), 4),
        change_pct=round(compute_change(df), 2),
        max_drawdown_pct=round(compute_max_drawdown(df), 2),
        volatility_pct=round(compute_volatility(df), 2),
        trend=classify_trend(df),
        avg_volume=round(float(df["volume"].mean()), 0),
    )
