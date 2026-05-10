"""Market Agent：根据 symbols + 问题推断时间窗口，调行情 API，再让 LLM 组织答案。"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from config import load_prompt
from schemas import FactItem, QAAnswer
from tools import market_data, trend_calc
from tools.guard import append_disclaimer, sanitize

from .llm import chat


def _detect_window(question: str) -> int:
    """从用户问题中粗略抽取窗口（天）。默认 30 天。"""
    q = question
    m = re.search(r"(\d+)\s*[天日]", q)
    if m:
        return max(2, int(m.group(1)))
    if "一周" in q or "1周" in q or "近周" in q:
        return 7
    if "近一月" in q or "一个月" in q or "1月" in q or "本月" in q:
        return 30
    if "近三月" in q or "3个月" in q or "近半年" in q:
        return 90
    if "今年" in q or "年初至今" in q:
        return 365
    return 30


def _format_data_block(symbol: str, summary: dict, sample_rows: list[dict]) -> str:
    """生成喂给 LLM 的结构化文本块。"""
    lines = [
        f"标的: {symbol}",
        f"时间区间: {summary['start_date']} ~ {summary['end_date']}",
        f"区间起始价: {summary['start_price']}",
        f"区间结束价: {summary['end_price']}",
        f"区间最高: {summary['high']}",
        f"区间最低: {summary['low']}",
        f"区间涨跌幅(%): {summary['change_pct']}",
        f"最大回撤(%): {summary['max_drawdown_pct']}",
        f"波动率(%): {summary['volatility_pct']}",
        f"趋势分类: {summary['trend']}",
        f"日均成交量: {summary['avg_volume']:.0f}",
        "数据来源: AKShare（A 股 / 港股 / 美股 / 中概股）+ yfinance 兜底",
        "",
        "最近若干交易日 OHLCV（已截断）:",
    ]
    for r in sample_rows:
        lines.append(
            f"  {r['date']}: O={r['open']:.2f} H={r['high']:.2f} "
            f"L={r['low']:.2f} C={r['close']:.2f} V={int(r['volume']):,}"
        )
    return "\n".join(lines)


def _facts_from_summary(symbol: str, summary: dict) -> list[FactItem]:
    return [
        FactItem(label=f"{symbol} 区间起始价", value=str(summary["start_price"]),
                 source="AKShare/yfinance", timestamp=summary["start_date"]),
        FactItem(label=f"{symbol} 区间结束价", value=str(summary["end_price"]),
                 source="AKShare/yfinance", timestamp=summary["end_date"]),
        FactItem(label=f"{symbol} 区间涨跌幅(%)", value=str(summary["change_pct"]),
                 source="AKShare/yfinance",
                 timestamp=f"{summary['start_date']}~{summary['end_date']}"),
        FactItem(label=f"{symbol} 区间最高 / 最低",
                 value=f"{summary['high']} / {summary['low']}",
                 source="AKShare/yfinance",
                 timestamp=f"{summary['start_date']}~{summary['end_date']}"),
        FactItem(label=f"{symbol} 趋势分类", value=summary["trend"],
                 source="本系统 trend_calc.classify_trend"),
        FactItem(label=f"{symbol} 波动率(%)", value=str(summary["volatility_pct"]),
                 source="AKShare/yfinance"),
    ]


def run(question: str, symbols: list[str]) -> QAAnswer:
    """主入口。失败时返回 error 字段。"""
    if not symbols:
        return QAAnswer(
            intent="market",
            question=question,
            error="未识别到股票符号；请在问题中明确公司名 / 代码。",
        )

    days = _detect_window(question)
    facts: list[FactItem] = []
    data_blocks: list[str] = []

    for sym in symbols[:3]:  # 最多处理前 3 只，避免 prompt 过长
        try:
            rs, df = market_data.get_history(sym, days=days)
        except Exception as e:
            logger.warning(f"行情获取失败 {sym}: {e}")
            facts.append(FactItem(label=f"{sym} 行情获取失败", value=str(e)))
            continue

        if df.empty:
            continue
        summary = trend_calc.summarize(df).to_dict()
        sample = df.tail(min(10, len(df))).to_dict("records")
        data_blocks.append(_format_data_block(rs.display, summary, sample))
        facts.extend(_facts_from_summary(rs.display, summary))

    if not data_blocks:
        return QAAnswer(
            intent="market",
            question=question,
            error="所有标的的行情数据获取失败，请稍后再试或检查网络。",
        )

    prompt = (
        load_prompt("market_answer")
        .replace("{question}", question)
        .replace("{data_block}", "\n\n".join(data_blocks))
    )

    try:
        raw = chat(prompt, temperature=0.2, max_tokens=900)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return QAAnswer(
            intent="market",
            question=question,
            facts=facts,
            error=f"LLM 不可用：{e}",
        )

    cleaned, hits = sanitize(raw)
    cleaned = append_disclaimer(cleaned)

    return QAAnswer(
        intent="market",
        question=question,
        facts=facts,
        analysis=cleaned,
        blocked_phrases=hits,
    )
