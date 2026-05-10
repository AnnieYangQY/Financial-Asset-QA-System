"""Router Agent：用 LLM 把用户问题分类为 market / knowledge / hybrid。"""
from __future__ import annotations

from loguru import logger

from config import load_prompt
from schemas import RouterDecision
from tools.symbol_resolver import extract_candidates

from .llm import chat_router, extract_json


_FALLBACK_KEYWORDS_MARKET = [
    "股价", "价格", "涨", "跌", "走势", "成交量", "市值",
    "k线", "K线", "现在多少", "几块"
]
_FALLBACK_KEYWORDS_KNOWLEDGE = [
    "什么是", "如何理解", "区别", "定义", "概念", "怎么算", "为何叫",
    "PE", "PB", "ROE", "EPS",
]


_HYBRID_TRIGGERS = ["原因", "为什么", "为何", "财报", "事件", "新闻", "解读", "影响"]


def _heuristic_route(question: str) -> RouterDecision:
    """LLM 不可用时的纯规则兜底。"""
    symbols = extract_candidates(question)
    has_market = any(k in question for k in _FALLBACK_KEYWORDS_MARKET)
    has_knowledge = any(k.lower() in question.lower() for k in _FALLBACK_KEYWORDS_KNOWLEDGE)
    has_hybrid_trigger = any(k in question for k in _HYBRID_TRIGGERS)

    if symbols and has_hybrid_trigger:
        intent = "hybrid"
    elif symbols and has_market and not has_knowledge:
        intent = "market"
    elif has_knowledge and not symbols:
        intent = "knowledge"
    elif symbols:
        intent = "market"
    else:
        intent = "knowledge"

    return RouterDecision(intent=intent, symbols=symbols, reason="heuristic-fallback")


def route(question: str) -> RouterDecision:
    """对外暴露的路由函数。"""
    try:
        prompt = load_prompt("router").replace("{question}", question)
        raw = chat_router(prompt)
        data = extract_json(raw)
        if not data or "intent" not in data:
            logger.warning(f"router LLM 输出异常: {raw[:120]}, 走兜底")
            return _heuristic_route(question)

        intent = data.get("intent", "knowledge")
        if intent not in ("market", "knowledge", "hybrid"):
            intent = "knowledge"

        symbols = data.get("symbols") or []
        # 兜底补齐：LLM 没抽到但启发式抽到也并入
        for cand in extract_candidates(question):
            if cand not in symbols:
                symbols.append(cand)

        return RouterDecision(
            intent=intent,
            symbols=symbols,
            reason=str(data.get("reason", ""))[:60],
        )
    except Exception as e:
        logger.warning(f"router 调用失败 {e}; 走兜底")
        return _heuristic_route(question)
