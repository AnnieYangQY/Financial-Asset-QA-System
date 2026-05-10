"""Hybrid Agent：行情 + RAG + Web 搜索三路汇合。"""
from __future__ import annotations

from loguru import logger

from config import load_prompt
from rag.retriever import retrieve
from schemas import Citation, QAAnswer
from tools import market_data, trend_calc, web_search
from tools.guard import append_disclaimer, sanitize

from .llm import chat
from .market_agent import _detect_window, _facts_from_summary, _format_data_block


def run(question: str, symbols: list[str]) -> QAAnswer:
    if not symbols:
        return QAAnswer(
            intent="hybrid",
            question=question,
            error="未识别到股票符号；混合查询需要明确公司名或代码。",
        )

    days = _detect_window(question)

    # 1) 行情数据
    facts = []
    data_blocks = []
    for sym in symbols[:2]:
        try:
            rs, df = market_data.get_history(sym, days=days)
        except Exception as e:
            logger.warning(f"行情获取失败 {sym}: {e}")
            continue
        if df.empty:
            continue
        summary = trend_calc.summarize(df).to_dict()
        sample = df.tail(min(10, len(df))).to_dict("records")
        data_blocks.append(_format_data_block(rs.display, summary, sample))
        facts.extend(_facts_from_summary(rs.display, summary))

    # 2) RAG（知识库 + 财报摘要）
    rag_chunks = retrieve(question, top_k=4)

    # 3) Web search（可选）
    web_query = f"{symbols[0]} {question}"
    news = web_search.search(web_query, k=3)

    # 拼接 context
    context_parts = []
    citations: list[Citation] = []
    idx = 1
    for c in rag_chunks:
        context_parts.append(f"[来源 {idx}]（{c.title or c.source}）\n{c.text.strip()}\n")
        citations.append(Citation(title=c.title, source=c.source, snippet=c.text[:160]))
        idx += 1
    for n in news:
        context_parts.append(f"[来源 {idx}]（{n.title} | {n.url}）\n{n.content.strip()}\n")
        citations.append(Citation(title=n.title, source=n.url, snippet=n.content[:160]))
        idx += 1

    if not data_blocks and not context_parts:
        return QAAnswer(
            intent="hybrid",
            question=question,
            error="未能获取行情数据，且知识库与新闻均无相关内容。",
        )

    data_block = "\n\n".join(data_blocks) if data_blocks else "（无可用行情数据）"
    context_block = "\n".join(context_parts) if context_parts else "（无可用资料）"

    prompt = (
        load_prompt("hybrid_answer")
        .replace("{question}", question)
        .replace("{data_block}", data_block)
        .replace("{context_block}", context_block)
    )

    try:
        raw = chat(prompt, temperature=0.3, max_tokens=1200)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return QAAnswer(
            intent="hybrid",
            question=question,
            facts=facts,
            citations=citations,
            error=f"LLM 不可用：{e}",
        )

    cleaned, hits = sanitize(raw)
    cleaned = append_disclaimer(cleaned)

    return QAAnswer(
        intent="hybrid",
        question=question,
        facts=facts,
        analysis=cleaned,
        citations=citations,
        blocked_phrases=hits,
    )
