"""RAG Agent：对 knowledge 类问题做向量检索 + LLM 组织答案 + 强制引用。"""
from __future__ import annotations

from loguru import logger

from config import load_prompt
from rag.retriever import retrieve
from schemas import Citation, QAAnswer
from tools.guard import append_disclaimer, sanitize

from .llm import chat


def _format_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[来源 {i}]（{c.title or c.source}）\n{c.text.strip()}\n"
        )
    return "\n".join(parts)


def run(question: str) -> QAAnswer:
    chunks = retrieve(question, top_k=5)
    if not chunks:
        return QAAnswer(
            intent="knowledge",
            question=question,
            analysis=append_disclaimer(
                "当前知识库中没有与此问题相关的内容。请先运行 `python -m rag.ingest` 构建向量库，"
                "或换个问法（例如：什么是市盈率 / 收入和净利润的区别）。"
            ),
        )

    context_block = _format_context(chunks)
    prompt = (
        load_prompt("rag_answer")
        .replace("{question}", question)
        .replace("{context_block}", context_block)
    )

    try:
        raw = chat(prompt, temperature=0.3, max_tokens=900)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return QAAnswer(
            intent="knowledge",
            question=question,
            error=f"LLM 不可用：{e}",
            citations=[
                Citation(title=c.title, source=c.source, snippet=c.text[:120])
                for c in chunks
            ],
        )

    cleaned, hits = sanitize(raw)
    cleaned = append_disclaimer(cleaned)

    return QAAnswer(
        intent="knowledge",
        question=question,
        analysis=cleaned,
        blocked_phrases=hits,
        citations=[
            Citation(title=c.title, source=c.source, snippet=c.text[:160])
            for c in chunks
        ],
    )
