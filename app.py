"""
Gradio 入口：
- 左侧 ChatInterface 与用户对话
- 用 Markdown 富格式渲染 facts / analysis / citations
- 提供文档示例的快捷气泡，便于演示

启动：python app.py
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

import gradio as gr
from loguru import logger

from agents.graph import answer
from schemas import QAAnswer

EXAMPLES_TEXT = [
    "阿里巴巴当前股价是多少？",
    "BABA 最近 7 天涨跌情况如何？",
    "阿里巴巴最近为何 1 月 15 日大涨？",
    "特斯拉近期走势如何？",
    "什么是市盈率？",
    "收入和净利润的区别是什么？",
    "贵州茅台最近一个月走势怎么样？",
]


INTENT_LABEL = {
    "market": "行情类",
    "knowledge": "知识类",
    "hybrid": "混合类（行情 + 资料）",
}


def _render(qa: QAAnswer) -> str:
    """把 QAAnswer 序列化成 Markdown，方便 Gradio 渲染。"""
    parts: list[str] = []

    parts.append(
        f"**意图分类**：`{INTENT_LABEL.get(qa.intent, qa.intent)}`"
    )

    if qa.error:
        parts.append(f"\n> 出错：{qa.error}")
        if qa.facts:
            parts.append(_render_facts(qa.facts))
        return "\n\n".join(parts)

    if qa.facts:
        parts.append(_render_facts(qa.facts))

    if qa.analysis:
        parts.append(qa.analysis.strip())

    if qa.citations:
        parts.append("\n---\n**引用列表**\n")
        for i, c in enumerate(qa.citations, start=1):
            line = f"- [来源 {i}] **{c.title or c.source}** &nbsp;`{c.source}`"
            if c.snippet:
                line += f"\n  > {c.snippet.replace(chr(10), ' ')}"
            parts.append(line)

    if qa.blocked_phrases:
        parts.append(
            f"\n> 合规过滤命中 {len(qa.blocked_phrases)} 处："
            f"{', '.join(set(qa.blocked_phrases))}"
        )

    return "\n\n".join(parts)


def _render_facts(facts) -> str:
    head = "### 客观数据\n"
    rows = ["| 指标 | 数值 | 时间 / 区间 | 来源 |", "|---|---|---|---|"]
    for f in facts:
        rows.append(
            f"| {f.label} | {f.value} | {f.timestamp or '-'} | {f.source or '-'} |"
        )
    return head + "\n".join(rows)


def chat_fn(message: str, history: list[Any] | None = None) -> str:
    msg = (message or "").strip()
    if not msg:
        return "请输入一个问题，例如：「阿里巴巴最近 7 天涨跌情况？」或「什么是市盈率？」"
    try:
        qa = answer(msg)
        return _render(qa)
    except Exception as e:  # pragma: no cover - 展示用兜底
        logger.error(traceback.format_exc())
        return f"系统出错：{e}\n\n请检查 .env 中的 LLM_API_KEY，或先运行 `python -m rag.ingest` 构建知识库。"


def build_ui() -> gr.Blocks:
    css = """
    .gradio-container {max-width: 1100px !important;}
    """
    with gr.Blocks(title="金融资产问答系统", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            # 金融资产问答系统
            **基于 LangGraph 路由 Agent + LlamaIndex RAG + AKShare 行情** 的中文金融问答系统。
            行情类问题走真实 API，知识类问题走 RAG，混合类两路汇合；统一禁止预测未来。
            """
        )
        gr.ChatInterface(
            fn=chat_fn,
            type="messages",
            examples=EXAMPLES_TEXT,
            chatbot=gr.Chatbot(
                type="messages",
                height=560,
                show_copy_button=True,
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            ),
        )
        with gr.Accordion("使用说明 / 系统能力", open=False):
            gr.Markdown(
                """
                - 行情类示例：`阿里巴巴当前股价是多少？` `BABA 最近 7 天涨跌情况？`
                - 知识类示例：`什么是市盈率？` `收入和净利润的区别？`
                - 混合类示例：`阿里巴巴最近一季财报摘要是什么？`
                - 数据来源：AKShare（A 股 / 港股 / 美股 / 中概股） + yfinance 兜底，知识库为本地 markdown 向量化。
                - 本系统仅描述已发生的公开数据，**不构成投资建议、不预测未来走势**。
                """
            )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
    except KeyboardInterrupt:
        print("\n已退出")
        sys.exit(0)
