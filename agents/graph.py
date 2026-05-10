"""
LangGraph 主图：

  +--------+    +-----------------+
  | router | -> | market_agent    |
  +--------+    +-----------------+
       |
       |        +-----------------+
       +-----> | rag_agent       |
       |        +-----------------+
       |
       |        +-----------------+
       +-----> | hybrid_agent    |
                +-----------------+

每个节点的输入 / 输出统一通过 GraphState 传递，最终返回标准 QAAnswer。
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from schemas import QAAnswer, RouterDecision

from . import hybrid_agent, market_agent, rag_agent
from .router import route


class GraphState(TypedDict, total=False):
    question: str
    decision: RouterDecision
    answer: QAAnswer


def _node_route(state: GraphState) -> GraphState:
    decision = route(state["question"])
    logger.info(
        f"[router] intent={decision.intent} symbols={decision.symbols} reason={decision.reason}"
    )
    return {"decision": decision}


def _node_market(state: GraphState) -> GraphState:
    decision: RouterDecision = state["decision"]
    return {"answer": market_agent.run(state["question"], decision.symbols)}


def _node_knowledge(state: GraphState) -> GraphState:
    return {"answer": rag_agent.run(state["question"])}


def _node_hybrid(state: GraphState) -> GraphState:
    decision: RouterDecision = state["decision"]
    return {"answer": hybrid_agent.run(state["question"], decision.symbols)}


def _branch(state: GraphState) -> str:
    return state["decision"].intent


def build() -> "object":
    graph = StateGraph(GraphState)
    graph.add_node("router", _node_route)
    graph.add_node("market", _node_market)
    graph.add_node("knowledge", _node_knowledge)
    graph.add_node("hybrid", _node_hybrid)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _branch,
        {
            "market": "market",
            "knowledge": "knowledge",
            "hybrid": "hybrid",
        },
    )
    graph.add_edge("market", END)
    graph.add_edge("knowledge", END)
    graph.add_edge("hybrid", END)

    return graph.compile()


_GRAPH: Optional[object] = None


def answer(question: str) -> QAAnswer:
    """对外的统一入口。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build()

    final = _GRAPH.invoke({"question": question})  # type: ignore[union-attr]
    return final["answer"]
