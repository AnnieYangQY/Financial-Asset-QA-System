"""项目级数据 schema：路由结果、回答 payload。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Router ----------

Intent = Literal["market", "knowledge", "hybrid"]


class RouterDecision(BaseModel):
    intent: Intent
    symbols: list[str] = Field(default_factory=list)
    reason: str = ""


# ---------- Answer ----------

class FactItem(BaseModel):
    """客观数据条目。"""
    label: str
    value: str
    source: str = ""
    timestamp: str = ""


class Citation(BaseModel):
    title: str
    source: str           # 文件名或 URL
    snippet: str = ""     # 用于前端展示的小段摘录


class QAAnswer(BaseModel):
    """统一输出结构。前端按 facts / analysis / citations 分段渲染。"""
    intent: Intent
    question: str
    facts: list[FactItem] = Field(default_factory=list)
    analysis: str = ""              # 已经合规化（关键词闸过滤后）的 markdown
    citations: list[Citation] = Field(default_factory=list)
    blocked_phrases: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    disclaimer: str = "本回答仅基于已发生的公开数据，不构成投资建议，不预测未来走势。"
