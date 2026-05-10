"""
轻量 Web 搜索包装：用于 Hybrid Agent 召回新闻 / 财报摘要。
- 优先使用 Tavily（金融搜索质量好、有 API）
- 无 key 时返回空列表（hybrid agent 会自动回退）
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from loguru import logger

from config import TAVILY_API_KEY, has_tavily_credentials


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def search(query: str, k: int = 5) -> list[SearchResult]:
    """返回 Tavily 搜索结果列表。无 key 时返回 []。"""
    if not has_tavily_credentials():
        logger.info("未配置 TAVILY_API_KEY，跳过 web search")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(
            query=query,
            search_depth="basic",
            max_results=k,
            topic="news",
        )
        items = resp.get("results", []) if isinstance(resp, dict) else []
        return [
            SearchResult(
                title=it.get("title", ""),
                url=it.get("url", ""),
                content=it.get("content", "")[:500],
                score=it.get("score"),
            )
            for it in items
        ]
    except Exception as e:
        logger.warning(f"Tavily 调用失败: {e}")
        return []
