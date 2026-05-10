"""
Hybrid Retriever：
- Dense 向量检索（Chroma + bge）
- BM25 关键词检索
- RRF 融合（QueryFusionRetriever）

输出统一为 (text, source, score) 列表，方便 prompt 注入。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "title": self.title,
        }


def _ensure_embed_model() -> None:
    if Settings.embed_model is None or not isinstance(
        Settings.embed_model, HuggingFaceEmbedding
    ):
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _build_retriever() -> QueryFusionRetriever:
    if not Path(CHROMA_PERSIST_DIR).exists():
        raise RuntimeError(
            f"未找到向量库 {CHROMA_PERSIST_DIR}，请先运行 `python -m rag.ingest`"
        )

    _ensure_embed_model()

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=Settings.embed_model
    )

    vector_retriever = index.as_retriever(similarity_top_k=5)
    nodes = list(index.docstore.docs.values())
    if not nodes:
        logger.warning("docstore 为空，BM25 退化为仅向量召回")
        return vector_retriever  # type: ignore[return-value]

    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    fusion = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,           # 不做 query rewrite，节省 LLM 调用
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )
    return fusion


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """对外的统一检索入口。"""
    try:
        retriever = _build_retriever()
    except Exception as e:
        logger.error(f"检索器初始化失败: {e}")
        return []

    try:
        nodes: list[NodeWithScore] = retriever.retrieve(query)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return []

    results: list[RetrievedChunk] = []
    for node in nodes[:top_k]:
        meta = node.node.metadata or {}
        results.append(
            RetrievedChunk(
                text=node.node.get_content(),
                source=meta.get("source", "unknown"),
                title=meta.get("title", ""),
                score=float(node.score) if node.score is not None else 0.0,
            )
        )
    return results
