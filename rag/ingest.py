"""
RAG 入库脚本：
- 读取 rag/data/ 下所有 .md 文件
- 用 LlamaIndex 进行 markdown 标题分块
- 用 HuggingFace 嵌入模型向量化
- 持久化到 Chroma

执行：python -m rag.ingest
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    KB_DATA_DIR,
)


def _read_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(KB_DATA_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(KB_DATA_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
        docs.append(
            Document(
                text=text,
                metadata={
                    "source": str(rel),
                    "title": path.stem,
                    "category": category,
                },
            )
        )
    logger.info(f"读取 {len(docs)} 篇文档（{KB_DATA_DIR}）")
    return docs


def build(reset: bool = False) -> None:
    if reset and Path(CHROMA_PERSIST_DIR).exists():
        logger.warning(f"重置向量库 {CHROMA_PERSIST_DIR}")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = MarkdownNodeParser(include_metadata=True)

    documents = _read_documents()
    if not documents:
        logger.error("未找到任何文档，请检查 rag/data/ 目录")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    n = collection.count()
    logger.success(
        f"向量库已构建：{n} 个 chunk，持久化目录 {CHROMA_PERSIST_DIR}"
    )
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="先清空再重建")
    args = parser.parse_args()
    build(reset=args.reset)
