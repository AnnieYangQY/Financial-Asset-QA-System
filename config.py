"""项目级配置：统一加载 .env、暴露常量、构造 LLM client。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---- LLM ----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_ROUTER_MODEL = os.getenv("LLM_ROUTER_MODEL", LLM_MODEL)

# ---- Web search ----
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---- Embedding & Chroma ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", ".chroma_db").lstrip("./"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "finance_kb")

# ---- 知识库语料路径 ----
KB_DATA_DIR = PROJECT_ROOT / "rag" / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


@lru_cache(maxsize=1)
def has_llm_credentials() -> bool:
    return bool(LLM_API_KEY) and LLM_API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


@lru_cache(maxsize=1)
def has_tavily_credentials() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "tvly-xxxxxxxxxxxxxxxxxxxxxxxx"


def load_prompt(name: str) -> str:
    """Load a prompt template by file name (without extension)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
