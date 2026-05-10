"""统一的 LLM 调用入口。OpenAI 兼容协议（DeepSeek / OpenAI / Qwen / Ollama 都行）。"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

from loguru import logger
from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_ROUTER_MODEL,
    has_llm_credentials,
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not has_llm_credentials():
        raise RuntimeError(
            "未配置 LLM_API_KEY，请在 .env 中填写后再运行。"
            "见 .env.example 的说明。"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """单轮对话；返回纯文本。"""
    client = get_client()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_router(prompt: str) -> str:
    """路由用：低温度 + 较小 token 数 + 用 router model。"""
    return chat(
        prompt,
        model=LLM_ROUTER_MODEL,
        temperature=0.0,
        max_tokens=256,
    )


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """从 LLM 输出中提取首个 JSON 对象。"""
    if not text:
        return {}
    # 先尝试整段直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底：抓首个 {...}
    m = _JSON_BLOCK.search(text)
    if not m:
        logger.warning(f"未在 LLM 输出中找到 JSON：{text[:120]}")
        return {}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        logger.warning(f"JSON 解析失败：{e}")
        return {}
