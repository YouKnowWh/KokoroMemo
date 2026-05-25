"""Shared helpers for protocol compatibility routes."""

from __future__ import annotations

import logging
import time

import httpx

from app.core.state import get_config

logger = logging.getLogger("kokoromemo.protocol_common")

# Cache LiteLLM models for 60 seconds
_litellm_models_cache: dict = {"models": [], "fetched_at": 0}
_CACHE_TTL = 60.0


def format_model_item(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "kokoromemo",
    }


def _fetch_models_from_litellm() -> list[str]:
    """Fetch available models from LiteLLM, with caching."""
    global _litellm_models_cache
    now = time.monotonic()
    if _litellm_models_cache["models"] and (now - _litellm_models_cache["fetched_at"]) < _CACHE_TTL:
        return _litellm_models_cache["models"]

    cfg = get_config()
    base_url = cfg.llm.base_url.rstrip("/")
    api_key = cfg.llm.get_api_key()
    if not base_url or not api_key:
        return []

    try:
        r = httpx.get(f"{base_url}/models", headers={
            "Authorization": f"Bearer {api_key}",
        }, timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            all_models = [m["id"] for m in data.get("data", [])]
            # Filter out embedding/reranker models
            _NON_CHAT_PATTERNS = {"embed", "rerank", "bge", "qwen"}
            chat_models = [
                m for m in all_models
                if not any(p in m.lower() for p in _NON_CHAT_PATTERNS)
            ]
            _litellm_models_cache = {"models": chat_models, "fetched_at": now}
            logger.info("Fetched %d chat models from LiteLLM (filtered from %d)", len(chat_models), len(all_models))
            return chat_models
    except Exception as e:
        logger.warning("Failed to fetch models from LiteLLM: %s", e)

    return _litellm_models_cache["models"]


def get_exposed_models() -> list[str]:
    cfg = get_config()
    exposed_models = [model for model in cfg.compatibility.exposed_models if model]
    if exposed_models:
        return exposed_models
    # No explicit list configured: dynamically fetch from LiteLLM
    fetched = _fetch_models_from_litellm()
    if fetched:
        return fetched
    # Last resort: config default
    return [cfg.llm.model] if cfg.llm.model else []
