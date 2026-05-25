"""OpenAI-compatible embedding provider (works with ModelArk and others)."""

from __future__ import annotations

import httpx

from app.providers.embedding_base import EmbeddingProvider

_shared_embedding_http_client: httpx.AsyncClient | None = None


def get_embedding_http_client(timeout: int) -> httpx.AsyncClient:
    global _shared_embedding_http_client
    if _shared_embedding_http_client is None or _shared_embedding_http_client.is_closed:
        _shared_embedding_http_client = httpx.AsyncClient(timeout=timeout)
    else:
        _shared_embedding_http_client.timeout = httpx.Timeout(timeout)
    return _shared_embedding_http_client


async def close_embedding_http_client() -> None:
    global _shared_embedding_http_client
    if _shared_embedding_http_client is not None and not _shared_embedding_http_client.is_closed:
        await _shared_embedding_http_client.aclose()
    _shared_embedding_http_client = None


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, api_key: str, model: str, dimension: int, timeout: int = 8):
        self.provider_name = "openai_compatible"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout

    async def embed_text(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model, "input": texts}

        client = get_embedding_http_client(self.timeout)
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = item["embedding"]
            if len(vec) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(vec)}"
                )
            embeddings.append(vec)
        return embeddings

    async def health_check(self) -> dict:
        try:
            vec = await self.embed_text("health check")
            return {"status": "ok", "dimension": len(vec)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
