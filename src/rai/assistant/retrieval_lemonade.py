"""
Lemonade-powered embedding and reranking adapters for assistant memory retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any, Optional

import httpx
from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass

from .ports import MemoryQuery, MemoryRetrievalSelection
from .records import MemoryRecord, make_assistant_failure
from .retrieval import (
    DenseEmbeddingProvider,
    MultiChannelMemoryRetriever,
    _memory_text,
)

logger = logging.getLogger(__name__)
DEFAULT_LEMONADE_HOST = "http://127.0.0.1:13305"


class LemonadeDenseEmbedder(DenseEmbeddingProvider):
    """Dense embedding provider using Lemonade Server's /api/v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v1-GGUF",
        host: str = DEFAULT_LEMONADE_HOST,
        api_key: Optional[str] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._version = f"lemonade-embed-{model_name}"
        self._cache: dict[str, tuple[float, ...]] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(self, text: str) -> tuple[float, ...]:
        """Synchronously compute or retrieve embedding vector."""
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with httpx.Client(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=self.timeout_seconds,
            ) as client:
                resp = client.post(
                    "/api/v1/embeddings",
                    json={"model": self.model_name, "input": text},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data and "embedding" in data[0]:
                        raw_vec = data[0]["embedding"]
                        norm = math.sqrt(sum(x * x for x in raw_vec))
                        norm_vec = (
                            tuple(x / norm for x in raw_vec)
                            if norm > 0.0
                            else tuple(raw_vec)
                        )
                        self._cache[cache_key] = norm_vec
                        return norm_vec
        except Exception as exc:  # noqa: BLE001
            logger.debug("Lemonade embedding request failed: %s", exc)

        # Fallback to zero vector if server call fails
        return tuple([0.0] * 384)


class LemonadeReranker:
    """Neural reranker client using Lemonade Server's /v1/reranking endpoint."""

    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3-GGUF",
        host: str = DEFAULT_LEMONADE_HOST,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
        return self._client

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> Result[list[dict[str, Any]], ActionFailure]:
        """Rerank a list of documents against a query string."""
        if not documents:
            return Success([])

        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        try:
            client = self._get_client()
            resp = await client.post("/v1/reranking", json=payload)
            if resp.status_code != 200:
                return Failure(
                    make_assistant_failure(
                        code="RERANK_FAILED",
                        message=f"Lemonade reranking failed with HTTP {resp.status_code}: {resp.text}",
                    )
                )

            data = resp.json()
            results = data.get("results", [])
            # Standard rerank response format: [{"index": int, "relevance_score": float}, ...]
            return Success(results)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RERANK_UNAVAILABLE",
                    message=f"Lemonade reranking network error: {exc}",
                )
            )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to close Lemonade reranker client: %s", exc)
            self._client = None
