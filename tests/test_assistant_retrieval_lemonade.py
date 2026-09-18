"""
Tests for Lemonade dense embedder and reranker adapters in assistant retrieval.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from returns.result import Failure, Success

from rai.assistant.retrieval_lemonade import (
    LemonadeDenseEmbedder,
    LemonadeReranker,
)


def test_lemonade_dense_embedder_success() -> None:
    embedder = LemonadeDenseEmbedder(model_name="nomic-embed-text-v1-GGUF")

    mock_resp = httpx.Response(
        200,
        json={"data": [{"embedding": [3.0, 4.0]}]},
        request=httpx.Request("POST", "http://127.0.0.1:13305/api/v1/embeddings"),
    )

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        vec = embedder.embed("user preference python")
        # 3.0 / 5.0 = 0.6, 4.0 / 5.0 = 0.8
        assert len(vec) == 2
        assert abs(vec[0] - 0.6) < 1e-5
        assert abs(vec[1] - 0.8) < 1e-5
        mock_post.assert_called_once()

        # Cached call should not hit API again
        vec2 = embedder.embed("user preference python")
        assert vec2 == vec
        assert mock_post.call_count == 1


def test_lemonade_dense_embedder_fallback_on_error() -> None:
    embedder = LemonadeDenseEmbedder()
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("Server down")):
        vec = embedder.embed("error test")
        assert len(vec) == 384
        assert all(v == 0.0 for v in vec)


@pytest.mark.asyncio
async def test_lemonade_reranker_success() -> None:
    reranker = LemonadeReranker(model_name="bge-reranker-v2-m3-GGUF")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        200,
        json={
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.35},
            ]
        },
        request=httpx.Request("POST", "http://127.0.0.1:13305/v1/reranking"),
    )
    mock_client.post.return_value = mock_resp

    with patch.object(reranker, "_get_client", return_value=mock_client):
        res = await reranker.rerank(
            query="Gdzie mieszkam?",
            documents=["Użytkownik lubi herbatę.", "Użytkownik mieszka w Warszawie."],
            top_n=2,
        )
        assert isinstance(res, Success)
        results = res.unwrap()
        assert len(results) == 2
        assert results[0]["index"] == 1
        assert results[0]["relevance_score"] == 0.95


@pytest.mark.asyncio
async def test_lemonade_reranker_empty_documents() -> None:
    reranker = LemonadeReranker()
    res = await reranker.rerank(query="test", documents=[])
    assert isinstance(res, Success)
    assert res.unwrap() == []


@pytest.mark.asyncio
async def test_lemonade_reranker_failure() -> None:
    reranker = LemonadeReranker()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        500,
        text="Internal Server Error in reranker",
        request=httpx.Request("POST", "http://127.0.0.1:13305/v1/reranking"),
    )
    mock_client.post.return_value = mock_resp

    with patch.object(reranker, "_get_client", return_value=mock_client):
        res = await reranker.rerank(
            query="test",
            documents=["doc 1"],
        )
        assert isinstance(res, Failure)
        assert res.failure().code == "RERANK_FAILED"
