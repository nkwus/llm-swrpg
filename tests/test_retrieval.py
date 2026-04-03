from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from retrieval import retrieve_relevant_chunks

SETTINGS = Settings(
    chroma_dir="chroma",
    collection_name="star_wars_rpg",
    embedding_model="all-MiniLM-L6-v2",
    ollama_url="http://localhost:11434",
    ollama_model="mistral",
    retrieval_k=15,
)


def _make_embedder(vector: list[float] | None = None) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns a fixed vector."""
    embedder = MagicMock()
    embedder.encode.return_value = MagicMock(tolist=lambda: vector or [0.1, 0.2, 0.3])
    return embedder


def _make_collection(docs: list[str]) -> MagicMock:
    """Return a mock ChromaDB collection that returns the given documents."""
    collection = MagicMock()
    collection.query.return_value = {"documents": [docs]}
    return collection


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks
# ---------------------------------------------------------------------------


def test_retrieve_returns_list_of_strings() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["chunk a", "chunk b"])

    result = retrieve_relevant_chunks("test query", embedder, collection, k=2)

    assert result == ["chunk a", "chunk b"]


def test_retrieve_calls_encode_with_query() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["chunk"])

    retrieve_relevant_chunks("How does initiative work?", embedder, collection, k=1)

    embedder.encode.assert_called_once_with("How does initiative work?")


def test_retrieve_passes_k_to_collection_query() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["a", "b", "c"])

    retrieve_relevant_chunks("q", embedder, collection, k=3)

    collection.query.assert_called_once()
    call_kwargs = collection.query.call_args[1]
    assert call_kwargs["n_results"] == 3


def test_retrieve_returns_empty_list_when_no_documents() -> None:
    embedder = _make_embedder()
    collection = MagicMock()
    collection.query.return_value = {"documents": [[]]}

    result = retrieve_relevant_chunks("q", embedder, collection, k=5)

    assert result == []


def test_retrieve_returns_empty_list_when_documents_key_missing() -> None:
    embedder = _make_embedder()
    collection = MagicMock()
    collection.query.return_value = {}

    result = retrieve_relevant_chunks("q", embedder, collection, k=5)

    assert result == []
