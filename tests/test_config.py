import os

import pytest

from config import Settings, get_settings


def test_get_settings_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() uses sensible defaults when no env vars are set."""
    for var in ("CHROMA_DIR", "COLLECTION_NAME", "EMBEDDING_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "RETRIEVAL_K"):
        monkeypatch.delenv(var, raising=False)

    s = get_settings()

    assert s.chroma_dir == "chroma"
    assert s.collection_name == "star_wars_rpg"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.groq_api_key == ""
    assert s.groq_model == "llama-3.3-70b-versatile"
    assert s.retrieval_k == 15


def test_get_settings_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() picks up overrides from environment variables."""
    monkeypatch.setenv("CHROMA_DIR", "/tmp/mydb")
    monkeypatch.setenv("COLLECTION_NAME", "genesys_rpg")
    monkeypatch.setenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    monkeypatch.setenv("GROQ_API_KEY", "sk-test-123")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    monkeypatch.setenv("RETRIEVAL_K", "25")

    s = get_settings()

    assert s.chroma_dir == "/tmp/mydb"
    assert s.collection_name == "genesys_rpg"
    assert s.embedding_model == "all-mpnet-base-v2"
    assert s.groq_api_key == "sk-test-123"
    assert s.groq_model == "llama-3.1-70b-versatile"
    assert s.retrieval_k == 25


def test_settings_is_immutable() -> None:
    """Settings is a frozen dataclass — fields cannot be mutated after creation."""
    s = Settings(
        chroma_dir="chroma",
        collection_name="star_wars_rpg",
        embedding_model="all-MiniLM-L6-v2",
        groq_api_key="sk-test",
        groq_model="llama-3.3-70b-versatile",
        retrieval_k=15,
    )
    with pytest.raises(Exception):
        s.groq_model = "changed"  # type: ignore[misc]


def test_retrieval_k_is_int(monkeypatch: pytest.MonkeyPatch) -> None:
    """RETRIEVAL_K env var string is coerced to int."""
    monkeypatch.setenv("RETRIEVAL_K", "42")
    s = get_settings()
    assert isinstance(s.retrieval_k, int)
    assert s.retrieval_k == 42
