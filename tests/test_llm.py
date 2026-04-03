from unittest.mock import MagicMock, patch

import pytest
import requests

from config import Settings
from llm import build_prompt, call_ollama

SETTINGS = Settings(
    chroma_dir="chroma",
    collection_name="star_wars_rpg",
    embedding_model="all-MiniLM-L6-v2",
    ollama_url="http://localhost:11434",
    ollama_model="mistral",
    retrieval_k=15,
)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_question() -> None:
    prompt = build_prompt(["some context"], "How does initiative work?")
    assert "How does initiative work?" in prompt


def test_build_prompt_contains_all_chunks() -> None:
    chunks = ["chunk one", "chunk two", "chunk three"]
    prompt = build_prompt(chunks, "test question")
    for chunk in chunks:
        assert chunk in prompt


def test_build_prompt_separates_chunks_with_delimiter() -> None:
    chunks = ["alpha", "beta"]
    prompt = build_prompt(chunks, "q")
    assert "---" in prompt


def test_build_prompt_contains_system_instruction() -> None:
    prompt = build_prompt(["ctx"], "q")
    assert "Star Wars roleplaying game" in prompt
    assert "ONLY the information in the context" in prompt


# ---------------------------------------------------------------------------
# call_ollama — successful response
# ---------------------------------------------------------------------------


def test_call_ollama_returns_response_text() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The Force is strong."}
    mock_response.raise_for_status.return_value = None

    with patch("llm.requests.post", return_value=mock_response) as mock_post:
        result = call_ollama("some prompt", SETTINGS)

    assert result == "The Force is strong."
    mock_post.assert_called_once()


def test_call_ollama_posts_to_correct_url() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_response.raise_for_status.return_value = None

    with patch("llm.requests.post", return_value=mock_response) as mock_post:
        call_ollama("prompt", SETTINGS)

    call_args = mock_post.call_args
    assert call_args[0][0] == "http://localhost:11434/api/generate"


def test_call_ollama_sends_correct_model() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_response.raise_for_status.return_value = None

    with patch("llm.requests.post", return_value=mock_response) as mock_post:
        call_ollama("prompt", SETTINGS)

    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "mistral"
    assert payload["stream"] is False


def test_call_ollama_strips_whitespace_from_response() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "  answer with spaces  "}
    mock_response.raise_for_status.return_value = None

    with patch("llm.requests.post", return_value=mock_response):
        result = call_ollama("prompt", SETTINGS)

    assert result == "answer with spaces"


# ---------------------------------------------------------------------------
# call_ollama — error handling
# ---------------------------------------------------------------------------


def test_call_ollama_returns_error_message_on_connection_failure() -> None:
    with patch("llm.requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
        result = call_ollama("prompt", SETTINGS)

    assert "Error" in result
    assert "Ollama running" in result


def test_call_ollama_returns_error_message_on_timeout() -> None:
    with patch("llm.requests.post", side_effect=requests.exceptions.Timeout()):
        result = call_ollama("prompt", SETTINGS)

    assert "Error" in result


def test_call_ollama_handles_missing_response_key() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None

    with patch("llm.requests.post", return_value=mock_response):
        result = call_ollama("prompt", SETTINGS)

    assert result == ""
