from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from llm import build_prompt, call_groq

SETTINGS = Settings(
    chroma_dir="chroma",
    collection_name="star_wars_rpg",
    embedding_model="all-MiniLM-L6-v2",
    groq_api_key="test-api-key",
    groq_model="llama-3.3-70b-versatile",
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
# call_groq — successful response
# ---------------------------------------------------------------------------


def test_call_groq_returns_response_text() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "The Force is strong."

    with patch("llm.Groq", return_value=mock_client) as mock_groq_class:
        result = call_groq("some prompt", SETTINGS)

    assert result == "The Force is strong."
    mock_groq_class.assert_called_once_with(api_key="test-api-key")


def test_call_groq_sends_correct_model() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"

    with patch("llm.Groq", return_value=mock_client):
        call_groq("prompt", SETTINGS)

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_call_groq_strips_whitespace_from_response() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "  answer with spaces  "

    with patch("llm.Groq", return_value=mock_client):
        result = call_groq("prompt", SETTINGS)

    assert result == "answer with spaces"


def test_call_groq_returns_empty_string_when_content_is_none() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = None

    with patch("llm.Groq", return_value=mock_client):
        result = call_groq("prompt", SETTINGS)

    assert result == ""


# ---------------------------------------------------------------------------
# call_groq — error handling
# ---------------------------------------------------------------------------


def test_call_groq_returns_error_message_on_connection_failure() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("connection refused")

    with patch("llm.Groq", return_value=mock_client):
        result = call_groq("prompt", SETTINGS)

    assert "Error" in result
    assert "Groq" in result


def test_call_groq_returns_error_message_on_api_error() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("rate limited")

    with patch("llm.Groq", return_value=mock_client):
        result = call_groq("prompt", SETTINGS)

    assert "Error" in result
