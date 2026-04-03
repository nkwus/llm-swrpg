from unittest.mock import MagicMock, patch

from config import Settings
from rag import answer_question

SETTINGS = Settings(
    chroma_dir="chroma",
    collection_name="star_wars_rpg",
    embedding_model="all-MiniLM-L6-v2",
    ollama_url="http://localhost:11434",
    ollama_model="mistral",
    retrieval_k=15,
)


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
    return embedder


def _make_collection(docs: list[str]) -> MagicMock:
    collection = MagicMock()
    collection.query.return_value = {"documents": [docs]}
    return collection


# ---------------------------------------------------------------------------
# answer_question — happy path
# ---------------------------------------------------------------------------


def test_answer_question_returns_llm_response() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["Relevant rulebook text about initiative."])

    with patch("rag.call_ollama", return_value="Initiative is determined by Agility."):
        result = answer_question("How does initiative work?", embedder, collection, SETTINGS)

    assert result == "Initiative is determined by Agility."


def test_answer_question_passes_question_to_retrieval() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["some context"])

    with patch("rag.call_ollama", return_value="answer"), \
         patch("rag.retrieve_relevant_chunks", return_value=["some context"]) as mock_retrieve:
        answer_question("What is a lightsaber?", embedder, collection, SETTINGS)

    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args[0][0] == "What is a lightsaber?"


def test_answer_question_uses_retrieval_k_from_settings() -> None:
    embedder = _make_embedder()
    collection = _make_collection(["ctx"])
    settings = Settings(
        chroma_dir="chroma",
        collection_name="star_wars_rpg",
        embedding_model="all-MiniLM-L6-v2",
        ollama_url="http://localhost:11434",
        ollama_model="mistral",
        retrieval_k=20,
    )

    with patch("rag.call_ollama", return_value="answer"), \
         patch("rag.retrieve_relevant_chunks", return_value=["ctx"]) as mock_retrieve:
        answer_question("q", embedder, collection, settings)

    assert mock_retrieve.call_args[1]["k"] == 20


# ---------------------------------------------------------------------------
# answer_question — no chunks found
# ---------------------------------------------------------------------------


def test_answer_question_returns_fallback_when_no_chunks() -> None:
    embedder = _make_embedder()
    collection = _make_collection([])

    with patch("rag.call_ollama") as mock_llm:
        result = answer_question("What is a krayt dragon?", embedder, collection, SETTINGS)

    assert "couldn't find" in result.lower()
    mock_llm.assert_not_called()


def test_answer_question_does_not_call_ollama_when_no_chunks() -> None:
    embedder = _make_embedder()
    collection = MagicMock()
    collection.query.return_value = {"documents": [[]]}

    with patch("rag.call_ollama") as mock_llm:
        answer_question("q", embedder, collection, SETTINGS)

    mock_llm.assert_not_called()
