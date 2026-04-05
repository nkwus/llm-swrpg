from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config import Settings
from server import app

_TEST_SETTINGS = Settings(
    chroma_dir="chroma",
    collection_name="star_wars_rpg",
    embedding_model="all-MiniLM-L6-v2",
    groq_api_key="test-api-key",
    groq_model="llama-3.3-70b-versatile",
    retrieval_k=15,
)


@pytest.fixture()
def client(tmp_path: pytest.TempPathFactory) -> TestClient:
    """
    Return a TestClient with the lifespan bypassed.
    We populate app.state manually so no real ChromaDB or embedder is needed.
    """
    app.state.settings = _TEST_SETTINGS
    app.state.embedder = MagicMock()
    app.state.collection = MagicMock()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_chat(tmp_path: pytest.TempPathFactory) -> TestClient:
    """Client where chat service has not been started."""
    app.state.settings = _TEST_SETTINGS
    app.state.embedder = None
    app.state.collection = None
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_index_contains_chat_ui_elements(client: TestClient) -> None:
    response = client.get("/")
    body = response.text
    assert "<form" in body
    assert "api/chat" in body


# ---------------------------------------------------------------------------
# GET /admin
# ---------------------------------------------------------------------------


def test_admin_returns_html(client: TestClient) -> None:
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Admin" in response.text


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


def test_status_reports_not_ready(client_no_chat: TestClient) -> None:
    response = client_no_chat.get("/api/status")
    assert response.json() == {"chat_ready": False}


def test_status_reports_ready(client: TestClient) -> None:
    response = client.get("/api/status")
    assert response.json() == {"chat_ready": True}


# ---------------------------------------------------------------------------
# POST /api/chat — happy path
# ---------------------------------------------------------------------------


def test_chat_returns_answer(client: TestClient) -> None:
    with patch("server.answer_question", return_value="Agility governs initiative."):
        response = client.post("/api/chat", json={"question": "How does initiative work?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Agility governs initiative."}


def test_chat_passes_question_to_answer_question(client: TestClient) -> None:
    with patch("server.answer_question", return_value="answer") as mock_aq:
        client.post("/api/chat", json={"question": "What is a blaster?"})

    mock_aq.assert_called_once()
    assert mock_aq.call_args[0][0] == "What is a blaster?"


def test_chat_strips_whitespace_from_question(client: TestClient) -> None:
    with patch("server.answer_question", return_value="answer") as mock_aq:
        client.post("/api/chat", json={"question": "  padded question  "})

    assert mock_aq.call_args[0][0] == "padded question"


# ---------------------------------------------------------------------------
# POST /api/chat — not started
# ---------------------------------------------------------------------------


def test_chat_returns_not_running_when_service_not_started(client_no_chat: TestClient) -> None:
    response = client_no_chat.post("/api/chat", json={"question": "test"})
    assert response.status_code == 200
    assert "not running" in response.json()["answer"].lower()


# ---------------------------------------------------------------------------
# POST /api/chat — empty question
# ---------------------------------------------------------------------------


def test_chat_empty_question_returns_prompt(client: TestClient) -> None:
    with patch("server.answer_question") as mock_aq:
        response = client.post("/api/chat", json={"question": ""})

    assert response.status_code == 200
    assert "Please ask a question" in response.json()["answer"]
    mock_aq.assert_not_called()


def test_chat_whitespace_only_question_returns_prompt(client: TestClient) -> None:
    with patch("server.answer_question") as mock_aq:
        response = client.post("/api/chat", json={"question": "   "})

    assert response.status_code == 200
    mock_aq.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/chat — malformed request
# ---------------------------------------------------------------------------


def test_chat_missing_question_field_returns_422(client: TestClient) -> None:
    response = client.post("/api/chat", json={"not_a_question": "hello"})
    assert response.status_code == 422


def test_chat_non_json_body_returns_422(client: TestClient) -> None:
    response = client.post("/api/chat", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
