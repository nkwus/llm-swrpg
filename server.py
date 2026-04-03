from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import get_settings
from rag import answer_question
from retrieval import get_chroma_collection, get_embedder

_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.embedder = get_embedder(settings)
    app.state.collection = get_chroma_collection(settings)
    yield


app = FastAPI(title="Star Wars RPG Chatbot", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        return ChatResponse(answer="Please ask a question.")

    answer = answer_question(
        question,
        app.state.embedder,
        app.state.collection,
        app.state.settings,
    )
    return ChatResponse(answer=answer)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text())
