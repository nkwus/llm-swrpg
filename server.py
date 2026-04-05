from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import get_settings
from rag import answer_question

_STATIC = Path(__file__).parent / "static"
_DATA = Path(__file__).parent / "data"


@dataclass
class ProcessProgress:
    running: bool = False
    current_file: str = ""
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    chunks_total: int = 0
    phase: str = "idle"
    error: str = ""

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "running": self.running,
            "phase": self.phase,
            "current_file": self.current_file,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "chunks_done": self.chunks_done,
            "chunks_total": self.chunks_total,
            "error": self.error,
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = get_settings()
    app.state.embedder = None
    app.state.collection = None
    app.state.progress = ProcessProgress()
    _DATA.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Star Wars RPG Chatbot", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Admin – Upload PDFs
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_pdf(file: UploadFile) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are accepted."}, status_code=400)
    safe_name = Path(file.filename).name  # strip any path components
    dest = _DATA / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse({"filename": safe_name, "size": len(content)})


@app.get("/api/pdfs")
def list_pdfs() -> list[dict[str, str | int]]:
    if not _DATA.exists():
        return []
    return [
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(_DATA.glob("*.pdf"))
    ]


@app.post("/api/delete-pdf")
def delete_pdf(body: dict[str, str]) -> JSONResponse:
    name = Path(body.get("name", "")).name
    target = _DATA / name
    if not target.exists() or not target.suffix == ".pdf":
        return JSONResponse({"error": "File not found."}, status_code=404)
    target.unlink()
    return JSONResponse({"deleted": name})


# ---------------------------------------------------------------------------
# Admin – Process / Ingest
# ---------------------------------------------------------------------------


@app.post("/api/process")
def process_pdfs() -> JSONResponse:
    progress: ProcessProgress = app.state.progress
    if progress.running:
        return JSONResponse({"error": "Processing is already running."}, status_code=409)

    pdf_files = list(_DATA.glob("*.pdf"))
    if not pdf_files:
        return JSONResponse({"error": "No PDF files found. Upload some first."}, status_code=400)

    def _run() -> None:
        from populate_database import chunk_text, extract_text_from_pdf
        from retrieval import get_embedder, get_or_create_chroma_collection

        try:
            progress.running = True
            progress.error = ""
            progress.files_total = len(pdf_files)
            progress.files_done = 0
            progress.chunks_done = 0
            progress.chunks_total = 0

            progress.phase = "Loading embedding model…"
            settings = app.state.settings
            embedder = get_embedder(settings)
            collection = get_or_create_chroma_collection(settings)

            # First pass: extract text and count total chunks
            progress.phase = "Extracting text from PDFs…"
            file_chunks: list[tuple[Path, list[str]]] = []
            for pdf_path in pdf_files:
                progress.current_file = pdf_path.name
                full_text = extract_text_from_pdf(pdf_path)
                if not full_text.strip():
                    progress.files_done += 1
                    continue
                chunks = chunk_text(full_text, size=1000, overlap=200)
                file_chunks.append((pdf_path, chunks))
                progress.chunks_total += len(chunks)
                progress.files_done += 1

            # Second pass: embed and store
            progress.phase = "Embedding chunks…"
            progress.files_done = 0
            progress.files_total = len(file_chunks)
            doc_id_counter = 0

            for pdf_path, chunks in file_chunks:
                progress.current_file = pdf_path.name
                for chunk in chunks:
                    embedding = embedder.encode(chunk).tolist()
                    doc_id = f"{pdf_path.name}-{doc_id_counter}"
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[doc_id],
                        metadatas=[{"source": pdf_path.name}],
                    )
                    doc_id_counter += 1
                    progress.chunks_done += 1
                progress.files_done += 1

            progress.phase = "done"
            progress.current_file = ""
        except Exception as e:
            progress.error = str(e)
            progress.phase = "error"
        finally:
            progress.running = False

    thread = Thread(target=_run, daemon=True)
    thread.start()
    return JSONResponse({"started": True, "files": len(pdf_files)})


@app.get("/api/process/progress")
def process_progress() -> dict[str, str | int | bool]:
    return app.state.progress.to_dict()


# ---------------------------------------------------------------------------
# Admin – Start Chat Service
# ---------------------------------------------------------------------------


@app.post("/api/start-chat")
def start_chat() -> JSONResponse:
    from retrieval import get_chroma_collection, get_embedder

    settings = app.state.settings
    try:
        app.state.embedder = get_embedder(settings)
        app.state.collection = get_chroma_collection(settings)
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to start chat service: {e}. Process PDFs first."},
            status_code=500,
        )
    return JSONResponse({"status": "Chat service is running."})


@app.get("/api/status")
def status() -> dict[str, bool]:
    return {"chat_ready": app.state.collection is not None}


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if app.state.embedder is None or app.state.collection is None:
        return ChatResponse(answer="Chat service is not running. Go to the admin page and start it first.")

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse((_STATIC / "admin.html").read_text())


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text())
