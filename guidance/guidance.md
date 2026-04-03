# Star Wars RPG Expert Chatbot (RAG) — Complete Setup Guide

> A step-by-step guide for **Linux** using [**uv**](https://docs.astral.sh/uv/) for dependency management.

---

## Table of Contents

- [Star Wars RPG Expert Chatbot (RAG) — Complete Setup Guide](#star-wars-rpg-expert-chatbot-rag--complete-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Why RAG?](#why-rag)
  - [What You Will Build](#what-you-will-build)
  - [Prerequisites](#prerequisites)
  - [Project Setup](#project-setup)
    - [Install uv](#install-uv)
    - [Create the Project](#create-the-project)
    - [Add Dependencies](#add-dependencies)
    - [Optional Configuration (.env)](#optional-configuration-env)
  - [Install and Configure Ollama](#install-and-configure-ollama)
    - [Install Ollama](#install-ollama)
    - [Pull a model](#pull-a-model)
    - [Start the Ollama server](#start-the-ollama-server)
  - [Script 1: Populate the Vector Database](#script-1-populate-the-vector-database)
    - [Run the ingestion](#run-the-ingestion)
  - [Script 2: Query the Chatbot](#script-2-query-the-chatbot)
    - [Run the chatbot](#run-the-chatbot)
  - [Prompt Design Tips](#prompt-design-tips)
    - [Example Prompts](#example-prompts)
  - [Troubleshooting](#troubleshooting)
    - [1. No text extracted from PDFs](#1-no-text-extracted-from-pdfs)
    - [2. `ModuleNotFoundError`](#2-modulenotfounderror)
    - [3. Ollama connection errors](#3-ollama-connection-errors)
    - [4. Slow first run](#4-slow-first-run)
    - [5. Out of memory / system slowdown](#5-out-of-memory--system-slowdown)
    - [6. Hallucinated / incorrect answers](#6-hallucinated--incorrect-answers)
  - [Next Steps](#next-steps)
  - [Appendix: Complete File Listings](#appendix-complete-file-listings)
    - [`pyproject.toml`](#pyprojecttoml)
    - [`.env`](#env)
    - [`populate_database.py`](#populate_databasepy)
    - [`query_chat.py`](#query_chatpy)

---

## Introduction

This guide walks you through building a **local, private** Star Wars RPG expert chatbot that answers questions about your PDF rulebooks and sourcebooks. It uses **Retrieval-Augmented Generation (RAG)**: your documents stay external and searchable, and a local language model generates answers from the retrieved passages.

Every step is explicit — copy and paste the commands into your terminal. If something fails, check the [Troubleshooting](#troubleshooting) section.

---

## Why RAG?

RAG separates knowledge storage from the language model. The model does not need retraining to use your PDFs — instead, the system finds relevant passages and feeds them to the model at query time.

**Pros:**

- **No model training required** — no expensive GPUs or long training runs.
- **Works with copyrighted PDFs** — text is retrieved at runtime, not baked into model weights.
- **Easy to update** — add or remove PDFs and re-run ingestion.
- **Runs locally and privately** — you control all data.
- **Beginner friendly** — simple Python scripts and widely used libraries.

**Limitations:**

- **Retrieval quality matters** — poor chunking or embeddings lead to poor answers.
- **Answers limited to your documents** — the model should say "I don't know" when the answer isn't in context.
- **Scanned PDFs need OCR** — image-based PDFs require an OCR step before ingestion.
- **Local model resource usage** — running an LLM locally requires available RAM/VRAM.

---

## What You Will Build

A command-line application that:

1. Reads all PDFs in a `data/` folder.
2. Extracts text and splits it into overlapping chunks.
3. Creates embeddings and stores them in a local **ChromaDB** collection.
4. Accepts user questions, retrieves the most relevant chunks, and sends them plus the question to a local LLM via **Ollama**.
5. Prints the answer in the terminal.

Example questions you can ask:

- *How does initiative work in this system?*
- *What are the rules for Force powers?*
- *How do I calculate damage for blaster rifles?*

---

## Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.12+** | Runtime | `sudo apt install python3` (or your distro's package manager) |
| **uv** | Python project & dependency manager | See [Install uv](#install-uv) below |
| **Ollama** | Local LLM runtime | See [Install and Configure Ollama](#install-and-configure-ollama) below |
| **VS Code** (recommended) | Editor | [code.visualstudio.com](https://code.visualstudio.com/) |
| **Git** (optional) | Version control | `sudo apt install git` |

---

## Project Setup

### Install uv

If you don't already have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal (or run `source ~/.bashrc` / `source ~/.zshrc`) so the `uv` command is on your PATH, then verify:

```bash
uv --version
```

### Create the Project

```bash
mkdir llm-swrpg && cd llm-swrpg
uv init
```

This creates a `pyproject.toml`, a `.python-version` file, and a basic project structure. Then create the data directories:

```bash
mkdir -p data chroma
```

Your project layout will look like this:

```text
llm-swrpg/
├── data/                    # Put your Star Wars RPG PDFs here
├── chroma/                  # ChromaDB persistent storage (auto-populated)
├── populate_database.py     # Ingestion script
├── query_chat.py            # Chat script
├── pyproject.toml           # Project metadata & dependencies
└── .env                     # Optional configuration
```

### Add Dependencies

Use `uv` to add the required packages. This automatically creates a virtual environment (`.venv/`) and installs everything:

```bash
uv add pypdf chromadb sentence-transformers requests python-dotenv rich
```

> **Note:** You never need to manually create or activate a venv. When you run scripts with `uv run`, it automatically uses the project's virtual environment.

To run any script in this project, use:

```bash
uv run python <script_name>.py
```

### Optional Configuration (.env)

Create a `.env` file in the project root to store configuration values:

```bash
cat > .env << 'EOF'
CHROMA_DIR=chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
OLLAMA_MODEL=mistral
EOF
```

The scripts read these values using `python-dotenv`. If you skip this file, the same defaults are used automatically.

---

## Install and Configure Ollama

Ollama runs LLMs locally and provides a simple HTTP API.

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull a model

```bash
ollama pull mistral
```

> This downloads model files (~4 GB for Mistral). Make sure you have enough disk space.

### Start the Ollama server

Ollama typically runs as a systemd service after installation. Verify it's running:

```bash
curl http://localhost:11434
```

If you get a response, Ollama is ready. If not, start it manually:

```bash
ollama serve
```

> **Tip:** If your machine has limited RAM, try a smaller model: `ollama pull tinyllama`.

---

## Script 1: Populate the Vector Database

This script extracts text from each PDF, splits it into overlapping chunks, computes embeddings, and stores everything in ChromaDB.

**Key design choices:**

- **Chunk size:** 1000 characters with 200-character overlap — preserves context across chunk boundaries.
- **Embedding model:** `all-MiniLM-L6-v2` — fast, compact, and effective for semantic search.
- **Vector DB:** ChromaDB — runs locally, persists data to `chroma/`.

Create `populate_database.py`:

```python
# populate_database.py
import os
from pathlib import Path

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = Path("data")


def extract_text_from_pdf(pdf_path: Path) -> str:
    print(f"[bold blue]Reading PDF:[/bold blue] {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[red]Error reading page {i} in {pdf_path.name}: {e}[/red]")
            text = ""
        pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main() -> None:
    print(f"[bold green]Loading embedding model:[/bold green] {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"[bold green]Initializing ChromaDB at:[/bold green] {CHROMA_DIR}")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection("star_wars_rpg")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[red]No PDF files found in data/ folder. Add your Star Wars PDFs there.[/red]")
        return

    doc_id_counter = 0

    for pdf_path in pdf_files:
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            print(f"[yellow]Warning: No text extracted from {pdf_path.name}. "
                  f"It may be scanned or image-based.[/yellow]")
            continue

        chunks = chunk_text(full_text, size=1000, overlap=200)
        print(f"[cyan]Created {len(chunks)} chunks from {pdf_path.name}[/cyan]")

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

    print("[bold green]Done! ChromaDB is populated.[/bold green]")


if __name__ == "__main__":
    main()
```

### Run the ingestion

```bash
uv run python populate_database.py
```

You should see progress messages for each PDF and the number of chunks created.

> **No text extracted?** The PDF may contain scanned images. See [Troubleshooting](#troubleshooting) for OCR options.

---

## Script 2: Query the Chatbot

This script accepts user questions, retrieves relevant chunks from ChromaDB, builds a prompt with the retrieved context, calls the local LLM via Ollama, and prints the answer.

Create `query_chat.py`:

```python
# query_chat.py
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print
import requests

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def get_chroma_collection() -> Any:
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("star_wars_rpg")


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def retrieve_relevant_chunks(
    query: str, embedder: SentenceTransformer, collection: Any, k: int = 5
) -> list[str]:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    documents: list[list[str]] = results.get("documents", [[]])
    return documents[0]


def build_prompt(context_chunks: list[str], question: str) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a Star Wars roleplaying game rules and lore expert.\n\n"
        "Use ONLY the information in the context below to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer as a helpful GM assistant, clearly and concisely:"
    )


def call_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[red]Error calling Ollama: {e}[/red]")
        return "Error: could not reach the local LLM. Is Ollama running?"

    data: dict[str, str] = response.json()
    return data.get("response", "").strip()


def chat_loop() -> None:
    print("[bold green]Star Wars RPG Chatbot[/bold green]")
    print("Type your question, or 'quit' to exit.\n")

    embedder = get_embedder()
    collection = get_chroma_collection()

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("[bold blue]Goodbye! May the Force be with you.[/bold blue]")
            break
        if not question:
            continue

        print("[cyan]Searching your Star Wars PDFs...[/cyan]")
        chunks = retrieve_relevant_chunks(question, embedder, collection, k=5)
        if not chunks:
            print("[red]No relevant information found in the database.[/red]")
            continue

        prompt = build_prompt(chunks, question)
        print("[cyan]Asking the local LLM...[/cyan]")
        answer = call_ollama(prompt)
        print(f"[bold magenta]Bot:[/bold magenta] {answer}\n")


if __name__ == "__main__":
    chat_loop()
```

### Run the chatbot

Before running, make sure:

1. Dependencies are installed (`uv sync` if needed).
2. Ollama is running and the model is pulled.
3. You have run `populate_database.py` at least once.

```bash
uv run python query_chat.py
```

Type a question and press Enter. Type `quit` to exit.

---

## Prompt Design Tips

The `build_prompt` function already instructs the model to use only the provided context. Additional tips:

- **Be explicit about the role:** *"You are a Star Wars RPG rules expert and GM."*
- **Request citations:** *"Cite the source filename for any rule you reference."*
- **Limit creativity for rules questions:** Ask for concise, factual answers.

### Example Prompts

**Rules clarification:**

```text
Use the context to explain how initiative is determined in the Star Wars RPG.
If the rules conflict, explain both and cite the source filenames.
```

**Character build help:**

```text
Use the context to recommend a starting build for a Force-sensitive scout.
Explain choices and reference the rule pages.
```

**Mechanics comparison:**

```text
Compare the damage rules for blaster pistols and blaster rifles using only the context.
Provide a short summary table.
```

---

## Troubleshooting

### 1. No text extracted from PDFs

**Symptom:** `Warning: No text extracted from <file>. It may be scanned or image-based.`

**Fix:** The PDF contains scanned images. Run OCR to convert images to text:

```bash
sudo apt install tesseract-ocr
# Then use a Python wrapper like pytesseract in a preprocessing step
```

### 2. `ModuleNotFoundError`

**Symptom:** `ModuleNotFoundError: No module named 'pypdf'`

**Fix:** Make sure you're running scripts with `uv run`:

```bash
uv run python populate_database.py
```

If you recently added a dependency, sync first:

```bash
uv sync
```

### 3. Ollama connection errors

**Symptom:** `Error calling Ollama: ... could not reach the local LLM.`

**Fix:**

- Ensure Ollama is running: `ollama serve` or `systemctl status ollama`.
- Ensure the model is pulled: `ollama pull mistral`.
- Test connectivity: `curl http://localhost:11434`.

### 4. Slow first run

**Symptom:** First run takes a long time.

**Cause:** `sentence-transformers` downloads the embedding model on first use, and Ollama may need time to load the model into memory. Subsequent runs will be faster.

### 5. Out of memory / system slowdown

**Fix:**

- Reduce chunk size or reduce `k` (number of retrieved chunks).
- Use a smaller embedding model.
- Use a smaller Ollama model (e.g., `tinyllama`).
- If you have a GPU, Ollama will use it automatically.

### 6. Hallucinated / incorrect answers

**Fix:**

- The prompt already constrains the model to context. If answers are still off:
  - Reduce `k` to retrieve fewer, more relevant chunks.
  - Improve chunking (split on paragraphs/headings instead of fixed character counts).
  - Add explicit instructions: *"If the context does not contain the answer, respond with 'I don't know.'"*

---

## Next Steps

Once the basic system is working, consider these improvements:

- **Web UI** — Add a frontend with Gradio or Streamlit.
- **Source citations** — Show which PDF and chunk the answer came from.
- **Smarter chunking** — Split on paragraphs or headings instead of fixed character counts.
- **OCR pipeline** — Integrate Tesseract for scanned PDFs.
- **Embedding cache** — Skip re-embedding unchanged documents.
- **Conversation memory** — Maintain context across multiple turns.

---

## Appendix: Complete File Listings

### `pyproject.toml`

After running `uv init` and `uv add ...`, your `pyproject.toml` should contain:

```toml
[project]
name = "llm-swrpg"
version = "0.1.0"
description = "Star Wars RPG RAG chatbot"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pypdf",
    "chromadb",
    "sentence-transformers",
    "requests",
    "python-dotenv",
    "rich",
]
```

### `.env`

```text
CHROMA_DIR=chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
OLLAMA_MODEL=mistral
```

### `populate_database.py`

See [Script 1](#script-1-populate-the-vector-database) above.

### `query_chat.py`

See [Script 2](#script-2-query-the-chatbot) above.
