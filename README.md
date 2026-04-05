# Star Wars RPG Chatbot

An AI chatbot that answers questions about Star Wars tabletop RPG rulebooks. It uses **Retrieval-Augmented Generation (RAG)**: your PDF rulebooks are indexed into a local vector database, and a cloud LLM (via the [Groq API](https://groq.com)) generates answers using only the text found in those PDFs.

The app includes a **web-based admin panel** where you can upload PDFs, process them into the vector database, and start the chat service — all from your browser. No command-line interaction is required after initial setup.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [First-Time Setup](#first-time-setup)
- [Configuration](#configuration)
- [Using the Admin Panel](#using-the-admin-panel)
- [Running the Web App](#running-the-web-app)
- [Running the Command-Line Interface](#running-the-command-line-interface)
- [Day-to-Day Operations](#day-to-day-operations)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Extending the App](#extending-the-app)
- [Testing](#testing)
- [Dependency Reference](#dependency-reference)

---

## How It Works

Understanding the architecture will help you operate and extend this app.

### The RAG Pipeline

Traditional chatbots are limited to knowledge baked into their training data. RAG solves this by adding a retrieval step before the language model is called. Here is what happens every time a user asks a question:

```text
User question
      │
      ▼
1. EMBED the question
   (convert the question text into a vector — a list of numbers
    that represents its meaning)
      │
      ▼
2. SEARCH ChromaDB
   (find the N most semantically similar text chunks from your PDFs)
      │
      ▼
3. BUILD a prompt
   (inject the retrieved chunks as context into a structured prompt)
      │
      ▼
4. CALL Groq API
   (send the prompt to the cloud LLM, which generates an answer
    using only the provided context)
      │
      ▼
5. RETURN the answer to the user
```

### The Ingestion Pipeline

Before the chatbot can answer anything, the PDF rulebooks must be ingested. You can do this from the **admin panel** (recommended) or by running `populate_database.py` from the command line. Either way, the process:

```text
PDF files in data/
      │
      ▼
1. Extract all text from each PDF page
      │
      ▼
2. Split text into overlapping chunks (1000 chars, 200-char overlap)
   (overlap ensures that sentences spanning a chunk boundary
    are not lost)
      │
      ▼
3. Embed each chunk using sentence-transformers
      │
      ▼
4. Store chunks + embeddings in ChromaDB (persisted to disk)
```

The ChromaDB database is stored in the `chroma/` directory and persists between runs.

---

## Project Structure

```text
llm-swrpg/
│
├── config.py              # All configuration — reads .env, single source of truth
├── retrieval.py           # ChromaDB client and embedding model (data layer)
├── llm.py                 # Prompt construction and Groq API call (LLM interface)
├── rag.py                 # answer_question() — orchestrates retrieval → LLM
│
├── server.py              # FastAPI web server — routes, admin API, progress tracking
├── populate_database.py   # CLI ingestion script — reads PDFs, fills ChromaDB
├── query_chat.py          # Terminal chat interface
│
├── static/
│   ├── index.html         # Chat UI (plain HTML/CSS/JS, no framework)
│   └── admin.html         # Admin panel — upload PDFs, process, start chat service
│
├── data/                  # PDF rulebooks — uploaded via admin panel or copied manually (not committed)
├── chroma/                # ChromaDB vector database (generated, not committed)
│
├── .env                   # Local configuration overrides (not committed)
├── pyproject.toml         # Python project definition and dependencies
└── uv.lock                # Locked dependency versions for reproducible installs
```

### Why is the code split this way?

Each file has a single responsibility so that changes in one area do not break others:

- **`config.py`** — env vars are read in exactly one place. If you add a new setting, this is the only file you touch.
- **`retrieval.py`** — if you switch from ChromaDB to a different vector store (e.g. pgvector), you only change this file.
- **`llm.py`** — if you swap Groq for a different LLM provider (e.g. OpenAI, Anthropic), you only change this file.
- **`rag.py`** — the orchestration logic. Calls retrieval and llm but knows nothing about HTTP or the terminal.
- **`server.py`** — HTTP concerns only. Hosts the chat API, admin API (upload, process, start), and progress tracking.

---

## Prerequisites

You need the following installed before you begin.

### Python 3.12+

Check your version:

```bash
python3 --version
```

If you need to install it, use your system package manager (e.g. `sudo apt install python3` on Ubuntu/Debian).

### uv

`uv` is a fast Python package and project manager. It replaces `pip`, `virtualenv`, and `pip-tools` with a single tool. It manages the `.venv` for you automatically.

Install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal so `uv` is on your PATH.

### Groq API Key

This app uses the [Groq API](https://console.groq.com/) for LLM inference. You need a free API key:

1. Sign up at [console.groq.com](https://console.groq.com/)
2. Go to **API Keys** and create a new key
3. Copy the key — you will add it to your `.env` file in the next section

---

## First-Time Setup

### 1. Clone the repository

```bash
git clone https://github.com/nkwus/llm-swrpg.git
cd llm-swrpg
```

### 2. Install Python dependencies

`uv` will create a virtual environment and install all dependencies automatically:

```bash
uv sync
```

You do not need to activate the virtual environment manually. All `uv run` commands below use it automatically.

### 3. Create your `.env` file

Copy the example values:

```bash
cp .env.example .env   # if an example exists, otherwise create it manually
```

Or create `.env` from scratch:

```text
GROQ_API_KEY=your-api-key-here
GROQ_MODEL=llama-3.3-70b-versatileQuestions like "which vehicle has the highest speed?" or "how many weapons have the Stun quality?" will produce unreliable answers. RAG retrieves only the N most relevant chunks — it cannot scan the entire database to find a maximum or count all matching entries. For these questions, rephrase to be specific: "What is the speed stat of a T-70 X-Wing?" works well.
CHROMA_DIR=chroma
COLLECTION_NAME=star_wars_rpg
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVAL_K=15
```

The only required value is `GROQ_API_KEY`. All other settings have sensible defaults. See [Configuration](#configuration) for what each setting does.

---

## Configuration

All configuration is managed through environment variables, loaded from `.env` by `config.py`. You should never need to change a value by editing Python source code.

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | *(none)* | **Required.** Your Groq API key. Obtain one from [console.groq.com](https://console.groq.com/). |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | The Groq model to use for generating answers. See [Groq’s model list](https://console.groq.com/docs/models) for options. |
| `CHROMA_DIR` | `chroma` | Path to the directory where ChromaDB stores its data. Relative to the project root. |
| `COLLECTION_NAME` | `star_wars_rpg` | The name of the ChromaDB collection. You could use separate collections for different RPG systems. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | The sentence-transformers model used to create embeddings. Must be the same model for both ingestion and querying. **Do not change this after ingestion without re-processing your PDFs.** |
| `RETRIEVAL_K` | `15` | How many text chunks to retrieve from ChromaDB per question. Higher values give the LLM more context but make responses slower. See [Known Limitations](#known-limitations) for guidance. |

> **Important:** The `EMBEDDING_MODEL` must be identical between ingestion and query time. If you change the embedding model, you must delete the `chroma/` directory and re-process your PDFs.

---

## Using the Admin Panel

The admin panel at **<http://localhost:8000/admin>** provides a three-step workflow to get the chatbot operational. No command-line interaction is needed.

### Step 1 — Upload PDFs

Drag and drop your Star Wars RPG PDF files onto the upload area, or click to browse. Uploaded files are stored in the `data/` directory. You can delete individual files from the list if needed.

### Step 2 — Process PDFs

Click **Process PDFs**. This:

1. Loads the embedding model (first run downloads it, ~90 MB)
2. Extracts text from each PDF
3. Splits text into overlapping chunks (1 000 chars, 200-char overlap)
4. Embeds each chunk and stores it in ChromaDB

A **progress bar** shows the current phase, file being processed, and chunk count. For a typical rulebook (400–600 pages), this takes 2–10 minutes depending on your hardware.

### Step 3 — Start Chat Service

Click **Start Chat Service**. This loads the embedding model and ChromaDB collection into memory so the chat API can serve requests. The status indicator will turn green when the service is ready.

Once the status shows “Chat service is running”, click the link to go to the **Chat** page.

### Alternative: CLI ingestion

If you prefer the command line, the original ingestion script still works:

```bash
cp "/path/to/Edge of the Empire Core Rulebook.pdf" data/
uv run python populate_database.py
```

After running ingestion via CLI, you still need to click **Start Chat Service** in the admin panel (or restart the server) to load the data.

---

## Running the Web App

### Start the web server

```bash
uv run uvicorn server:app --reload
```

- `server:app` — the `app` object inside `server.py`
- `--reload` — automatically restarts the server when you edit Python files (development mode; remove this in production)

The server starts light — no embedding model or database is loaded. Open your browser:

- **Admin panel:** <http://localhost:8000/admin> — upload PDFs, process them, and start the chat service
- **Chat UI:** <http://localhost:8000> — ask questions once the chat service is running

### Running on a different port

```bash
uv run uvicorn server:app --reload --port 8080
```

### Making it accessible on your local network

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Other devices on your network can then access it at `http://<your-ip>:8000`.

---

## Running the Command-Line Interface

If you prefer not to use the web interface, the terminal chatbot is still available:

```bash
uv run python query_chat.py
```

Type your question and press Enter. Type `quit` or `exit` to stop.

---

## Day-to-Day Operations

### Normal startup sequence

1. Start the server: `uv run uvicorn server:app --reload`
2. Open <http://localhost:8000/admin>
3. Upload PDFs (if not already uploaded), click **Process PDFs**, then **Start Chat Service**
4. Open <http://localhost:8000> to chat

### After pulling changes from git

```bash
git pull
uv sync          # installs any new dependencies
```

Then restart the server.

### After adding new PDFs

Upload them via the admin panel and click **Process PDFs** again. Then click **Start Chat Service** to reload the collection.

Alternatively, copy files into `data/` and run:

```bash
uv run python populate_database.py
```

### Switching to a different LLM

Update `.env`:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

Restart the server. No re-ingestion required — the embedding model is independent of the LLM. See [Groq’s model list](https://console.groq.com/docs/models) for available models.

### Re-building the vector database from scratch

If you change the embedding model or chunk settings, delete the database and re-process:

```bash
rm -rf chroma/
```

Then use the admin panel to process PDFs again, or run `uv run python populate_database.py`.

---

## Troubleshooting

### "Error: could not reach the Groq API"

The Groq API is unreachable or the API key is invalid.

1. Check that `GROQ_API_KEY` in `.env` is set to a valid key
2. Verify your key at [console.groq.com](https://console.groq.com/)
3. Check your internet connection — Groq is a cloud API

### "No relevant information found in the database"

The vector database is either empty or the query did not match any chunks.

1. Check the `chroma/` directory exists and is not empty
2. Re-run ingestion: `uv run python populate_database.py`
3. Try rephrasing the question to use language closer to what would appear in the rulebook

### "collection star_wars_rpg does not exist"

The database has not been populated yet, or `COLLECTION_NAME` in `.env` does not match what was used during ingestion.

Go to the admin panel and click **Process PDFs**, then **Start Chat Service**. Or run `uv run python populate_database.py` from the command line.

### The model gives wrong or hallucinated answers

The LLM is constrained by the prompt to only use the retrieved context. If it is still hallucinating:

1. Increase `RETRIEVAL_K` to give it more context (try 20–30)
2. Check that the relevant rulebook is in `data/` and was processed
3. Try a larger Groq model (e.g. `llama-3.3-70b-versatile`)
4. Some questions require information spread across many pages — this is a fundamental RAG limitation (see [Known Limitations](#known-limitations))

### "No text extracted from [filename]. It may be scanned or image-based."

The PDF contains images of text rather than actual text. This requires OCR (Optical Character Recognition) before ingestion. A tool like `ocrmypdf` can process the file first:

```bash
sudo apt install ocrmypdf
ocrmypdf input.pdf output.pdf
# Then move output.pdf to data/ and re-run ingestion
```

### First startup is very slow

On first run, the `sentence-transformers` library downloads the embedding model (~90 MB). This is cached for subsequent runs. This download happens during PDF processing, not at server startup.

### `embeddings.position_ids | UNEXPECTED` warning on startup

You may see a table like this printed when the embedding model loads:

```text
Key                     | Status     |
------------------------+------------+
embeddings.position_ids | UNEXPECTED |
```

This is harmless. The `all-MiniLM-L6-v2` checkpoint was saved before newer versions of the `transformers` library added an `embeddings.position_ids` buffer. The library detects the mismatch and flags it, but initialises the buffer correctly and the model works as expected. The warning is suppressed in `retrieval.py` by setting the `transformers.modeling_utils` logger to `ERROR` level so it does not appear on every startup.

### `uv sync` fails

Ensure you are running Python 3.12 or later:

```bash
python3 --version
```

Also ensure `uv` itself is up to date:

```bash
uv self update
```

---

## Known Limitations

### Aggregation questions

Questions like *"which vehicle has the highest speed?"* or *"how many weapons have the Stun quality?"* will produce unreliable answers. RAG retrieves only the N most relevant chunks — it cannot scan the entire database to find a maximum or count all matching entries. For these questions, rephrase to be specific: *"What is the speed stat of a T-70 X-Wing?"* works well.

### Multi-hop reasoning

Questions that require combining information from multiple unrelated sections (e.g. *"Which Force power would be most effective against the creature described in the Bestiary?"*) may produce poor results because the relevant chunks may not all be retrieved together.

### Scanned PDFs

PDFs that are images of pages rather than text-based PDFs produce no usable text. They must be OCR'd first (see Troubleshooting above).

### Context window limits

Very long answers, or questions that need many chunks to answer accurately, may cause the LLM to lose track of earlier context. Reducing `RETRIEVAL_K` may paradoxically improve answer quality by keeping the prompt shorter and more focused.

---

## Extending the App

### Swapping the LLM provider (e.g. to use OpenAI or Anthropic instead of Groq)

All LLM logic is isolated in `llm.py`. To switch to a different provider:

1. Add the provider’s SDK: `uv add openai`
2. Replace `call_groq()` in `llm.py` with a function that calls the new client
3. Add the new API key to `.env` and read it in `config.py`

Nothing else in the codebase needs to change.

### Adding a different embedding model

1. Change `EMBEDDING_MODEL` in `.env` to any model name from [huggingface.co/models](https://huggingface.co/models?library=sentence-transformers)
2. Delete `chroma/` and re-run ingestion
3. Larger models (e.g. `all-mpnet-base-v2`) produce more accurate embeddings but are slower

### Supporting multiple game systems with separate collections

Change `COLLECTION_NAME` in `.env` to a different value (e.g. `genesys_rpg`) and re-run ingestion with the relevant PDFs in `data/`. You can switch between collections by changing the env var and restarting the server.

### Changing the chunk size

In `populate_database.py`, `chunk_text()` is called with `size=1000, overlap=200`. Larger chunks give the LLM more context per retrieved result but reduce how many distinct topics you can cover in a single query. Smaller chunks improve topic diversity but may cut off important context. After changing these values, delete `chroma/` and re-ingest.

### Adding source citations to answers

`retrieve_relevant_chunks()` in `retrieval.py` currently returns only the text. ChromaDB also stores metadata (the source PDF filename) alongside each chunk. To surface this, modify `retrieve_relevant_chunks()` to return both `documents` and `metadatas`, and update `build_prompt()` in `llm.py` to include the source filename in the context.

### Modifying the chat UI

The frontend is a single plain HTML file at `static/index.html`. It has no build step and no JavaScript framework — edit it directly and refresh the browser. The only API contract it relies on is:

- `POST /api/chat` with body `{"question": "..."}` returning `{"answer": "..."}`

### Modifying the admin panel

The admin panel is at `static/admin.html`, also plain HTML/CSS/JS. It relies on these API endpoints:

- `POST /api/upload` — upload a PDF file
- `GET /api/pdfs` — list uploaded files
- `POST /api/delete-pdf` — delete a file
- `POST /api/process` — start PDF ingestion (returns immediately, runs in background)
- `GET /api/process/progress` — poll processing progress
- `POST /api/start-chat` — load embedder and collection into memory
- `GET /api/status` — check whether the chat service is ready

### Adding a new API endpoint

Add a new route to `server.py` following the same pattern as the existing `/api/chat` route. If the endpoint needs database access, it is available via `app.state.embedder`, `app.state.collection`, and `app.state.settings`. Check `app.state.progress` for processing state.

---

## Testing

The test suite covers all business logic using mocks — no Groq API key, no ChromaDB database, and no embedding model downloads are required to run tests. Tests are fast and fully self-contained.

### Test dependencies

Test dependencies are declared as a separate `dev` group in `pyproject.toml` and are **not** installed in production:

| Package | Purpose |
| --- | --- |
| `pytest` | Test runner |
| `pytest-mock` | Provides the `mocker` fixture for concise `unittest.mock` usage |
| `httpx` | Required by FastAPI's `TestClient` to make in-process HTTP requests |

Install them with:

```bash
uv sync --all-groups
```

If you only want production dependencies (e.g. on a server), omit the flag:

```bash
uv sync
```

### Running the tests

Run the full suite:

```bash
uv run pytest tests/ -v
```

Run a single file:

```bash
uv run pytest tests/test_llm.py -v
```

Run a single test by name:

```bash
uv run pytest tests/test_rag.py::test_answer_question_returns_fallback_when_no_chunks -v
```

### Test structure

```text
tests/
├── __init__.py
├── test_config.py     # Settings defaults, env var overrides, immutability
├── test_llm.py        # Prompt building, Groq API call, error handling
├── test_retrieval.py  # ChromaDB query wrapping, embedding call, empty results
├── test_rag.py        # RAG orchestration, no-chunks fallback, k setting
└── test_server.py     # HTTP routes, request validation, empty input handling
```

### What each test file covers

**`test_config.py`** — verifies that `get_settings()` returns correct defaults when no environment variables are set, that it correctly reads overrides from env vars, that the `Settings` dataclass is immutable (frozen), and that `RETRIEVAL_K` is coerced from a string to an integer.

**`test_llm.py`** — verifies that `build_prompt()` includes the question, all context chunks, and the system instruction. For `call_groq()`, it verifies the correct model is used, the response text is returned and whitespace-stripped, and that connection errors, API errors, and missing response content all return a readable error string rather than raising an exception.

**`test_retrieval.py`** — verifies that `retrieve_relevant_chunks()` calls the embedder with the query text, passes `k` to the ChromaDB query, returns the documents list, and handles empty results and missing dictionary keys without crashing.

**`test_rag.py`** — verifies the full orchestration: that `answer_question()` calls retrieval and the LLM, passes the correct `k` value from settings, returns the LLM's response, and returns the fallback message (without calling the LLM at all) when no chunks are found.

**`test_server.py`** — verifies the HTTP layer using FastAPI's `TestClient`. Covers: the `GET /` route returns HTML containing the chat UI, `GET /admin` returns the admin panel, `GET /api/status` reports readiness correctly, `POST /api/chat` returns the answer from `answer_question()` or a "not running" message when the service has not been started, question whitespace is stripped, empty questions return a prompt message, and malformed requests return HTTP 422.

### Approach: mocking external dependencies

All tests mock the boundaries of the unit under test:

- `call_groq()` tests mock `llm.Groq` — no API call leaves the process
- `retrieve_relevant_chunks()` tests pass a `MagicMock` collection — no ChromaDB on disk is needed
- `answer_question()` tests mock `retrieve_relevant_chunks` and `call_groq` — they test orchestration logic only
- Server tests mock `answer_question` — they test that the HTTP layer wires things correctly

This means tests run in milliseconds and can be run anywhere, including in CI, without any infrastructure.

### Adding new tests

When you add a new feature:

1. Write tests for the layer you changed — if you add a parameter to `build_prompt()`, add a test in `test_llm.py`
2. Mock external dependencies at the boundary of the module under test
3. Use `pytest.MonkeyPatch` (via the `monkeypatch` fixture) to override environment variables in `test_config.py`
4. Use `unittest.mock.patch` as a context manager to replace functions in other modules

---

## Dependency Reference

### Production dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `fastapi` | ≥0.135 | ASGI web framework for the HTTP API |
| `uvicorn[standard]` | ≥0.42 | ASGI server that runs FastAPI |
| `chromadb` | ≥1.5 | Local vector database for storing and querying embeddings |
| `sentence-transformers` | ≥5.3 | Creates text embeddings from the `all-MiniLM-L6-v2` model |
| `groq` | ≥1.1 | Python SDK for the Groq cloud LLM API |
| `pypdf` | ≥6.9 | Extracts text from PDF files |
| `python-dotenv` | ≥1.2 | Loads `.env` files into environment variables |
| `python-multipart` | ≥0.0.22 | Handles file uploads in FastAPI |
| `rich` | ≥14.3 | Coloured terminal output for the CLI and ingestion script |

### Development dependencies

Declared under `[dependency-groups] dev` in `pyproject.toml`. Only installed when you run `uv sync --all-groups`.

| Package | Version | Purpose |
| --- | --- | --- |
| `pytest` | ≥9.0 | Test runner |
| `pytest-mock` | ≥3.15 | Cleaner mock API via the `mocker` fixture |
| `httpx` | ≥0.28 | In-process HTTP client used by FastAPI's `TestClient` |

All dependency versions are locked in `uv.lock`. Running `uv sync` will always produce an identical environment on any machine.
