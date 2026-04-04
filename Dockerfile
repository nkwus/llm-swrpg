FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only, no dev deps, into the system Python
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY config.py llm.py rag.py retrieval.py server.py populate_database.py query_chat.py ./
COPY static/ ./static/

# Copy the pre-built ChromaDB vector store
COPY chroma/ ./chroma/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
