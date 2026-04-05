import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import Settings

# Suppress a noisy but harmless warning from the transformers library when loading
# all-MiniLM-L6-v2. The model checkpoint pre-dates the `embeddings.position_ids`
# buffer that newer versions of transformers expect; the library handles the
# mismatch gracefully but logs it as UNEXPECTED at WARNING level on every startup.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def get_chroma_collection(settings: Settings) -> Any:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_collection(settings.collection_name)


def get_or_create_chroma_collection(settings: Settings) -> Any:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(settings.collection_name)


def get_embedder(settings: Settings) -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def retrieve_relevant_chunks(
    query: str,
    embedder: SentenceTransformer,
    collection: Any,
    k: int,
) -> list[str]:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    documents: list[list[str]] = results.get("documents", [[]])
    return documents[0]
