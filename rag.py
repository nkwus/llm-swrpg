from typing import Any

from sentence_transformers import SentenceTransformer

from config import Settings
from llm import build_prompt, call_ollama
from retrieval import retrieve_relevant_chunks


def answer_question(
    question: str,
    embedder: SentenceTransformer,
    collection: Any,
    settings: Settings,
) -> str:
    chunks = retrieve_relevant_chunks(
        question, embedder, collection, k=settings.retrieval_k
    )
    if not chunks:
        return "I couldn't find any relevant information in the database."

    prompt = build_prompt(chunks, question)
    return call_ollama(prompt, settings)
