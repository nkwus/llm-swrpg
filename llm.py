import requests

from config import Settings


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


def call_ollama(prompt: str, settings: Settings) -> str:
    url = f"{settings.ollama_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: could not reach the local LLM ({e}). Is Ollama running?"

    data: dict[str, str] = response.json()
    return data.get("response", "").strip()
