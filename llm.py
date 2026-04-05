from groq import Groq

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


def call_groq(prompt: str, settings: Settings) -> str:
    client = Groq(api_key=settings.groq_api_key)
    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return f"Error: could not reach Groq API ({e})."
    return (completion.choices[0].message.content or "").strip()
