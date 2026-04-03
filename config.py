import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection_name: str
    embedding_model: str
    ollama_url: str
    ollama_model: str
    retrieval_k: int


def get_settings() -> Settings:
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", "chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "star_wars_rpg"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "15")),
    )
