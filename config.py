import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection_name: str
    embedding_model: str
    groq_api_key: str
    groq_model: str
    retrieval_k: int


def get_settings() -> Settings:
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", "chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "star_wars_rpg"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "15")),
    )
