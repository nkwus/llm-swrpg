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