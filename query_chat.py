from rich import print

from config import get_settings
from rag import answer_question
from retrieval import get_chroma_collection, get_embedder


def chat_loop() -> None:
    print("[bold green]Star Wars RPG Chatbot[/bold green]")
    print("Type your question, or 'quit' to exit.\n")

    settings = get_settings()
    embedder = get_embedder(settings)
    collection = get_chroma_collection(settings)

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("[bold blue]Goodbye! May the Force be with you.[/bold blue]")
            break
        if not question:
            continue

        print("[cyan]Searching your Star Wars PDFs...[/cyan]")
        answer = answer_question(question, embedder, collection, settings)
        print(f"[bold magenta]Bot:[/bold magenta] {answer}\n")


if __name__ == "__main__":
    chat_loop()