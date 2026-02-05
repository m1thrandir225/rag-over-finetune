import json
import sys

from internal.config import ConfigLoader
from internal.rag import RAG

CONFIG_PATH = "./config.json"
SAMPLE_DOCUMENTS_PATH = "./sample_data/sample_documents.json"

TEST_QUERIES = [
    "Кој е главниот град на Македонија?",
    "Колку е длабоко Охридското Езеро?",
    "Што е тавче гравче?",
    "Кој ја создал кирилицата?",
    "Кога е роден Александар Македонски?",
]


def load_sample_documents(path: str) -> list[dict]:
    """
    Loads sample documents from json file
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_demo() -> RAG:
    print("=" * 60)
    print("Simple Mode")
    print("=" * 60)

    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.load_config()

    print(f"\nProvider: {config.llm_provider.value}")
    print(f"Model: {config.llm_model}")

    rag = RAG(config)

    print("\nClearing existing documents...")
    rag.clear()

    print("\nLoading sample documents...")
    sample_docs = load_sample_documents(SAMPLE_DOCUMENTS_PATH)

    texts = [doc["text"] for doc in sample_docs]
    metadatas = [doc["metadata"] for doc in sample_docs]

    print(f"Adding {len(texts)} documents...")
    rag.add_texts(texts, metadatas)
    print(f"Total chunks in store: {rag.document_count()}")

    print("\n" + "=" * 60)
    print("Queries used for simple run: ")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"ПРАШАЊЕ: {query}")
        print(f"{'─' * 60}")

        result = rag.query(query, include_scores=True)

        print(f"\nОДГОВОР: {result.answer}")

        if result.sources:
            print(f"\nИЗВОРИ ({len(result.sources)}):")
            for i, source in enumerate(result.sources, 1):
                preview = source["content"][:60].replace("\n", " ")  # pyright: ignore
                print(f"  [{i}] Score: {source['score']:.3f} | {preview}...")  # pyright: ignore

    print("\n" + "-" * 60)
    return rag


def interactive_mode(rag: RAG) -> None:
    print("\n" + "-" * 60)
    print("Commands: 'quit', 'sources', 'count', 'clear', 'reload'")
    print("-" * 60)

    show_sources = True

    while True:
        try:
            query = input("\nПрашање: ").strip()

            if not query:
                continue

            match query.lower():
                case "quit":
                    print("Довидување!")
                    break
                case "sources":
                    show_sources = not show_sources
                    print(f"Sources: {'ON' if show_sources else 'OFF'}")
                    continue
                case "count":
                    print(f"Documents: {rag.document_count()}")
                    continue
                case "clear":
                    rag.clear()
                    print("Knowledge base cleared.")
                    continue
                case "reload":
                    print("Reloading sample documents...")
                    sample_docs = load_sample_documents(SAMPLE_DOCUMENTS_PATH)
                    texts = [doc["text"] for doc in sample_docs]
                    metadatas = [doc["metadata"] for doc in sample_docs]
                    rag.add_texts(texts, metadatas)
                    print(f"Loaded {rag.document_count()} chunks.")
                    continue

            result = rag.query(query, include_scores=show_sources)
            print(f"\nОдговор: {result.answer}")

            if show_sources and result.sources:
                print("\nИзвори:")
                for i, src in enumerate(result.sources, 1):
                    preview = src["content"][:50].replace("\n", " ")  # pyright: ignore
                    print(f"  [{i}] ({src['score']:.2f}) {preview}...")  # pyright: ignore

        except KeyboardInterrupt:
            print("\n\nДовидување!")
            break


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        config_loader = ConfigLoader(CONFIG_PATH)
        config = config_loader.load_config()
        rag = RAG(config)
        interactive_mode(rag)
    else:
        run_demo()


if __name__ == "__main__":
    main()
