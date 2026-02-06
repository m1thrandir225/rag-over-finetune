import json
import os
import sys

from internal.config import ConfigLoader
from internal.importer import DocumentImporter
from internal.rag import RAG

CONFIG_PATH = "./config.json"
SAMPLE_DOCUMENTS_PATH = "./sample_data/sample_documents.json"
DATA_FOLDER_PATH = "./data"

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


def parse_documents(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Parses documents from a predefined format(To be determined) #TODO: add default format
    """
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        title = doc["title"]
        content = doc.get("content", "")
        if content:
            text = f"{title}\n\n{content}"
        else:
            text = title
        texts.append(text)

        metadata = {}
        if "id" in doc:
            metadata["id"] = doc["id"]
        if "site_url" in doc:
            metadata["site_url"] = doc["site_url"]
        if "page_url" in doc:
            metadata["page_url"] = doc["page_url"]
        if "published_at" in doc:
            metadata["published_at"] = doc["published_at"]
        if "categories" in doc:
            metadata["categories"] = ", ".join(doc["categories"])

        metadata["title"] = title

        metadatas.append(metadata)

    return texts, metadatas


def load_all_documents(importer: DocumentImporter) -> tuple[list[str], list[dict]]:
    all_records: list[dict] = []

    # always load sample documents
    if os.path.isfile(SAMPLE_DOCUMENTS_PATH):
        sample_docs = load_sample_documents(SAMPLE_DOCUMENTS_PATH)
        print(f"  Loaded {len(sample_docs)} records from sample_documents.json")
        all_records.extend(sample_docs)

    # load data folder if it exists
    if os.path.isdir(DATA_FOLDER_PATH):
        print(f"\nDetected data folder: {DATA_FOLDER_PATH}")
        folder_records = importer.load_data_folder(DATA_FOLDER_PATH)
        all_records.extend(folder_records)
    else:
        print(f"\nNo data folder found at {DATA_FOLDER_PATH} — skipping.")

    return parse_documents(all_records)


def run_demo(should_clear: bool = True, should_load_documents: bool = True) -> RAG:
    print("=" * 60)
    print("Simple Mode")
    print("=" * 60)

    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.load_config()

    print(f"\nProvider: {config.llm_provider.value}")
    print(f"Model: {config.llm_model}")
    print(f"Embedding device: {config.embedding_device}")
    print(f"Embedding batch size: {config.embedding_batch_size}")

    rag = RAG(config)

    if should_clear:
        print("\nClearing existing documents...")
        rag.clear()

    if should_load_documents:
        print("\nLoading documents...")
        importer = DocumentImporter()
        texts, metadatas = load_all_documents(importer)

        print(f"\nAdding {len(texts)} documents...")
        rag.add_texts(texts, metadatas)
        print(f"Total chunks in store: {rag.document_count()}")

    print("\n" + "=" * 60)
    print("Queries used for simple run: ")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"Question: {query}")
        print(f"{'─' * 60}")

        result = rag.query(query, include_scores=True)

        print(f"Answer: {result.answer}")

        if result.sources:
            print(f"Sources ({len(result.sources)}):")
            for i, source in enumerate(result.sources, 1):
                meta = source["metadata"]  # pyright: ignore
                title = meta.get("title", "")
                page_url = meta.get("page_url", "")
                print(f"  [{i}] Score: {source['score']:.3f} | {title}")  # pyright: ignore
                if page_url:
                    print(f"       URL: {page_url}")

    print("\n" + "-" * 60)
    return rag


def interactive_mode(rag: RAG) -> None:
    print("\n" + "-" * 60)
    print("Commands: 'quit', 'sources', 'count', 'clear', 'reload'")
    print("-" * 60)

    show_sources = True

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if not query:
                continue

            match query.lower():
                case "quit":
                    print("Goodbye!")
                    break
                case "sources":
                    show_sources = not show_sources
                    print(f"Sources: {'ON' if show_sources else 'OFF'}")
                    continue
                case "count":
                    print(f"Total chunks in store: {rag.document_count()}")
                    continue
                case "clear":
                    rag.clear()
                    print("Knowledge base cleared.")
                    continue
                case "reload":
                    print("Reloading documents...")
                    importer = DocumentImporter()
                    texts, metadatas = load_all_documents(importer)
                    rag.add_texts(texts, metadatas)
                    print(f"Loaded {rag.document_count()} chunks.")
                    continue

            result = rag.query(query, include_scores=show_sources)
            print(f"\nAnswer: {result.answer}")

            if show_sources and result.sources:
                print("\nSources:")
                for i, src in enumerate(result.sources, 1):
                    meta = src["metadata"]  # pyright: ignore
                    title = meta.get("title", "")  # pyright: ignore
                    page_url = meta.get("page_url", "")  # pyright: ignore
                    print(f"  [{i}] ({src['score']:.2f}) {title}")  # pyright: ignore
                    if page_url:
                        print(f"       URL: {page_url}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        rag = run_demo(
            should_clear=False, should_load_documents=False
        )  # TODO: make this configurable
        interactive_mode(rag=rag)
    else:
        run_demo(should_clear=False)  # TODO: make this configurable


if __name__ == "__main__":
    main()
