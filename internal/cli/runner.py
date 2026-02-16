import json
import os
from dataclasses import replace
from typing import Callable

from internal.config import Config, ConfigLoader
from internal.constants import DEFAULT_TEST_QUERIES
from internal.importer import DocumentImporter
from internal.rag import RAG

from .context import CLIContext


def load_sample_documents(path: str) -> list[dict]:
    """
    Load sample documents from a JSON file
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_documents(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Parse documents from predefined format into texts and metadatas
    """
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        title = doc["title"]
        content = doc.get("content", "")
        text = f"{title}\n\n{content}" if content else title
        texts.append(text)

        metadata: dict = {}
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


def load_all_documents(
    importer: DocumentImporter,
    data_path: str,
    sample_path: str,
    *,
    print_fn: Callable[..., None] = print,
) -> tuple[list[str], list[dict]]:
    """
    Load documents from sample file and data folder
    """
    all_records: list[dict] = []

    if os.path.isfile(sample_path):
        sample_docs = load_sample_documents(sample_path)
        print_fn(f"  Loaded {len(sample_docs)} records from sample_documents.json")
        all_records.extend(sample_docs)

    if os.path.isdir(data_path):
        print_fn(f"\nDetected data folder: {data_path}")
        folder_records = importer.load_data_folder(data_path)
        all_records.extend(folder_records)
    else:
        print_fn(f"\nNo data folder found at {data_path} — skipping.")

    return parse_documents(all_records)


def bootstrap_rag(
    ctx: CLIContext,
    *,
    should_clear: bool = False,
    should_load_documents: bool = True,
    top_k: int | None = None,
    config_loader: ConfigLoader | None = None,
    importer: DocumentImporter | None = None,
    verbose: bool = False,
    print_fn: Callable[..., None] = print,
) -> RAG:
    """
    Shared bootstrap for both demo and interactive modes
    """
    loader = config_loader or ConfigLoader(ctx.config_path)
    config: Config = loader.load_config()

    if top_k is not None:
        config = replace(config, top_k=top_k)

    if verbose:
        print_fn(f"Config path: {ctx.config_path}")
        print_fn(f"Data path: {ctx.data_path}")
        print_fn(f"Chroma dir: {config.chroma_persist_dir}")
        print_fn(f"Top-k: {config.top_k}")

    print_fn(f"Provider: {config.llm_provider.value}")
    print_fn(f"Model: {config.llm_model}")
    print_fn(f"Embedding provider: {config.embedding_provider.value}")
    print_fn(f"Embedding model: {config.embedding_model}")
    if config.embedding_provider.value == "huggingface":
        print_fn(f"Embedding device: {config.embedding_device}")
    print_fn(f"Embedding batch size: {config.embedding_batch_size}")

    rag = RAG(config)
    imp = importer or DocumentImporter()

    if should_clear:
        print_fn("Clearing existing documents...")
        rag.clear()

    if should_load_documents:
        print_fn("Loading documents...")
        texts, metadatas = load_all_documents(
            imp, ctx.data_path, ctx.sample_documents_path, print_fn=print_fn
        )
        print_fn(f"Adding {len(texts)} documents...")
        rag.add_texts(texts, metadatas)
        print_fn(f"Total chunks in store: {rag.document_count()}")

    return rag


def run_demo(
    ctx: CLIContext,
    *,
    should_clear: bool = False,
    should_load_documents: bool = True,
    top_k: int | None = None,
    config_loader: ConfigLoader | None = None,
    importer: DocumentImporter | None = None,
    test_queries: list[str] | None = None,
    verbose: bool = False,
    print_fn: Callable[..., None] = print,
) -> RAG:
    print_fn("=" * 60)
    print_fn("Simple Mode")
    print_fn("=" * 60)

    rag = bootstrap_rag(
        ctx,
        should_clear=should_clear,
        should_load_documents=should_load_documents,
        top_k=top_k,
        config_loader=config_loader,
        importer=importer,
        verbose=verbose,
        print_fn=print_fn,
    )

    queries = test_queries or DEFAULT_TEST_QUERIES
    print_fn("\n" + "=" * 60)
    print_fn("Queries used for simple run: ")
    print_fn("=" * 60)

    for q in queries:
        print_fn(f"\n{'─' * 60}")
        print_fn(f"Question: {q}")
        print_fn(f"{'─' * 60}")
        result = rag.query(q, include_scores=True)
        print_fn(f"Answer: {result.answer}")
        if result.sources:
            print_fn(f"Sources ({len(result.sources)}):")
            for i, source in enumerate(result.sources, 1):
                meta = source["metadata"]  # pyright: ignore
                title = meta.get("title", "")
                page_url = meta.get("page_url", "")
                print_fn(f"  [{i}] Score: {source['score']:.3f} | {title}")  # pyright: ignore
                if page_url:
                    print_fn(f"       URL: {page_url}")

    print_fn("\n" + "-" * 60)
    return rag


def run_single_query(
    ctx: CLIContext,
    *,
    query: str,
    should_clear: bool = False,
    should_load_documents: bool = False,
    top_k: int | None = None,
    config_loader: ConfigLoader | None = None,
    importer: DocumentImporter | None = None,
    verbose: bool = False,
    print_fn: Callable[..., None] = print,
) -> None:
    """
    Run a single query and exit. Uses existing RAG data unless --load-docs is set.
    """
    rag = bootstrap_rag(
        ctx,
        should_clear=should_clear,
        should_load_documents=should_load_documents,
        top_k=top_k,
        config_loader=config_loader,
        importer=importer,
        verbose=verbose,
        print_fn=print_fn,
    )

    result = rag.query(query, include_scores=True)
    print_fn(result.answer)
    if result.sources:
        print_fn("\nSources:")
        for i, source in enumerate(result.sources, 1):
            meta = source["metadata"]  # pyright: ignore
            title = meta.get("title", "")
            page_url = meta.get("page_url", "")
            print_fn(f"  [{i}] Score: {source['score']:.3f} | {title}")  # pyright: ignore
            if page_url:
                print_fn(f"       URL: {page_url}")


def run_interactive(
    ctx: CLIContext,
    *,
    should_clear: bool = False,
    should_load_documents: bool = False,
    top_k: int | None = None,
    config_loader: ConfigLoader | None = None,
    importer: DocumentImporter | None = None,
    verbose: bool = False,
    print_fn: Callable[..., None] = print,
    input_fn: Callable[[str], str] = input,
) -> None:
    print_fn("=" * 60)
    print_fn("Interactive Mode")
    print_fn("=" * 60)

    rag = bootstrap_rag(
        ctx,
        should_clear=should_clear,
        should_load_documents=should_load_documents,
        top_k=top_k,
        config_loader=config_loader,
        importer=importer,
        verbose=verbose,
        print_fn=print_fn,
    )

    print_fn("\n" + "-" * 60)
    print_fn("Commands: 'quit', 'sources', 'count', 'clear', 'reload'")
    print_fn("-" * 60)

    show_sources = True
    imp = importer or DocumentImporter()

    while True:
        try:
            query = input_fn("\nQuestion: ").strip()

            if not query:
                continue

            match query.lower():
                case "quit":
                    print_fn("Goodbye!")
                    break
                case "sources":
                    show_sources = not show_sources
                    print_fn(f"Sources: {'ON' if show_sources else 'OFF'}")
                    continue
                case "count":
                    print_fn(f"Total chunks in store: {rag.document_count()}")
                    continue
                case "clear":
                    rag.clear()
                    print_fn("Knowledge base cleared.")
                    continue
                case "reload":
                    print_fn("Reloading documents...")
                    texts, metadatas = load_all_documents(
                        imp,
                        ctx.data_path,
                        ctx.sample_documents_path,
                        print_fn=print_fn,
                    )
                    rag.add_texts(texts, metadatas)
                    print_fn(f"Loaded {rag.document_count()} chunks.")
                    continue

            result = rag.query(query, include_scores=show_sources)
            print_fn(f"\nAnswer: {result.answer}")

            if show_sources and result.sources:
                print_fn("\nSources:")
                for i, src in enumerate(result.sources, 1):
                    meta = src["metadata"]  # pyright: ignore
                    title = meta.get("title", "")  # pyright: ignore
                    page_url = meta.get("page_url", "")  # pyright: ignore
                    print_fn(f"  [{i}] ({src['score']:.2f}) {title}")  # pyright: ignore
                    if page_url:
                        print_fn(f"       URL: {page_url}")

        except (KeyboardInterrupt, EOFError):
            print_fn("\n\nGoodbye!")
            break
