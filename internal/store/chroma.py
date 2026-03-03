import shutil
from typing import Any, List, Optional
from uuid import uuid4

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from tqdm import tqdm

from ..config import Config
from ..constants import DEFAULT_CHROMA_COLLECTION_NAME, DEFAULT_CHROMA_PERSIST_DIR
from ..processing import EmbeddingService
from .vector import _DEFAULT_MAX_BATCH, VectorStoreBase


class ChromaVectorStore(VectorStoreBase):
    """
    Deprecated Chroma-based vector store kept as a fallback behind the
    `vector_store_provider` config flag.
    """

    def __init__(self, config: Config, embedding_service: EmbeddingService) -> None:
        self._config = config
        self._embedding_service = embedding_service
        self._collection_name = DEFAULT_CHROMA_COLLECTION_NAME
        self._persist_dir = DEFAULT_CHROMA_PERSIST_DIR
        self._store: Any | None = None

    @property
    def store(self) -> Any:
        if self._store is None:
            from langchain_chroma import Chroma  # type: ignore[import]

            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_service.embeddings,
                persist_directory=self._persist_dir,
            )
        return self._store

    def add_documents_batched(
        self,
        documents: list[Document],
        batch_size: int = _DEFAULT_MAX_BATCH,
    ) -> list[str]:
        sanitized_documents = filter_complex_metadata(documents)
        sanitized_texts = [doc.page_content for doc in sanitized_documents]
        sanitized_metadatas = [doc.metadata for doc in sanitized_documents]
        return self.add_texts_batched(
            sanitized_texts, sanitized_metadatas, batch_size=batch_size
        )

    def add_texts_batched(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = _DEFAULT_MAX_BATCH,
    ) -> list[str]:
        if not texts:
            return []

        metadatas = metadata or [{}] * len(texts)
        filtered_pairs = [
            (text, meta)
            for text, meta in zip(texts, metadatas)
            if isinstance(text, str) and text.strip()
        ]
        if not filtered_pairs:
            return []
        if len(filtered_pairs) != len(texts):
            skipped = len(texts) - len(filtered_pairs)
            print(f"Skipping {skipped} empty chunk(s) before embedding.")
        texts = [text for text, _ in filtered_pairs]
        metadatas = [meta for _, meta in filtered_pairs]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self._embedding_service.embed_documents(texts)
        if not embeddings:
            raise ValueError("Embedding service returned no embeddings.")
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(texts)} texts."
            )

        collection = self.store._collection
        all_ids: list[str] = []
        total = len(texts)

        for start in tqdm(
            range(0, total, batch_size),
            desc="Inserting into Chroma",
            unit="batch",
        ):
            end = min(start + batch_size, total)
            batch_ids = [str(uuid4()) for _ in range(end - start)]

            collection.add(
                documents=texts[start:end],
                metadatas=metadatas[start:end],  # type: ignore[arg-type]
                embeddings=embeddings[start:end],  # type: ignore[arg-type]
                ids=batch_ids,
            )
            all_ids.extend(batch_ids)

        return all_ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        sanitized_documents = filter_complex_metadata(documents)
        return self.store.add_documents(sanitized_documents)

    def add_texts(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
    ) -> list[str]:
        metadatas = metadata or [{}] * len(texts)
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        sanitized_documents = filter_complex_metadata(docs)
        sanitized_metadatas = [doc.metadata for doc in sanitized_documents]
        return self.store.add_texts(texts, metadatas=sanitized_metadatas)

    def similarity_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        k = k or self._config.top_k
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        k = k or self._config.top_k
        return self.store.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs: Any):
        search_kwargs = {"k": self._config.top_k}
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        return self.store.as_retriever(
            search_type="similarity", search_kwargs=search_kwargs
        )

    def clear(self) -> None:
        try:
            self.store.delete_collection()
        finally:
            self._store = None

    def purge(self) -> None:
        self.clear()
        shutil.rmtree(self._persist_dir, ignore_errors=True)

    def document_count(self) -> int:
        return self.store._collection.count()
