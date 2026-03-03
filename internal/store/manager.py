from typing import Any, List, Optional

from langchain_core.documents import Document

from ..config import Config
from ..processing import EmbeddingService
from .chroma import ChromaVectorStore
from .qdrant import QdrantVectorStore
from .vector import _DEFAULT_MAX_BATCH, VectorStoreBase


class VectorStoreManager:
    """
    High-level manager that selects the appropriate backend based on config
    and exposes a stable interface to the rest of the application.
    """

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config: Config = config
        self._embedding_service: EmbeddingService = embedding_service
        self._backend: Optional[VectorStoreBase] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def backend(self) -> VectorStoreBase:
        if self._backend is None:
            provider = (self._config.vector_store_provider or "qdrant").lower()
            if provider == "qdrant":
                self._backend = QdrantVectorStore(
                    config=self._config, embedding_service=self._embedding_service
                )
            elif provider == "chroma":
                self._backend = ChromaVectorStore(
                    config=self._config, embedding_service=self._embedding_service
                )
            else:
                raise ValueError(
                    f"Unknown vector_store_provider: {self._config.vector_store_provider}"
                )
        return self._backend

    def add_documents_batched(
        self, documents: list[Document], batch_size: int = _DEFAULT_MAX_BATCH
    ) -> list[str]:
        return self.backend.add_documents_batched(
            documents=documents, batch_size=batch_size
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.backend.add_documents(documents)

    def add_texts_batched(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = _DEFAULT_MAX_BATCH,
    ) -> list[str]:
        return self.backend.add_texts_batched(
            texts=texts, metadata=metadata, batch_size=batch_size
        )

    def add_texts(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
    ) -> list[str]:
        return self.backend.add_texts(texts=texts, metadata=metadata)

    def similarity_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        return self.backend.similarity_search(query=query, k=k)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        return self.backend.similarity_search_with_score(query=query, k=k)

    def as_retriever(self, **kwargs: Any):
        return self.backend.as_retriever(**kwargs)

    def clear(self) -> None:
        self.backend.clear()

    def purge(self) -> None:
        self.backend.purge()

    def document_count(self) -> int:
        return self.backend.document_count()
