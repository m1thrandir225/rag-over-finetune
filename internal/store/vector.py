from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.documents import Document

_DEFAULT_MAX_BATCH = 5_000


class VectorStoreBase(ABC):
    """
    Abstract interface for vector store backends.
    """

    @abstractmethod
    def add_documents_batched(
        self, documents: list[Document], batch_size: int = _DEFAULT_MAX_BATCH
    ) -> list[str]: ...

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]: ...

    @abstractmethod
    def add_texts_batched(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = _DEFAULT_MAX_BATCH,
    ) -> list[str]: ...

    @abstractmethod
    def add_texts(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
    ) -> list[str]: ...

    @abstractmethod
    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> list[Document]: ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]: ...

    @abstractmethod
    def as_retriever(self, **kwargs: Any): ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def purge(self) -> None: ...

    @abstractmethod
    def document_count(self) -> int: ...
