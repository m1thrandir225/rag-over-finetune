from typing import Optional

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ..config import Config


class VectorStoreManager:
    def __init__(self, config: Config, embeddings: HuggingFaceEmbeddings) -> None:
        self.config: Config = config
        self.embeddings: HuggingFaceEmbeddings = embeddings
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.config.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.chroma_persist_dir,
            )
        return self._vector_store

    def add_documents(self, documents: list[Document]) -> None:
        chunks = []  # TOOD: add chunker

        self.vector_store.add_documents(chunks)

    def add_texts(
        self, texts: list[str], metadatas: Optional[list[dict]] = None
    ) -> None:
        documents = []  # TODO: create documents from texts

        self.add_documents(documents)

    def similarity_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        k = k or self.config.top_k

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        k = k or self.config.top_k

        return self.vector_store.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs):
        search_kwargs = {"k": self.config.top_k}
        search_kwargs.update(kwargs.get("search_kwargs", {}))

        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs=search_kwargs
        )

    def clear(self) -> None:
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None

        _ = self.vector_store
