from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from ..config import Config
from ..llm import LLMService
from ..processing import Chunker, EmbeddingService
from ..store import VectorStoreManager
from .chain import RAGChain
from .query_result import QueryResult


class RAG:
    """
    Main entry point for the system. Coordinates all the other components and services
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self._config = config

        self._embedding_service = EmbeddingService(config=self._config)
        self._chunker = Chunker(
            config=self._config,
            embeddings=self._embedding_service.embeddings,
        )

        self._vector_store = VectorStoreManager(
            config=self._config, embedding_service=self._embedding_service
        )
        self._llm_service = LLMService(config=self._config)
        self._chain_builder = RAGChain(
            config=self._config,
            vector_store=self._vector_store,
            llm_service=self._llm_service,
        )
        self._chain: Runnable | None = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def chunker(self) -> Chunker:
        return self._chunker

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStoreManager:
        return self._vector_store

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self._chain_builder.build()
        return self._chain

    def _invalidate_chain(self) -> None:
        """
        Clear the chain cache when documents change
        """
        self._chain = None

    def add_documents(
        self, documents: list[Document], use_batched_path: bool = True
    ) -> list[str]:
        """
        Add documents to the vector store after chunking them.
        """

        chunks = self._chunker.split_documents(documents)
        if use_batched_path:
            ids = self._vector_store.add_documents_batched(chunks)
        else:
            ids = self._vector_store.add_documents(chunks)

        self._invalidate_chain()
        return ids

    def add_texts(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
        use_batched_path: bool = True,
    ) -> list[str]:
        """
        Add raw texts to the vector store after creating documents and chunking them.
        """

        documents = self._chunker.create_documents(texts, metadata)
        return self.add_documents(documents, use_batched_path=use_batched_path)

    def clear(self) -> None:
        self._vector_store.clear()
        self._invalidate_chain()

    def purge(self) -> None:
        """
        Clear the Chroma collection and delete the persist directory from disk.
        """
        self._vector_store.purge()
        self._invalidate_chain()

    def document_count(self) -> int:
        return self._vector_store.document_count()

    def query(self, question: str, include_scores: bool = False) -> QueryResult:
        answer = self.chain.invoke(question)

        sources: list[str] | list[dict] = []

        if include_scores:
            source_docs = self._vector_store.similarity_search_with_score(question)
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in source_docs
            ]
        return QueryResult(
            answer=answer,
            sources=sources,
            query=question,
        )

    def query_simple(self, question: str) -> str:
        return self.chain.invoke(question)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return self._vector_store.similarity_search(query, k=k)

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        return self._vector_store.similarity_search_with_score(query, k=k)
