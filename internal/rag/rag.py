from langchain_core.runnables import Runnable

from ..config import Config
from ..llm import LLMService
from ..processing import EmbeddingService
from ..rag import RAGChain
from ..store import VectorStoreManager


class RAG:
    """
    Main entry point for the system. Coordinates all the other components and services
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        self._config = config

        self._document_processor = None  # TODO: Implement document processor
        self._embedding_service = EmbeddingService(config=self._config)

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
    def document_processor(self) -> None:
        return self._document_processor

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

    def add_documents():
        pass

    def add_texts():
        pass

    def add_file():
        pass

    def add_directory():
        pass

    def clear(self) -> None:
        self._vector_store.clear()
        self._invalidate_chain()

    def document_count(self) -> int:
        return self._vector_store.document_count()

    def query():
        pass

    def query_simple():
        pass

    def search():
        pass

    def search_with_scores():
        pass
