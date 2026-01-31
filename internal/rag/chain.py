from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from ..config import Config
from ..llm import LLMService
from ..store import VectorStoreManager


class RAGChain:
    """
    Builds and configures the RAG chain
    """

    def __init__(
        self,
        config: Config,
        vector_store: VectorStoreManager,
        llm_service: LLMService,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._llm_service = llm_service

    @property
    def config(self) -> Config:
        return self._config

    @property
    def vector_store(self) -> VectorStoreManager:
        return self._vector_store

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    def build(self) -> Runnable:
        retriever = self.vector_store.as_retriever()

        prompt = PromptTemplate.from_template(self.config.prompt_template)

        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm_service.llm
            | StrOutputParser()
        )
        return chain

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """
        Formats documents into a context string
        """

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
