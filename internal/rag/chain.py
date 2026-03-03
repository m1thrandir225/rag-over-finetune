from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from ..config import Config
from ..llm import LLMService
from ..processing import EmbeddingService, QueryTransformer, QueryTransformMode
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
        embedding_service: EmbeddingService,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._embedding_service = embedding_service
        self._query_transformer = QueryTransformer(
            llm=llm_service.llm,
            mode=QueryTransformMode(self._config.query_transform_mode),
        )

    @property
    def config(self) -> Config:
        return self._config

    @property
    def vector_store(self) -> VectorStoreManager:
        return self._vector_store

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    def _retrieve_with_transform(self, question: str) -> list[Document]:
        """
        Transform query, retrieve for each variant, deduplicate, and return docs.
        """
        result = self._query_transformer.transform(question)
        k = self._config.top_k
        all_docs: list[Document] = []
        seen: set[str] = set()

        for i, q in enumerate(result.queries):
            if result.use_embedding_of_first and i == 0:
                embedding = self._embedding_service.embed_query(q)
                docs = self._vector_store.similarity_search_by_vector(
                    embedding=embedding, k=k
                )
            else:
                docs = self._vector_store.similarity_search(query=q, k=k)

            for doc in docs:
                key = doc.page_content[:200] or str(id(doc))
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        return all_docs[:k]

    def build(self) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_prompt),
                ("human", self.config.prompt_template),
            ]
        )

        retriever = RunnableLambda(self._retrieve_with_transform)
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
