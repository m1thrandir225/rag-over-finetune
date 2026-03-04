from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from ..config import Config
from ..llm import LLMService
from ..mcp import MCPClient
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
            embedding_service=self._embedding_service,
        )
        self._chain: Runnable | None = None

        # MCP Client (optional, only if enabled)
        self._mcp_client: Optional[MCPClient] = None
        if self._config.mcp_enabled and self._config.mcp_servers:
            self._mcp_client = MCPClient(servers=self._config.mcp_servers)

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
    def mcp_client(self) -> Optional[MCPClient]:
        """Get the MCP client if MCP is enabled."""
        return self._mcp_client

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
            # use cached docs from the chain
            source_docs = self._chain_builder.last_retrieved_docs
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "rank": rank,
                }
                for rank, doc in enumerate(source_docs, start=1)
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

    async def query_with_tools(
            self, question: str, include_scores: bool = False
    ) -> dict:
        if not self._config.mcp_enabled:
            raise ValueError(
                "MCP tools are not enabled. Set 'mcp_enabled': true in config.json"
            )

        if not self._mcp_client:
            raise ValueError(
                "MCP client not initialized. Add MCP servers to 'mcp_servers' in config.json"
            )

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        # 1. Get RAG context
        context_docs = self.search(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

        # 2. Get MCP tools and build a lookup map
        tools = await self._mcp_client.get_tools()
        tools_by_name = {tool.name: tool for tool in tools}

        # 3. Bind tools to LLM
        llm_with_tools = self.llm_service.llm.bind_tools(tools)

        # 4. Build the initial messages
        messages = [
            SystemMessage(content=self._config.system_prompt),
            HumanMessage(
                content=self._config.prompt_template.format(
                    context=context, question=question
                )
            ),
        ]

        # 5. First LLM call
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 6. Tool execution loop — keep calling tools until the LLM gives a final answer
        all_tool_calls = []
        max_iterations = 5  # safety limit

        for _ in range(max_iterations):
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", tool_name)

                all_tool_calls.append({"name": tool_name, "args": tool_args})

                # Execute the tool
                if tool_name in tools_by_name:
                    try:
                        tool_result = await tools_by_name[tool_name].ainvoke(tool_args)
                        tool_result_str = str(tool_result)
                    except Exception as e:
                        tool_result_str = f"Error executing tool: {e}"
                else:
                    tool_result_str = f"Unknown tool: {tool_name}"

                # Add tool result to conversation
                messages.append(
                    ToolMessage(content=tool_result_str, tool_call_id=tool_call_id)
                )

            # Call LLM again with tool results
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

        # 7. Build response
        result = {
            "answer": response.content if hasattr(response, "content") else str(response),
            "tool_calls": all_tool_calls,
            "query": question,
        }

        if include_scores:
            source_docs = self._vector_store.similarity_search_with_score(question)
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in source_docs
            ]

        return result