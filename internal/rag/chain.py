from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from ..config import Config
from ..llm import LLMService
from ..processing import EmbeddingService, QueryTransformer, Reranker, TransformResult
from ..store import VectorStoreManager

logger = logging.getLogger(__name__)

# RRF smoothing constant
_RRF_K = 60

_SUPPORTED_MERGE_STRATEGIES = {"rrf"}


def _doc_key(doc: Document) -> str:
    """
    Dedub key which is the hash of the first 300 chars of page_content
    """

    snippet = doc.page_content[:300]
    return hashlib.md5(snippet.encode("utf-8")).hexdigest()


def _rrf_merge(
    ranked_lists: list[list[Document]],
    k_total: int,
) -> list[tuple[Document, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked result lists.

    For each document seen across all lists, the RRF score is:
        sum( 1 / (rrf_k + rank_i) )  for every list that contains it.

    Returns the top *k_total* (document, rrf_score) pairs sorted by
    descending RRF score.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_by_key: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (_RRF_K + rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [(doc_by_key[k], scores[k]) for k in sorted_keys[:k_total]]


def _simple_dedup(
    docs: list[Document],
    k_total: int,
) -> list[tuple[Document, float]]:
    """
    Deduplicate by content hash, cap at *k_total*, and assign a positional
    RRF-style score (`1 / (rrf_k + rank)`) so callers always get a
    comparable numeric score regardless of the merge path taken
    """

    seen: set[str] = set()
    result: list[tuple[Document, float]] = []
    rank = 0
    for doc in docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            rank += 1
            result.append((doc, 1.0 / (_RRF_K + rank)))
        if len(result) >= k_total:
            break
    return result


class RAGChain:
    """
    Builds and configures the RAG chain with composable query transforms
    and RRF-based retrieval merge.
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

        # cache the last retrieved docs so we dont run multiple retrievals
        self._last_retrieved_docs: list[tuple[Document, float]] = []

        if config.merge_strategy not in _SUPPORTED_MERGE_STRATEGIES:
            raise ValueError(
                f"Unsupported merge_strategy {config.merge_strategy!r}. "
                f"Supported values: {sorted(_SUPPORTED_MERGE_STRATEGIES)}"
            )

        self._query_transformer = QueryTransformer(
            llm=llm_service.llm,
            enabled_transforms=list(self._config.enabled_transforms),
            max_generated_queries=self._config.max_generated_queries,
            gate_enabled=self._config.transform_gate_enabled,
            timeout_ms=self._config.query_transform_timeout_ms,
            hyde_include_original_query=self._config.hyde_include_original_query,
        )

        self._reranker: Reranker | None = None
        if config.reranker_enabled:
            self._reranker = Reranker(
                model_name=config.reranker_model,
                device=config.embedding_device,
            )

    @property
    def config(self) -> Config:
        return self._config

    @property
    def last_retrieved_docs(self) -> list[tuple[Document, float]]:
        """
        Documents returned by the most recent '_retrieve_with_transform'
        """

        return self._last_retrieved_docs

    @property
    def vector_store(self) -> VectorStoreManager:
        return self._vector_store

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    def _retrieve_with_transform(self, question: str) -> list[Document]:
        """
        1. Transform the query using the query transformer
        2. Retrieve for each expanded query with respect to the per-query limit
        3. Merge via RRF (or simple dedup) with global cap
        """

        t0 = time.perf_counter()

        result: TransformResult = self._query_transformer.transform(question)

        k_per = self._config.k_per_query
        k_total = self._config.k_total_before_rerank
        merge_strategy = self._config.merge_strategy

        ranked_lists: list[list[Document]] = []

        for idx, q in enumerate(result.expanded_queries):
            if idx in result.use_embedding_of:
                # HyDE path: embed the text and search by vector
                embedding = self._embedding_service.embed_query(q)
                docs = self._vector_store.similarity_search_by_vector(
                    embedding=embedding, k=k_per
                )
                logger.debug(
                    "Retrieval (embed) query[%d]: %d docs (%.40s…)",
                    idx,
                    len(docs),
                    q,
                )
            else:
                docs = self._vector_store.similarity_search(query=q, k=k_per)
                logger.debug(
                    "Retrieval (text) query[%d]: %d docs (%.40s…)",
                    idx,
                    len(docs),
                    q,
                )
            ranked_lists.append(docs)

        if len(ranked_lists) <= 1:
            flat = [
                doc for rl in ranked_lists for doc in rl
            ]  # flatten the ranked lists into a single one
            merged = _simple_dedup(flat, k_total)
        else:
            merged = _rrf_merge(ranked_lists, k_total)

        if self._reranker is not None:
            rerank_top_n = self._config.reranker_top_n
            merged = self._reranker.rerank(question, merged, rerank_top_n)

        elapsed = (time.perf_counter() - t0) * 1_000

        if len(ranked_lists) > 1:
            per_list_keys = [{_doc_key(d) for d in rl} for rl in ranked_lists]
            total_raw = sum(len(rl) for rl in ranked_lists)
            unique_keys = set().union(*per_list_keys) if per_list_keys else set()
            overlap_count = total_raw - len(unique_keys)
            logger.debug(
                "Retrieve: %d queries -> %d raw docs, %d unique, "
                "%d overlapping -> %d after merge (%s) in %.1f ms "
                "(transforms=%s, gate_skipped=%s)",
                len(result.expanded_queries),
                total_raw,
                len(unique_keys),
                overlap_count,
                len(merged),
                merge_strategy,
                elapsed,
                result.applied_transforms,
                result.gate_skipped,
            )
        else:
            logger.debug(
                "Retrieve: %d queries -> %d docs in %.1f ms "
                "(transforms=%s, gate_skipped=%s)",
                len(result.expanded_queries),
                len(merged),
                elapsed,
                result.applied_transforms,
                result.gate_skipped,
            )

        self._last_retrieved_docs = merged
        return [doc for doc, _score in merged]

    def build(self) -> Runnable:
        """
        Build the RAG chain
        """

        t0 = time.perf_counter()

        result: TransformResult = self._query_transformer.transform(question)

        k_per = self._config.k_per_query
        k_total = self._config.k_total_before_rerank
        merge_strategy = self._config.merge_strategy

        ranked_lists: list[list[Document]] = []

        for idx, q in enumerate(result.expanded_queries):
            if idx in result.use_embedding_of:
                # HyDE path: embed the text and search by vector
                embedding = self._embedding_service.embed_query(q)
                docs = self._vector_store.similarity_search_by_vector(
                    embedding=embedding, k=k_per
                )
                logger.debug(
                    "Retrieval (embed) query[%d]: %d docs (%.40s…)",
                    idx,
                    len(docs),
                    q,
                )
            else:
                docs = self._vector_store.similarity_search(query=q, k=k_per)
                logger.debug(
                    "Retrieval (text) query[%d]: %d docs (%.40s…)",
                    idx,
                    len(docs),
                    q,
                )
            ranked_lists.append(docs)

        if len(ranked_lists) <= 1 or merge_strategy != "rrf":
            flat = [
                doc for rl in ranked_lists for doc in rl
            ]  # flatten the ranked lists into a single one
            merged = _simple_dedup(flat, k_total)
        else:
            merged = _rrf_merge(ranked_lists, k_total)

        elapsed = (time.perf_counter() - t0) * 1_000

        # Overlap stats (TODO: remove)
        if len(ranked_lists) > 1:
            per_list_keys = [{_doc_key(d) for d in rl} for rl in ranked_lists]
            total_raw = sum(len(rl) for rl in ranked_lists)
            unique_keys = set().union(*per_list_keys) if per_list_keys else set()
            overlap_count = total_raw - len(unique_keys)
            logger.info(
                "Retrieve: %d queries -> %d raw docs, %d unique, "
                "%d overlapping -> %d after merge (%s) in %.1f ms "
                "(transforms=%s, gate_skipped=%s)",
                len(result.expanded_queries),
                total_raw,
                len(unique_keys),
                overlap_count,
                len(merged),
                merge_strategy,
                elapsed,
                result.applied_transforms,
                result.gate_skipped,
            )
        else:
            logger.info(
                "Retrieve: %d queries -> %d docs in %.1f ms "
                "(transforms=%s, gate_skipped=%s)",
                len(result.expanded_queries),
                len(merged),
                elapsed,
                result.applied_transforms,
                result.gate_skipped,
            )

        return merged

    def build(self) -> Runnable:
        """
        Build the RAG chain
        """

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
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
