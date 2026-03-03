import json
from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from tqdm import tqdm

from ..config import Config
from ..processing import EmbeddingService
from .vector import _DEFAULT_MAX_BATCH, VectorStoreBase

# Qdrant's HTTP API has a default 32MB payload limit. Because each point includes:
# - the embedding vector
# - the chunk text i.e page_content
# - metadata
# batching must be payload-size aware, not just count-based.
_QDRANT_MAX_HTTP_PAYLOAD_BYTES = 32 * 1024 * 1024

# Safety buffer for request overhead / float string size variance.
_QDRANT_TARGET_HTTP_PAYLOAD_BYTES = int(_QDRANT_MAX_HTTP_PAYLOAD_BYTES * 0.80)


def _estimate_point_payload_size_bytes(
    *,
    text: str,
    metadata: dict,
    vector_size: int,
) -> int:
    """
    Heuristic estimate of JSON bytes contributed by a single point.

    We intentionally over-estimate the vector size: JSON float encoding varies,
    but ~12 bytes per float is a reasonable upper-ish bound for typical embeddings.
    """

    # `ensure_ascii=False` matches typical payload encoding and avoids inflating non-ascii.
    meta_bytes = len(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    text_bytes = len(text.encode("utf-8"))

    # Embedding vector as JSON array of floats.
    vector_bytes = vector_size * 12

    # Structural overhead for fields, quotes, id, etc.
    overhead = 256

    return meta_bytes + text_bytes + vector_bytes + overhead


def _iter_size_bounded_slices(
    *,
    texts: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    max_batch_items: int,
    target_payload_bytes: int,
) -> Iterable[Tuple[int, int]]:
    if not texts:
        return
    vector_size = len(embeddings[0]) if embeddings else 0
    n = len(texts)

    start = 0
    while start < n:
        # Always make progress (even for one huge item).
        end = min(start + max_batch_items, n)

        # Grow a batch until adding the next item would exceed the target.
        size_acc = 0
        chosen_end = start
        while chosen_end < end:
            est = _estimate_point_payload_size_bytes(
                text=texts[chosen_end],
                metadata=metadatas[chosen_end],
                vector_size=vector_size,
            )
            if chosen_end == start and est > target_payload_bytes:
                # Single point likely too large for Qdrant's HTTP limit.
                # We'll still yield it alone so the caller can fail with a clearer error.
                chosen_end = start + 1
                break
            if size_acc + est > target_payload_bytes:
                break
            size_acc += est
            chosen_end += 1

        if chosen_end == start:
            chosen_end = min(start + 1, n)

        yield start, chosen_end
        start = chosen_end


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant-backed vector store using LangChain's QdrantVectorStore for retrieval
    and qdrant-client for fast batched upserts with pre-computed embeddings.
    """

    def __init__(self, config: Config, embedding_service: EmbeddingService) -> None:
        self._config = config
        self._embedding_service = embedding_service
        self._client: Any | None = None
        self._models: Any | None = None
        self._vector_store: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import (
                QdrantClient,  # type: ignore[import]
                models,
            )

            url = self._config.qdrant_url
            if url == ":memory:":
                self._client = QdrantClient(location=":memory:")
            else:
                self._client = QdrantClient(
                    url=url,
                    api_key=self._config.qdrant_api_key,
                    prefer_grpc=self._config.qdrant_prefer_grpc,
                )
            self._models = models
        return self._client

    @property
    def models(self) -> Any:
        if self._models is None:
            # Ensure client initialization, which also sets models
            _ = self.client
        return self._models

    @property
    def collection_name(self) -> str:
        return self._config.qdrant_collection_name

    def _ensure_collection(self, vector_size: int) -> None:
        client = self.client
        models = self.models

        try:
            client.get_collection(collection_name=self.collection_name)
        except Exception:
            # TODO: Handle not found error
            client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

            # Create payload indexes for common metadata filters
            for field_name in ("source", "tags", "published_at"):
                try:
                    client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    # Ignore index creation errors like already exists
                    pass

    @property
    def vector_store(self) -> Any:
        if self._vector_store is None:
            from langchain_qdrant import (
                QdrantVectorStore as LCQdrantVectorStore,  # type: ignore[import]
            )

            self._vector_store = LCQdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self._embedding_service.embeddings,
            )
        return self._vector_store

    def add_documents_batched(
        self,
        documents: list[Document],
        batch_size: int = _DEFAULT_MAX_BATCH,
    ) -> list[str]:
        sanitized_documents = filter_complex_metadata(documents)
        texts = [doc.page_content for doc in sanitized_documents]
        metadatas = [doc.metadata for doc in sanitized_documents]
        return self.add_texts_batched(texts, metadatas, batch_size=batch_size)

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

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size=vector_size)

        client = self.client
        models = self.models
        all_ids: list[str] = []
        max_batch_items = max(1, int(min(batch_size, 10_000)))
        slices = list(
            _iter_size_bounded_slices(
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                max_batch_items=max_batch_items,
                target_payload_bytes=_QDRANT_TARGET_HTTP_PAYLOAD_BYTES,
            )
        )

        def _upsert_with_split_retry(start: int, end: int) -> None:
            batch_ids = [str(uuid4()) for _ in range(end - start)]
            batch_texts = texts[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadatas = metadatas[start:end]

            points = [
                models.PointStruct(
                    id=pid,
                    vector=emb,
                    payload={
                        "page_content": text,
                        **meta,
                    },
                )
                for pid, text, emb, meta in zip(
                    batch_ids, batch_texts, batch_embeddings, batch_metadatas
                )
            ]

            try:
                client.upsert(collection_name=self.collection_name, points=points)
                all_ids.extend(batch_ids)
                return
            except Exception as e:
                msg = str(e)
                payload_too_large = (
                    "JSON payload" in msg and "larger than allowed" in msg
                ) or ("Payload error" in msg and "limit" in msg)

                if not payload_too_large or (end - start) <= 1:
                    if payload_too_large and (end - start) <= 1:
                        est = _estimate_point_payload_size_bytes(
                            text=batch_texts[0],
                            metadata=batch_metadatas[0],
                            vector_size=len(batch_embeddings[0]),
                        )
                        raise ValueError(
                            "A single chunk is too large to upsert into Qdrant over HTTP. "
                            f"Estimated payload bytes for this chunk is ~{est:,}, "
                            f"but Qdrant's HTTP limit is {_QDRANT_MAX_HTTP_PAYLOAD_BYTES:,}. "
                            "Consider reducing your chunk size, omitting `page_content` from the payload, "
                            "or enabling gRPC (`qdrant_prefer_grpc=true`) if your deployment allows it."
                        ) from e
                    raise

                mid = start + ((end - start) // 2)
                _upsert_with_split_retry(start, mid)
                _upsert_with_split_retry(mid, end)

        for start, end in tqdm(slices, desc="Inserting into Qdrant", unit="batch"):
            _upsert_with_split_retry(start, end)

        return all_ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        sanitized_documents = filter_complex_metadata(documents)
        return self.vector_store.add_documents(sanitized_documents)

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
        return self.add_documents_batched(sanitized_documents)

    def similarity_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        k = k or self._config.top_k
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_by_vector(
        self, embedding: list[float], k: Optional[int] = None
    ) -> list[Document]:
        k = k or self._config.top_k
        return self.vector_store.similarity_search_by_vector(embedding, k=k)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        k = k or self._config.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs: Any):
        search_kwargs = {"k": self._config.top_k}
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs=search_kwargs
        )

    def clear(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        finally:
            self._vector_store = None

    def purge(self) -> None:
        """
        For Qdrant, purge is equivalent to clear since storage is managed
        by the Qdrant service itself.
        """
        self.clear()

    def document_count(self) -> int:
        try:
            result = self.client.count(collection_name=self.collection_name, exact=True)
            return int(result.count)
        except Exception:
            return 0
