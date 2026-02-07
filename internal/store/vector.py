from typing import List, Optional
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from ..config import Config
from ..processing import EmbeddingService

# Chroma's hard limit per upsert call
_CHROMA_MAX_BATCH = 5_000


class VectorStoreManager:
    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config: Config = config
        self._embedding_service: EmbeddingService = embedding_service
        self._vector_store: Optional[Chroma] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store

    def _create_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.config.chroma_collection_name,
            embedding_function=self.embedding_service.embeddings,
            persist_directory=self.config.chroma_persist_dir,
        )

    def add_documents_batched(
        self,
        documents: list[Document],
        batch_size: int = _CHROMA_MAX_BATCH,
    ) -> list[str]:
        """
        Embeds all documents in one pass, then insert the pre-computed vectors in batches,
        avoids Chroma's internal re-embedding overhead.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return self.add_texts_batched(texts, metadatas, batch_size=batch_size)

    def add_texts_batched(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
        batch_size: int = _CHROMA_MAX_BATCH,
    ) -> list[str]:
        if not texts:
            return []

        metadatas = metadata or [{}] * len(texts)

        # precompute all embeddings
        # Note: sentence-transformers handles its own internal batching and shows a progress bar
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self._embedding_service.embed_documents(texts)

        collection = self.vector_store._collection
        all_ids: list[str] = []
        total = len(texts)

        for start in tqdm(
            range(0, total, batch_size),
            desc="Inserting into Chroma",
            unit="batch",
        ):
            end = min(start + batch_size, total)
            batch_ids = [str(uuid4()) for _ in range(end - start)]

            collection.add(
                documents=texts[start:end],
                metadatas=metadatas[start:end],  # type: ignore[arg-type]
                embeddings=embeddings[start:end],  # type: ignore[arg-type]
                ids=batch_ids,
            )
            all_ids.extend(batch_ids)

        return all_ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def add_texts(
        self,
        texts: list[str],
        metadata: Optional[List[dict]] = None,
    ) -> list[str]:
        return self.vector_store.add_texts(texts, metadatas=metadata)

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
        # TODO: delete the chroma folder and add configuration to delete the folder
        store = self.vector_store
        store.delete_collection()
        self._vector_store = None

        _ = self.vector_store

    def document_count(self) -> int:
        return self.vector_store._collection.count()
