from typing import Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config import Config


class EmbeddingService:
    def __init__(self, config: Config):
        self._config = config
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Lazy initialization of embedding model.
        """

        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create and configure the embedding model.
        """

        model_kwargs: dict[str, Any] = {"device": self._config.embedding_device}
        encode_kwargs: dict[str, Any] = {
            "normalize_embeddings": self._config.normalize_embeddings
        }

        # e5 models need query prefix in prompt
        if "e5" in self._config.embedding_model.lower():
            encode_kwargs["prompt"] = "query: "

        return HuggingFaceEmbeddings(
            model_name=self._config.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True,
        )

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        """
        return self.embeddings.embed_documents(texts)
