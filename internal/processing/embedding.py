from typing import Any, Optional, Union

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from ..config import Config, EmbeddingProvider, get_required_env_var
from .openrouter import OpenRouterSDKEmbeddings


class EmbeddingService:
    """
    Responsible for creating and configuring the embedding model.
    Supports HuggingFace (local), OpenAI, and OpenRouter (remote API).
    """

    def __init__(self, config: Config):
        self._config = config
        self._embeddings: Optional[
            Union[HuggingFaceEmbeddings, OpenAIEmbeddings, OpenRouterSDKEmbeddings]
        ] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def embeddings(
        self,
    ) -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings, OpenRouterSDKEmbeddings]:
        """
        Lazy initialization of embedding model.
        """
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(
        self,
    ) -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings, OpenRouterSDKEmbeddings]:
        """
        Create and configure the embedding model based on embedding_provider.
        """
        provider = self._config.embedding_provider

        if provider == EmbeddingProvider.HUGGINGFACE:
            return self._create_huggingface()
        elif provider == EmbeddingProvider.OPENAI:
            return self._create_openai()
        elif provider == EmbeddingProvider.OPENROUTER:
            return self._create_openrouter()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_huggingface(self) -> HuggingFaceEmbeddings:
        """
        Create HuggingFace embedding model
        """
        model_kwargs: dict[str, Any] = {"device": self._config.embedding_device}

        encode_kwargs: dict[str, Any] = {
            "normalize_embeddings": self._config.normalize_embeddings,
            "batch_size": self._config.embedding_batch_size,
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

    def _create_openai(self) -> OpenAIEmbeddings:
        """
        Create OpenAI embedding model
        """
        api_key = get_required_env_var("OPENAI_API_KEY")
        return OpenAIEmbeddings(
            model=self._config.embedding_model,
            api_key=api_key,
            chunk_size=self._config.embedding_batch_size,
        )

    def _create_openrouter(self) -> OpenRouterSDKEmbeddings:
        """
        Create OpenRouter embedding model
        Supported: https://openrouter.ai/models?output_modalities=embeddings
        """
        api_key = get_required_env_var("OPENROUTER_API_KEY")
        return OpenRouterSDKEmbeddings(
            api_key=api_key,
            model=self._config.embedding_model,
            batch_size=self._config.embedding_batch_size,
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
