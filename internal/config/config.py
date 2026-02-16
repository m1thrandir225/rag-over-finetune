from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMProvider(str, Enum):
    """
    Currently supported LLM providers
    Ollama is local; OpenAI, Anthropic, Google, and OpenRouter are remote.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class EmbeddingProvider(str, Enum):
    """
    Supported embedding providers.
    HuggingFace is local; OpenAI and OpenRouter are remote.
    """

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class ChunkDocumentOptions:
    json_mode_options: dict
    html_mode_options: dict
    markdown_mode_options: dict


@dataclass(frozen=True)
class ChunkLengthOptions:
    mode: str  # "char" or "token"
    char_mode_options: dict
    token_mode_options: dict


@dataclass(frozen=True)
class Config:
    """
    Configuration for the RAG system
    """

    embedding_model: str
    embedding_provider: EmbeddingProvider
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_collection_name: str
    chroma_persist_dir: str
    chunk_mode: str  # "text", "length", or "document"
    chunk_document_options: Optional[ChunkDocumentOptions]
    chunk_length_options: Optional[ChunkLengthOptions]

    # LLM Configuration
    llm_url: str
    llm_temperature: float
    llm_max_tokens: Optional[int]
    system_prompt: str
    llm_provider: LLMProvider

    prompt_template: str

    # Embedding Manager
    embedding_device: str
    normalize_embeddings: bool
    embedding_batch_size: int
