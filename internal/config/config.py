from dataclasses import dataclass
from typing import Optional


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
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_collection_name: str
    chroma_persist_dir: str
    chunk_mode: str  # "text", "length", or "document"
    chunk_document_options: Optional[ChunkDocumentOptions] = None
    chunk_length_options: Optional[ChunkLengthOptions] = None

    # LLM Configuration
    llm_url: str = "http://localhost:11434"
    llm_temperature: float = 0.7
    system_prompt: str = """Ти си помошник кој одговара на прашања.
Користи го САМО дадениот контекст за да одговориш на прашањето.
Ако одговорот не е во контекстот, кажи дека немаш доволно информации.
Одговарај концизно и точно."""

    prompt_template: str = """
    Контекст:
{context}

Прашање: {question}

Одговор:
"""

    # Embedding Manager
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True
