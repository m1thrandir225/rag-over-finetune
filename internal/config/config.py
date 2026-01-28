from dataclasses import dataclass


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
