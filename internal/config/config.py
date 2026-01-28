from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Configuration for the RAG system
    """

    embedding_model: str
    llm_model: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3
    chroma_collection_name: str = "vector_collection"
    chroma_persist_dir: str = "./chroma_db"
