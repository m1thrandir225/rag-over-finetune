from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import Config

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class ChunkMode(Enum):
    TEXT = "text"
    LENGTH = "length"
    DOCUMENT = "document"
    SEMANTIC = "semantic"


class DocumentType(Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class LengthType(Enum):
    CHAR = "char"
    TOKEN = "token"


class Chunker:
    """
    Chunker is responsible for splitting 'chunking' text into smaller parts which then are transformed into embeddings and
    stored in a vector store.

    The goal is to support all text splitting structures from Langchain with comprehensive options passed by the 'config.json' file.
    """

    def __init__(
        self,
        config: Config,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        self._config = config
        self._embeddings = embeddings
        self._splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._semantic_splitter: Optional[object] = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._config.chunk_overlap

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        if self._splitter is None:
            self._splitter = self._create_splitter()
        return self._splitter

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        separators = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str, mode: Optional[ChunkMode] = None) -> list[str]:
        """
        Splits a text into overlapping chunks.
        Supports all three kinds of text splitting from Langchain.
        """

        if mode is None:
            mode = ChunkMode(self._config.chunk_mode)

        match mode:
            case ChunkMode.TEXT:
                return self._split_text_mode(text)
            case ChunkMode.DOCUMENT:
                return self._split_document_mode(text)
            case ChunkMode.LENGTH:
                return self._split_length_mode(text)
            case ChunkMode.SEMANTIC:
                return self._split_semantic_mode(text)

    def _split_text_mode(self, text: str) -> list[str]:
        """
        Split text using RecursiveCharacterTextSplitter with multilingual separators
        """

        return self.splitter.split_text(text)

    def _split_length_mode(self, text: str) -> list[str]:
        """
        Split text using CharacterTextSplitter based on config
        """

        if not self._config.chunk_length_options:
            length_type = LengthType.CHAR
            separator = "\n\n"
            is_separator_regex = False
        else:
            len_opts = self._config.chunk_length_options
            length_type = LengthType(len_opts.mode)

            if length_type == LengthType.CHAR:
                char_opts = len_opts.char_mode_options
                separator = char_opts.get("separator", "\n\n")
                is_separator_regex = char_opts.get("is_separator_regex", False)
            else:
                token_opts = len_opts.token_mode_options
                encoding_name = token_opts.get("encoding_name", "cl100k_base")

                splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name=encoding_name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                return splitter.split_text(text)

        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=is_separator_regex,
        )
        return splitter.split_text(text)

    def _split_document_mode(self, text: str) -> list[str]:
        pass  # TODO: implement document mode

    @property
    def semantic_splitter(self) -> object:
        if self._semantic_splitter is None:
            self._semantic_splitter = self._create_semantic_splitter()
        return self._semantic_splitter

    def _create_semantic_splitter(self) -> object:
        if self._embeddings is None:
            raise ValueError(
                "Semantic chunking requires an embeddings instance. "
                "Pass embeddings= when constructing the Chunker or use a different chunk_mode."
            )

        from langchain_experimental.text_splitter import SemanticChunker

        kwargs: dict = {
            "embeddings": self._embeddings,
            "breakpoint_threshold_type": self._config.semantic_breakpoint_type,
        }
        if self._config.semantic_breakpoint_amount is not None:
            kwargs["breakpoint_threshold_amount"] = (
                self._config.semantic_breakpoint_amount
            )

        return SemanticChunker(**kwargs)

    def _split_semantic_mode(self, text: str) -> list[str]:
        """
        Split text using embedding similarity to detect topic boundaries.
        Falls back to TEXT mode for short texts that can't be semantically split.
        """
        try:
            return self.semantic_splitter.split_text(text)  # type: ignore[union-attr]
        except (IndexError, ValueError):
            return self._split_text_mode(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks based on the configured chunk mode.
        Each chunk inherits its parent's metadata plus chunk_index,
        total_chunks, and parent_doc_id for traceability.
        """
        mode = ChunkMode(self._config.chunk_mode)
        all_chunks: list[Document] = []

        for doc in documents:
            if mode == ChunkMode.SEMANTIC:
                raw_chunks = self._split_semantic_mode(doc.page_content)
                doc_chunks = [
                    Document(page_content=chunk, metadata=doc.metadata.copy())
                    for chunk in raw_chunks
                ]
            else:
                doc_chunks = self.splitter.split_documents([doc])

            parent_id = doc.metadata.get("id", "")
            total = len(doc_chunks)
            for idx, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = total
                chunk.metadata["parent_doc_id"] = parent_id

            all_chunks.extend(doc_chunks)

        return all_chunks

    def split_text(self, text: str) -> list[str]:
        """
        Split a single text into chunks using the configured mode
        """

        return self.chunk_text(text)

    def create_documents(
        self, texts: list[str], metadatas: Optional[list[dict]] = None
    ) -> list[Document]:
        """
        Create Document objects text
        """

        if metadatas is None:
            metadatas = [{}] * len(texts)

        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
