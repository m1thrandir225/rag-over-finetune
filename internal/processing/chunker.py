from enum import Enum
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import Config


class ChunkMode(Enum):
    TEXT = "text"
    LENGTH = "length"
    DOCUMENT = "document"


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

    def __init__(self, config: Config) -> None:
        self._config = config
        self._splitter: Optional[RecursiveCharacterTextSplitter] = None

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
        separators = [
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
            ";",
            ",",
            " ",
            ""
        ]

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
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
        pass #TODO: implement document mode

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks
        """

        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        """
        Split a single text into chunks using the configured mode
        """

        return self.chunk_text(text)

    def create_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None
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
