from enum import Enum

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)


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
    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = overlap

    def chunk_text(self, text: str, mode: ChunkMode = ChunkMode.TEXT) -> list[str]:
        """
        Splits a text into overlapping chunks.
        This is character based chunking.
        """
        match mode:
            case ChunkMode.TEXT:
                return self.__split_text_mode(text)
            case ChunkMode.DOCUMENT:
                return self.__split_document_mode(text)
            case ChunkMode.LENGTH:
                return self.__split_length_mode(text)

    def __split_text_mode(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def __split_length_mode(
        self,
        text: str,
        type: LengthType = LengthType.CHAR,
        separator: str = "\n\n",
    ) -> list[str]:
        if type == LengthType.CHAR:
            # TODO: add options to customize lenght function and regex
            splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            # TODO: add options to customize encoding_name
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        return splitter.split_text(text)

    def __split_document_mode(
        self, text: str, type: DocumentType = DocumentType.JSON
    ) -> list[str]:
        match type:
            case DocumentType.JSON:
                splitter = RecursiveJsonSplitter(max_chunk_size=self.chunk_size)
            case DocumentType.HTML:
                pass
            case DocumentType.MARKDOWN:
                pass

        # TODO: return proper chunks from each type
        return []
