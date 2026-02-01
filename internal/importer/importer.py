from typing import Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from .document_type import DocumentType


class DocumentImporter:
    """
    Imports documents based on specific type to the vector store.
    """

    def __init__(self) -> None:
        pass

    def load_text_file(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> list[Document]:
        loader = TextLoader(file_path, encoding=encoding)
        return loader.load()

    def load_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.txt",
        encoding: str = "utf-8"
    ) -> list[Document]:
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": encoding}
        )
        return loader.load()

    def load_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_docx(self, file_path: str) -> list[Document]:
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def import_document(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        encoding: str = "utf-8"
    ) -> list[Document]:
        """
        Import a single document based on file extension or explicit type.
        """

        path = Path(file_path)
        
        if document_type is None:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return self.load_pdf(file_path)
            elif suffix in [".docx", ".doc"]:
                return self.load_docx(file_path)
            elif suffix == ".txt":
                return self.load_text_file(file_path, encoding=encoding)
            else:
                return self.load_text_file(file_path, encoding=encoding) # default to text loader for unknown
        else:
            match document_type:
                case DocumentType.JSON:
                    return self.load_text_file(file_path, encoding=encoding) # TODO: implement JSONLoader
                case DocumentType.CSV:
                    return self.load_text_file(file_path, encoding=encoding) # TODO: implement CSVLoader
                case _:
                    return self.load_text_file(file_path, encoding=encoding)

    def import_documents(
        self,
        file_paths: list[str],
        document_type: Optional[DocumentType] = None,
        encoding: str = "utf-8"
    ) -> list[Document]:
        """
        Import multiple documents.
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.import_document(file_path, document_type, encoding)
            all_documents.extend(documents)
        return all_documents
