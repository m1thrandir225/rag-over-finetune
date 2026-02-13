import json
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from pydantic import ValidationError
from vezilka_schemas import Record

from .document_type import DocumentType


class DocumentImporter:
    """
    Imports documents based on specific type to the vector store.
    """

    def __init__(self) -> None:
        pass

    def load_text_file(self, file_path: str, encoding: str = "utf-8") -> list[Document]:
        loader = TextLoader(file_path, encoding=encoding)
        return loader.load()

    def load_directory(
        self, directory: str, glob_pattern: str = "**/*.txt", encoding: str = "utf-8"
    ) -> list[Document]:
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": encoding},
        )
        return loader.load()

    def load_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_docx(self, file_path: str) -> list[Document]:
        loader = Docx2txtLoader(file_path)
        return loader.load()

    @staticmethod
    def load_json_records(
        file_path: str,
        encoding: str = "utf-8",
    ) -> list[Record]:
        """
        Load records from a JSON file, validating each against the Record schema.
        """
        with open(file_path, "r", encoding=encoding) as f:
            data = json.load(f)

        raw_records: list[dict]
        if isinstance(data, list):
            raw_records = data
        elif isinstance(data, dict):
            raw_records = [data]
        else:
            raise ValueError(
                f"Unexpected JSON structure in {file_path}: "
                f"expected array or object, got {type(data).__name__}"
            )

        records: list[Record] = []
        for i, raw in enumerate(raw_records):
            try:
                records.append(Record.model_validate(raw))
            except ValidationError as e:
                raise ValueError(
                    f"Invalid record at index {i} in {file_path}: {e}"
                ) from e
        return records

    @staticmethod
    def load_jsonl_records(
        file_path: str,
        encoding: str = "utf-8",
    ) -> list[Record]:
        """
        Load records from a .jsonl file, validating each against the Record schema.
        """
        records: list[Record] = []
        with open(file_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {file_path}: {e}"
                    ) from e
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Expected object on line {line_num} in {file_path}, "
                        f"got {type(raw).__name__}"
                    )
                try:
                    records.append(Record.model_validate(raw))
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid record on line {line_num} in {file_path}: {e}"
                    ) from e
        return records

    def load_data_folder(
        self,
        folder_path: str,
        encoding: str = "utf-8",
    ) -> list[Record]:
        """
        Scan a folder for .json and .jsonl files and return all records.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            return []

        records: list[Record] = []

        files = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in (".json", ".jsonl")
        )

        for file_path in files:
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".json":
                    file_records = self.load_json_records(
                        str(file_path), encoding=encoding
                    )
                else:
                    file_records = self.load_jsonl_records(
                        str(file_path), encoding=encoding
                    )
                print(f"  Loaded {len(file_records)} records from {file_path.name}")
                records.extend(file_records)
            except (ValueError, json.JSONDecodeError, ValidationError) as e:
                print(f"  Warning: skipping {file_path.name}: {e}")

        return records

    def import_document(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        encoding: str = "utf-8",
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
                return self.load_text_file(file_path, encoding=encoding)
        else:
            match document_type:
                case DocumentType.JSON:
                    return self.load_text_file(file_path, encoding=encoding)
                case DocumentType.CSV:
                    return self.load_text_file(file_path, encoding=encoding)
                case _:
                    return self.load_text_file(file_path, encoding=encoding)

    def import_documents(
        self,
        file_paths: list[str],
        document_type: Optional[DocumentType] = None,
        encoding: str = "utf-8",
    ) -> list[Document]:
        """
        Import multiple documents.
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.import_document(file_path, document_type, encoding)
            all_documents.extend(documents)
        return all_documents
