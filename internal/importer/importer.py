import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from vezilka_schemas import Record

from .document_type import DocumentType


class DocumentImporter:
    """
    Imports Record documents from JSON and JSONL files.
    """

    def _load_json_records(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> list[Record]:
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

    def _load_jsonl_records(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> list[Record]:
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

    def import_records(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        encoding: str = "utf-8",
    ) -> list[Record]:
        """
        Import Record objects from a JSON or JSONL file.
        Auto-detects format from extension when document_type is None.
        """
        if document_type is None:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".jsonl":
                document_type = DocumentType.JSONL
            elif suffix == ".json":
                document_type = DocumentType.JSON
            else:
                raise ValueError(
                    f"Cannot auto-detect record format for '{file_path}': "
                    f"expected .json or .jsonl extension."
                )

        match document_type:
            case DocumentType.JSON:
                return self._load_json_records(file_path, encoding=encoding)
            case DocumentType.JSONL:
                return self._load_jsonl_records(file_path, encoding=encoding)
            case _:
                raise ValueError(
                    f"Unsupported document type for record import: {document_type.value}"
                )

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
            try:
                file_records = self.import_records(str(file_path), encoding=encoding)
                print(f"  Loaded {len(file_records)} records from {file_path.name}")
                records.extend(file_records)
            except (ValueError, json.JSONDecodeError, ValidationError) as e:
                print(f"  Warning: skipping {file_path.name}: {e}")

        return records
