from vezilka_schemas import Record

from internal.cli.args import Argument
from internal.cli.args import args as cli_args
from internal.cli.parser import CLIParser
from internal.cli.runner import CLIContext, parse_documents
from internal.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    SAMPLE_DOCUMENTS_PATH,
)


class TestArgument:
    def test_add_to_parser_store_true(self) -> None:
        parser = CLIParser()
        arg = Argument("test-flag", "A test flag", action="store_true")
        arg.add_to_parser(parser.parser)
        ns = parser.parse_args(["--test-flag"])
        assert ns.test_flag is True

    def test_add_to_parser_with_short(self) -> None:
        parser = CLIParser()
        arg = Argument(
            "config-path", "Config path", short_name="c", default=DEFAULT_CONFIG_PATH
        )
        arg.add_to_parser(parser.parser)
        ns = parser.parse_args([])
        assert ns.config_path == DEFAULT_CONFIG_PATH
        ns2 = parser.parse_args(["-c", "/custom/config.json"])
        assert ns2.config_path == "/custom/config.json"


class TestCLIParser:
    def test_parse_args_with_argv(self) -> None:
        parser = CLIParser()
        parser.add_arguments(cli_args)
        ns = parser.parse_args(["-i", "-c", "/my/config.json"])
        assert ns.interactive is True
        assert ns.config_path == "/my/config.json"

    def test_parse_args_defaults(self) -> None:
        parser = CLIParser()
        parser.add_arguments(cli_args)
        ns = parser.parse_args([])
        assert ns.interactive is False
        assert ns.config_path == DEFAULT_CONFIG_PATH
        assert ns.data_path == DEFAULT_DATA_PATH
        assert ns.clear_db is False
        assert ns.load_docs is False
        assert ns.purge_db is False
        assert ns.top_k is None
        assert ns.query is None

    def test_parse_args_version_verbose_topk_query(self) -> None:
        parser = CLIParser()
        parser.add_arguments(cli_args)
        ns = parser.parse_args(["-V", "-v", "-k", "5", "-q", "test question"])
        assert ns.version is True
        assert ns.verbose is True
        assert ns.top_k == 5
        assert ns.query == "test question"

    def test_parse_args_purge_db(self) -> None:
        parser = CLIParser()
        parser.add_arguments(cli_args)
        ns = parser.parse_args(["--purge-db"])
        assert ns.purge_db is True


class TestParseDocuments:
    def test_parse_documents_basic(self) -> None:
        docs = [
            Record.model_validate(
                {
                    "id": "record-1",
                    "text": "Doc 1\n\nContent 1",
                    "type": "narrative",
                    "last_modified_at": "2026-01-01T00:00:00",
                    "meta": {
                        "source": "example.com",
                        "url": None,
                        "tags": [],
                        "labels": [],
                        "scraped_at": "2026-01-01T00:00:00",
                    },
                }
            ),
            Record.model_validate(
                {
                    "id": "record-2",
                    "text": "Doc 2",
                    "type": "narrative",
                    "last_modified_at": "2026-01-02T00:00:00",
                    "meta": {
                        "source": "example.com",
                        "url": None,
                        "tags": [],
                        "labels": [],
                        "scraped_at": "2026-01-02T00:00:00",
                    },
                }
            ),
        ]
        texts, metadatas = parse_documents(docs)
        assert texts == ["Doc 1\n\nContent 1", "Doc 2"]
        assert len(metadatas) == 2
        assert metadatas[0]["title"] == "Doc 1"
        assert metadatas[1]["title"] == "Doc 2"

    def test_parse_documents_with_metadata(self) -> None:
        docs = [
            Record.model_validate(
                {
                    "id": "123",
                    "text": "Test\n\nBody",
                    "type": "narrative",
                    "last_modified_at": "2026-01-01T00:00:00",
                    "meta": {
                        "source": "example.com",
                        "url": "https://example.com",
                        "tags": ["a", "b"],
                        "labels": [],
                        "scraped_at": "2026-01-01T00:00:00",
                    },
                }
            )
        ]
        texts, metadatas = parse_documents(docs)
        assert metadatas[0]["id"] == "123"
        assert metadatas[0]["source"] == "example.com"
        assert metadatas[0]["url"] == "https://example.com"
        assert metadatas[0]["tags"] == ["a", "b"]
        assert metadatas[0]["labels"] == []
        assert metadatas[0]["scraped_at"] == "2026-01-01T00:00:00"
        assert metadatas[0]["site_url"] == "example.com"
        assert metadatas[0]["page_url"] == "https://example.com"
        assert metadatas[0]["categories"] == "a, b"

    def test_parse_documents_record_input(self) -> None:
        docs = [
            Record.model_validate(
                {
                    "id": "record-1",
                    "text": "Record Title\n\nRecord Body",
                    "type": "narrative",
                    "last_modified_at": "2026-01-01T00:00:00",
                    "meta": {
                        "source": "example.com",
                        "url": "https://example.com/record",
                        "tags": ["news", "tech"],
                        "labels": [],
                        "scraped_at": "2026-01-01T00:00:00",
                    },
                }
            )
        ]
        texts, metadatas = parse_documents(docs)
        assert texts == ["Record Title\n\nRecord Body"]
        assert metadatas[0]["id"] == "record-1"
        assert metadatas[0]["site_url"] == "example.com"
        assert metadatas[0]["page_url"] == "https://example.com/record"
        assert metadatas[0]["categories"] == "news, tech"


class TestCLIContext:
    def test_context_creation(self) -> None:
        ctx = CLIContext(
            config_path="/cfg.json",
            data_path="/data",
            sample_documents_path=SAMPLE_DOCUMENTS_PATH,
        )
        assert ctx.config_path == "/cfg.json"
        assert ctx.data_path == "/data"
        assert "sample_documents" in ctx.sample_documents_path
