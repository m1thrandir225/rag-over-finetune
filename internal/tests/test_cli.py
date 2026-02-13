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


class TestParseDocuments:
    def test_parse_documents_basic(self) -> None:
        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": ""},
        ]
        texts, metadatas = parse_documents(docs)
        assert texts == ["Doc 1\n\nContent 1", "Doc 2"]
        assert len(metadatas) == 2
        assert metadatas[0]["title"] == "Doc 1"
        assert metadatas[1]["title"] == "Doc 2"

    def test_parse_documents_with_metadata(self) -> None:
        docs = [
            {
                "title": "Test",
                "content": "Body",
                "id": "123",
                "page_url": "https://example.com",
                "categories": ["a", "b"],
            }
        ]
        texts, metadatas = parse_documents(docs)
        assert metadatas[0]["id"] == "123"
        assert metadatas[0]["page_url"] == "https://example.com"
        assert metadatas[0]["categories"] == "a, b"


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
