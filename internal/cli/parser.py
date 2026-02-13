import argparse

from .args import Argument


class CLIParser:
    """
    Parses the CLI arguments and options
    """

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="Vezilka RAG CLI"
        )

    @property
    def parser(self) -> argparse.ArgumentParser:
        return self._parser

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        return self._parser.parse_args(argv)

    def add_arguments(self, arguments: list[Argument]) -> None:
        for argument in arguments:
            argument.add_to_parser(self._parser)
