import argparse

from .args import Argument


class CLIParser:
    """
    Responsible for parsing the CLI arguments and options
    """

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="CLI for POC rag-over-finetune"
        )

    @property
    def parser(self) -> argparse.ArgumentParser:
        return self._parser

    def parse_args(self) -> argparse.Namespace:
        return self._parser.parse_args()

    def add_arguments(self, arguments: list[Argument]) -> None:
        for argument in arguments:
            argument.add_to_parser(self._parser)
