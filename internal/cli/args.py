import argparse
from typing import Any, Optional


class Argument:
    def __init__(
        self,
        short_name: str,
        long_name: str,
        description: str,
        action: str | argparse.Action,
        required: bool = False,
        default: Optional[Any] = None,
    ) -> None:
        self.short_name: str = short_name
        self.long_name: str = long_name
        self.description: str = description
        self.required: bool = required
        self.action: str | argparse.Action = action
        self.default: Optional[Any] = default

    def __str__(self) -> str:
        return f"{self.short_name}, {self.long_name}, {self.description}, {self.required}, {self.default}"

    def __repr__(self) -> str:
        return f"Argument(short_name={self.short_name}, long_name={self.long_name}, description={self.description}, required={self.required}, default={self.default})"

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"-{self.short_name}",
            f"--{self.long_name}",
            help=self.description,
            required=self.required,
            default=self.default,
            action=self.action,
        )


args: list[Argument] = [
    Argument(
        short_name="i",
        long_name="interactive",
        description="Run in interactive mode",
        action="store_true",
        required=False,
    ),
    Argument(
        short_name="d",
        long_name="demo",
        description="Run in demo mode",
        action="store_true",
        required=False,
    ),
    Argument(
        short_name="config",
        long_name="config-path",
        description="Path to the config file",
        action="store",
        required=False,
        default="./config.json",
    ),
    Argument(
        short_name="data",
        long_name="data-path",
        description="Path to the data folder",
        action="store",
        required=False,
        default="./data",
    ),
    Argument(
        short_name="clear_db",
        long_name="clear-db",
        description="Clear the database",
        action="store_true",
        required=False,
    ),
    Argument(
        short_name="load_docs",
        long_name="load-docs",
        description="Load the documents",
        action="store_true",
        required=False,
    ),
]
