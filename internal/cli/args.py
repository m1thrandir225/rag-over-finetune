import argparse
from typing import Any, Optional

from ..constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH


class Argument:
    """
    Wrapper class for argparse arguments
    """

    def __init__(
        self,
        long_name: str,
        description: str,
        *,
        short_name: Optional[str] = None,
        action: str | type[argparse.Action] = "store",
        required: bool = False,
        default: Optional[Any] = None,
        dest: Optional[str] = None,
        arg_type: Optional[type] = None,
    ) -> None:
        self.short_name: Optional[str] = short_name
        self.long_name: str = long_name
        self.description: str = description
        self.required: bool = required
        self.action: str | type[argparse.Action] = action
        self.default: Optional[Any] = default
        self.dest: Optional[str] = dest
        self.arg_type: Optional[type] = arg_type

    def __str__(self) -> str:
        return f"{self.short_name}, {self.long_name}, {self.description}, {self.required}, {self.default}"

    def __repr__(self) -> str:
        return f"Argument(short_name={self.short_name}, long_name={self.long_name}, description={self.description}, required={self.required}, default={self.default})"

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        option_strings = [f"--{self.long_name}"]
        if self.short_name:
            option_strings.insert(0, f"-{self.short_name}")

        kwargs: dict[str, Any] = {
            "help": self.description,
            "required": self.required,
            "action": self.action,
        }
        if self.default is not None and self.action not in (
            "store_true",
            "store_false",
        ):
            kwargs["default"] = self.default
        if self.dest is not None:
            kwargs["dest"] = self.dest
        if self.arg_type is not None:
            kwargs["type"] = self.arg_type

        parser.add_argument(*option_strings, **kwargs)


args: list[Argument] = [
    Argument(
        "interactive",
        "Run in interactive mode",
        short_name="i",
        action="store_true",
    ),
    Argument(
        "config-path",
        "Path to the config file",
        short_name="c",
        default=DEFAULT_CONFIG_PATH,
    ),
    Argument(
        "data-path",
        "Path to the data folder",
        short_name="D",
        default=DEFAULT_DATA_PATH,
    ),
    Argument(
        "clear-db",
        "Clear the database before loading",
        action="store_true",
    ),
    Argument(
        "load-docs",
        "Load documents from data folder and sample data",
        action="store_true",
    ),
    Argument(
        "version",
        "Show version and exit",
        short_name="V",
        action="store_true",
    ),
    Argument(
        "verbose",
        "Enable verbose output",
        short_name="v",
        action="store_true",
    ),
    Argument(
        "top-k",
        "Number of chunks to retrieve for RAG (overrides config)",
        short_name="k",
        default=None,
        dest="top_k",
        arg_type=int,
    ),
    Argument(
        "query",
        "Run a single query and exit",
        short_name="q",
        default=None,
        dest="query",
    ),
]
