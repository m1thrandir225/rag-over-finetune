from dataclasses import dataclass


@dataclass(frozen=True)
class CLIContext:
    """
    Current running context for the CLI
    """

    config_path: str
    data_path: str
    sample_documents_path: str
