import sys
from argparse import Namespace

from ..constants import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    SAMPLE_DOCUMENTS_PATH,
)
from .args import args as cli_args
from .parser import CLIParser
from .runner import CLIContext, run_demo, run_interactive, run_single_query


def _build_context(ns: Namespace) -> CLIContext:
    """
    Builds a context from the parsed Namespace
    """

    return CLIContext(
        config_path=getattr(ns, "config_path", DEFAULT_CONFIG_PATH),
        data_path=getattr(ns, "data_path", DEFAULT_DATA_PATH),
        sample_documents_path=SAMPLE_DOCUMENTS_PATH,
    )


def main(argv: list[str] | None = None) -> None:
    parser = CLIParser()
    parser.add_arguments(cli_args)
    ns = parser.parse_args(argv)

    if getattr(ns, "version", False):
        print(f"{APP_NAME} {APP_VERSION}")
        sys.exit(0)

    query_str = getattr(ns, "query", None)
    purge_db = getattr(ns, "purge_db", False)

    if query_str is not None:
        ctx = _build_context(ns)
        run_single_query(
            ctx,
            query=query_str,
            should_clear=getattr(ns, "clear_db", False),
            should_purge=purge_db,
            should_load_documents=getattr(ns, "load_docs", False),
            top_k=getattr(ns, "top_k", None),
            verbose=getattr(ns, "verbose", False),
        )
        return

    ctx = _build_context(ns)
    should_clear = getattr(ns, "clear_db", False)
    load_docs = getattr(ns, "load_docs", False)
    is_interactive = getattr(ns, "interactive", False)
    top_k = getattr(ns, "top_k", None)
    verbose = getattr(ns, "verbose", False)

    should_load = load_docs if is_interactive else True

    if is_interactive:
        run_interactive(
            ctx,
            should_clear=should_clear,
            should_purge=purge_db,
            should_load_documents=should_load,
            top_k=top_k,
            verbose=verbose,
        )
    else:
        run_demo(
            ctx,
            should_clear=should_clear,
            should_purge=purge_db,
            should_load_documents=should_load,
            top_k=top_k,
            verbose=verbose,
        )
