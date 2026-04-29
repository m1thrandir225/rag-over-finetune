#!/usr/bin/env python3
"""
Retrieval evaluation harness for comparing query-transform strategies.

Usage
-----
    # Ensure Qdrant is running and the collection is populated.
    python -m scripts.eval_retrieval --dataset eval_queries.jsonl

    # Override config path
    python -m scripts.eval_retrieval --config config.json --dataset eval_queries.jsonl --k 5

Dataset format (JSONL)
----------------------
Each line is a JSON object with:

    {
        "query": "Кој е главниот град на Македонија?",
        "relevant_snippets": ["Скопје е главен град", "столица на Северна Македонија"]
    }

'relevant_snippets' are sub-strings expected to appear in at least one
retrieved chunk's 'page_content'.  A retrieved document counts as a hit
if *any* relevant snippet is a sub-string of its content

If no dataset file is provided, the script falls back to the queries in
'DEFAULT_TEST_QUERIES' from constants
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# TODO: maybe include not as a script?? and in the chain

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from internal.config import ConfigLoader  # noqa: E402
from internal.constants import (
    DEFAULT_CONFIG_PATH,  # noqa: E402
    DEFAULT_TEST_QUERIES,
)
from internal.llm import LLMService  # noqa: E402
from internal.processing import (
    EmbeddingService,  # noqa: E402
    QueryTransformer,
)
from internal.store import VectorStoreManager  # noqa: E402

logger = logging.getLogger("eval_retrieval")


@dataclass
class EvalQuery:
    query: str
    relevant_snippets: list[str]


@dataclass
class StrategyResult:
    name: str
    recall_at_k: float
    mrr_at_k: float
    avg_latency_ms: float
    avg_docs_returned: float


def _is_hit(page_content: str, relevant_snippets: list[str]) -> bool:
    """
    True if any relevant snippet appears in the page_content (case-insensitive)
    """

    lc = page_content.lower()
    return any(s.lower() in lc for s in relevant_snippets)


def compute_recall_at_k(
    retrieved_contents: list[str],
    relevant_snippets: list[str],
) -> float:
    """
    Fraction of relevant_snippets that are found in at least one retrieved doc
    """

    if not relevant_snippets:
        return 0.0
    found = sum(
        1
        for s in relevant_snippets
        if any(s.lower() in rc.lower() for rc in retrieved_contents)
    )
    return found / len(relevant_snippets)


def compute_mrr(
    retrieved_contents: list[str],
    relevant_snippets: list[str],
) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first hit, or 0 if no hit
    """

    for rank, rc in enumerate(retrieved_contents, start=1):
        if _is_hit(rc, relevant_snippets):
            return 1.0 / rank
    return 0.0


STRATEGIES: dict[str, list[str]] = {
    "baseline": [],
    "multi_query": ["multi_query"],
    "step_back+multi_query": ["step_back", "multi_query"],
    "hyde": ["hyde"],
    "hyde+multi_query": ["hyde", "multi_query"],
}


def _run_strategy(
    strategy_name: str,
    enabled_transforms: list[str],
    queries: list[EvalQuery],
    vector_store: VectorStoreManager,
    embedding_service: EmbeddingService,
    llm_service: LLMService,
    k: int,
    max_generated_queries: int,
) -> StrategyResult:
    transformer = QueryTransformer(
        llm=llm_service.llm,
        enabled_transforms=enabled_transforms,
        max_generated_queries=max_generated_queries,
        gate_enabled=False,
        hyde_include_original_query=True,
    )

    recalls: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []
    doc_counts: list[int] = []

    for eq in queries:
        t0 = time.perf_counter()
        result = transformer.transform(eq.query)

        all_docs = []
        seen: set[str] = set()

        for idx, q in enumerate(result.expanded_queries):
            if idx in result.use_embedding_of:
                emb = embedding_service.embed_query(q)
                docs = vector_store.similarity_search_by_vector(embedding=emb, k=k)
            else:
                docs = vector_store.similarity_search(query=q, k=k)

            for doc in docs:
                key = doc.page_content[:300]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        all_docs = all_docs[:k]
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        contents = [d.page_content for d in all_docs]

        if eq.relevant_snippets:
            recalls.append(compute_recall_at_k(contents, eq.relevant_snippets))
            mrrs.append(compute_mrr(contents, eq.relevant_snippets))

        latencies.append(elapsed_ms)
        doc_counts.append(len(all_docs))

    avg_recall = sum(recalls) / len(recalls) if recalls else float("nan")
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else float("nan")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_docs = sum(doc_counts) / len(doc_counts) if doc_counts else 0.0

    return StrategyResult(
        name=strategy_name,
        recall_at_k=avg_recall,
        mrr_at_k=avg_mrr,
        avg_latency_ms=avg_latency,
        avg_docs_returned=avg_docs,
    )


def load_dataset(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: %s", line_num, exc)
                continue
            queries.append(
                EvalQuery(
                    query=obj["query"],
                    relevant_snippets=obj.get("relevant_snippets", []),
                )
            )
    return queries


def _default_queries() -> list[EvalQuery]:
    return [EvalQuery(query=q, relevant_snippets=[]) for q in DEFAULT_TEST_QUERIES]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval strategies (query transforms)."
    )
    _ = parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json",
    )
    _ = parser.add_argument(
        "--dataset",
        default=None,
        help="Path to JSONL evaluation dataset. Falls back to default test queries.",
    )
    _ = parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (Recall@k, MRR@k).",
    )
    _ = parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help=f"Strategies to evaluate. Choices: {list(STRATEGIES.keys())}",
    )
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = ConfigLoader(args.config).load_config()
    embedding_service = EmbeddingService(config=config)
    vector_store = VectorStoreManager(
        config=config, embedding_service=embedding_service
    )
    llm_service = LLMService(config=config)

    doc_count = vector_store.document_count()
    logger.info("Vector store contains %d documents.", doc_count)
    if doc_count == 0:
        logger.error("Store is empty. Populate it before running eval.")
        sys.exit(1)

    # Load dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("Dataset file not found: %s", dataset_path)
            sys.exit(1)
        queries = load_dataset(dataset_path)
        logger.info("Loaded %d evaluation queries from %s", len(queries), dataset_path)
    else:
        queries = _default_queries()
        logger.info(
            "No dataset provided; using %d default test queries (no relevance judgements).",
            len(queries),
        )

    strategy_names = args.strategies or list(STRATEGIES.keys())
    for s in strategy_names:
        if s not in STRATEGIES:
            logger.error("Unknown strategy: %s. Known: %s", s, list(STRATEGIES.keys()))
            sys.exit(1)

    k = args.k
    max_gen = config.max_generated_queries

    results: list[StrategyResult] = []
    for name in strategy_names:
        logger.info("--- Evaluating strategy: %s ---", name)
        sr = _run_strategy(
            strategy_name=name,
            enabled_transforms=STRATEGIES[name],
            queries=queries,
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_service=llm_service,
            k=k,
            max_generated_queries=max_gen,
        )
        results.append(sr)
        logger.info(
            "  Recall@%d=%.4f  MRR@%d=%.4f  avg_latency=%.1f ms  avg_docs=%.1f",
            k,
            sr.recall_at_k,
            k,
            sr.mrr_at_k,
            sr.avg_latency_ms,
            sr.avg_docs_returned,
        )

    print("\n" + "=" * 80)
    print(
        f"{'Strategy':<28} {'Recall@' + str(k):>10} {'MRR@' + str(k):>10} {'Latency(ms)':>12} {'Avg Docs':>10}"
    )
    print("-" * 80)
    for sr in results:
        print(
            f"{sr.name:<28} {sr.recall_at_k:>10.4f} {sr.mrr_at_k:>10.4f} "
            f"{sr.avg_latency_ms:>12.1f} {sr.avg_docs_returned:>10.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
