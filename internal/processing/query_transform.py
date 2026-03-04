from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel

from .query_transform_models import TransformResult, TransformStageResult
from .query_transform_prompts import (
    _COMPLEX_PATTERNS,
    _HYDE_PROMPT,
    _MIN_WORDS_FOR_TRANSFORM,
    _MULTI_QUERY_PROMPT,
    _STEP_BACK_PROMPT,
)

logger = logging.getLogger(__name__)


def should_transform(query: str, *, gate_enabled: bool = True) -> bool:
    """
    Lightweight, rule-based gate.

    Returns True when the query is complex enough to benefit from
    transformation.  When *gate_enabled* is False, always returns True.
    """
    if not gate_enabled:
        return True

    words = query.split()
    if len(words) < _MIN_WORDS_FOR_TRANSFORM:
        logger.debug(
            "Gate: skipping transforms (query too short: %d words)", len(words)
        )
        return False

    for pattern in _COMPLEX_PATTERNS:
        if pattern.search(query):
            logger.debug("Gate: enabling transforms (complex pattern match)")
            return True

    # Default: allow transforms for medium+ length queries
    if len(words) >= 5:
        return True

    logger.debug("Gate: skipping transforms (short/simple query)")
    return False


class QueryTransformer:
    """
    Composable query-transformation pipeline.

    Parameters
    ----------
    llm : BaseChatModel
        The language model used to generate query variants.
    enabled_transforms : list[str]
        Ordered list of transform names to apply, e.g.
        ``["step_back", "multi_query"]``.  Executed left-to-right.
        If empty or ``["none"]``, no transformation is performed.
    max_generated_queries : int
        How many alternative queries multi_query should generate.
    gate_enabled : bool
        Whether to run the lightweight gate before transforming.
    timeout_ms : int
        Per-LLM-call soft timeout (currently advisory / for logging).
    hyde_include_original_query : bool
        If True (default), the original query is included alongside HyDE-
        generated hypothetical documents for dual-channel retrieval.
        If False, only the hypothetical document embedding is used

    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        enabled_transforms: list[str] | None = None,
        max_generated_queries: int = 2,
        gate_enabled: bool = True,
        timeout_ms: int = 2_000,
        hyde_include_original_query: bool = True,
    ) -> None:
        self._llm = llm
        self._max_generated_queries = max_generated_queries
        self._gate_enabled = gate_enabled
        self._timeout_ms = timeout_ms
        self._hyde_include_original_query = hyde_include_original_query

        if enabled_transforms is not None:
            self._transforms = [t for t in enabled_transforms if t != "none"]
        else:
            self._transforms = []

    def transform(self, query: str) -> TransformResult:
        t0 = time.perf_counter()

        if not query.strip():
            return TransformResult(
                original_query=query,
                expanded_queries=[query],
                total_duration_ms=0.0,
            )

        if not self._transforms:
            return TransformResult(
                original_query=query,
                expanded_queries=[query],
                total_duration_ms=_elapsed_ms(t0),
            )

        if not should_transform(query, gate_enabled=self._gate_enabled):
            logger.debug(
                "QueryTransform: gate skipped transforms for query=%r", query[:80]
            )
            return TransformResult(
                original_query=query,
                expanded_queries=[query],
                gate_skipped=True,
                total_duration_ms=_elapsed_ms(t0),
            )

        hyde_active = "hyde" in self._transforms
        include_original = not hyde_active or self._hyde_include_original_query

        accumulated_queries: list[str] = [query] if include_original else []
        use_embedding_of: set[int] = set()
        applied: list[str] = []
        stages: list[TransformStageResult] = []

        for transform_name in self._transforms:
            handler = self._get_handler(transform_name)
            if handler is None:
                logger.warning("Unknown transform '%s' -- skipping", transform_name)
                continue

            stage_t0 = time.perf_counter()
            try:
                new_queries, embed_indices = handler(query, accumulated_queries)
            except Exception:
                logger.exception(
                    "Transform '%s' failed -- skipping stage", transform_name
                )
                new_queries, embed_indices = [], []
            stage_ms = _elapsed_ms(stage_t0)

            # Record indices relative to the global accumulated list
            offset = len(accumulated_queries)
            for idx in embed_indices:
                use_embedding_of.add(offset + idx)

            accumulated_queries.extend(new_queries)
            applied.append(transform_name)
            stages.append(
                TransformStageResult(
                    transform=transform_name,
                    generated_queries=new_queries,
                    duration_ms=stage_ms,
                )
            )
            logger.debug(
                "Transform '%s' generated %d queries in %.1f ms",
                transform_name,
                len(new_queries),
                stage_ms,
            )

        total_ms = _elapsed_ms(t0)

        # if all fails include the original query
        if not accumulated_queries:
            logger.warning(
                "All transforms produced zero queries — falling back to original"
            )
            accumulated_queries = [query]

        result = TransformResult(
            original_query=query,
            expanded_queries=accumulated_queries,
            use_embedding_of=use_embedding_of,
            applied_transforms=applied,
            stages=stages,
            gate_skipped=False,
            total_duration_ms=total_ms,
        )

        logger.debug(
            "QueryTransform complete: input=%r | transforms=%s | "
            "expanded=%d queries | embed_of=%s | total=%.1f ms | "
            "stage_breakdown=%s",
            query[:80],
            applied,
            len(accumulated_queries),
            use_embedding_of or "none",
            total_ms,
            [
                {
                    "stage": s.transform,
                    "generated": len(s.generated_queries),
                    "ms": round(s.duration_ms, 1),
                }
                for s in stages
            ],
        )

        return result

    def _get_handler(self, name: str):
        return {
            "multi_query": self._handle_multi_query,
            "hyde": self._handle_hyde,
            "step_back": self._handle_step_back,
        }.get(name)

    def _handle_multi_query(
        self,
        original_query: str,
        _current: list[str],
    ) -> tuple[list[str], list[int]]:
        """
        Generate alternative phrasings of the question
        """

        prompt = _MULTI_QUERY_PROMPT.format(
            question=original_query, n=self._max_generated_queries
        )
        text = self._invoke_llm(prompt)
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return lines[: self._max_generated_queries], []

    def _handle_hyde(
        self,
        original_query: str,
        _current: list[str],
    ) -> tuple[list[str], list[int]]:
        """
        Generate a hypothetical document for embedding-based retrieval
        """

        prompt = _HYDE_PROMPT.format(question=original_query)
        text = self._invoke_llm(prompt)
        hypothetical = text.strip()
        if not hypothetical:
            return [], []

        new_queries = [hypothetical]
        embed_indices = [0]  # embed the hypothetical doc text

        return new_queries, embed_indices

    def _handle_step_back(
        self,
        original_query: str,
        _current: list[str],
    ) -> tuple[list[str], list[int]]:
        """
        Generate a broader step-back question
        """

        prompt = _STEP_BACK_PROMPT.format(question=original_query)
        text = self._invoke_llm(prompt)
        step_back = text.strip()
        if not step_back:
            return [], []
        return [step_back], []

    def _invoke_llm(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        content: object = response.content if hasattr(response, "content") else response
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1_000
