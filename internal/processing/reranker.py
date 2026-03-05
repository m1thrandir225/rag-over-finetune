from __future__ import annotations

import logging
import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder re-ranker that rescores (query, document) pairs
    and returns the top-N results sorted by relevance.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(
                "Loading cross-encoder model %r on device=%s",
                self._model_name,
                self._device,
            )
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        docs_with_scores: list[tuple[Document, float]],
        top_n: int,
    ) -> list[tuple[Document, float]]:
        """
        Re-rank *docs_with_scores* using a cross-encoder and return
        the top *top_n* results with their new cross-encoder scores.

        Parameters
        ----------
        query:
            The original user question.
        docs_with_scores:
            Candidate documents with their pre-rerank scores.
            Only the documents are sent to the cross-encoder; the old scores are discarded.
        top_n:
            How many documents to keep after re-ranking.

        Returns
        -------
        list[tuple[Document, float]]
            Re-ranked documents with cross-encoder scores,
            sorted descending by score and capped at *top_n*.
        """
        if not docs_with_scores:
            return []

        model = self._get_model()

        pairs = [(query, doc.page_content) for doc, _score in docs_with_scores]

        t0 = time.perf_counter()
        ce_scores: list[float] = model.predict(pairs).tolist()
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        scored = list(zip(docs_with_scores, ce_scores))
        scored.sort(key=lambda t: t[1], reverse=True)

        result = [
            (doc, float(ce_score)) for (doc, _old_score), ce_score in scored[:top_n]
        ]

        logger.debug(
            "Rerank: %d candidates -> top %d in %.1f ms "
            "(model=%s, score range=%.4f..%.4f)",
            len(docs_with_scores),
            len(result),
            elapsed_ms,
            self._model_name,
            min(ce_scores),
            max(ce_scores),
        )

        return result
