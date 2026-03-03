from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# TODO: Translate to Macedonian
_MULTI_QUERY_PROMPT = """You are an expert at rephrasing questions for information retrieval.
Given the following question, generate 2-3 alternative phrasings that a user might use to search for the same information.
Write each alternative on a new line. Do not include numbering or bullets.
Respond only with the alternative questions, one per line.

Original question: {question}"""

# TODO: Translate to Macedonian
_HYDE_PROMPT = """Based on the following question, write a short, factual paragraph that could appear in a document answering this question.
Write as if you are excerpting from a relevant Wikipedia article or textbook. Use the same language as the question.
Do not say you don't know; write a plausible answer.

Question: {question}

Hypothetical answer:"""

_STEP_BACK_PROMPT = """You are an expert at breaking down complex questions into simpler, more general ones.
Given the following question, write a single more general "step-back" question that would help gather broader context.
The step-back question should be conceptual and abstract, not specific to the original question's details.

Original question: {question}

Step-back question:"""


class QueryTransformMode(str, Enum):
    NONE = "none"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"


@dataclass
class QueryTransformResult:
    """
    Result of query transformation.
    - queries: text(s) to use for retrieval
    - use_embedding_of_first: if True, embed queries[0] and search by vector (HyDE)
    """

    queries: list[str]
    use_embedding_of_first: bool = False


class QueryTransformer:
    """
    Transforms user queries before retrieval to improve recall.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: QueryTransformMode = QueryTransformMode.MULTI_QUERY,
    ) -> None:
        self._llm = llm
        self._mode = mode

    def transform(self, query: str) -> QueryTransformResult:
        """
        Transform the query into one or more retrieval inputs.
        """
        if not query.strip():
            return QueryTransformResult(queries=[query])

        match self._mode:
            case QueryTransformMode.NONE:
                return QueryTransformResult(queries=[query])
            case QueryTransformMode.MULTI_QUERY:
                return self._multi_query(query)
            case QueryTransformMode.HYDE:
                return self._hyde(query)
            case QueryTransformMode.STEP_BACK:
                return self._step_back(query)
            case _:
                return QueryTransformResult(queries=[query])

    def _multi_query(self, query: str) -> QueryTransformResult:
        prompt = _MULTI_QUERY_PROMPT.format(question=query)
        response = self._llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        queries = [query] + lines[:3]
        return QueryTransformResult(queries=queries)

    def _hyde(self, query: str) -> QueryTransformResult:
        prompt = _HYDE_PROMPT.format(question=query)
        response = self._llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        hypothetical = text.strip() or query
        return QueryTransformResult(
            queries=[hypothetical],
            use_embedding_of_first=True,
        )

    def _step_back(self, query: str) -> QueryTransformResult:
        prompt = _STEP_BACK_PROMPT.format(question=query)
        response = self._llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        step_back = text.strip() or query
        return QueryTransformResult(queries=[step_back, query])
