import json
from typing import Any

from langchain_core.embeddings import Embeddings
from openrouter import OpenRouter
from openrouter.errors import (
    OpenRouterDefaultError,
    PaymentRequiredResponseError,
    ResponseValidationError,
)


class OpenRouterSDKEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings adapter backed by OpenRouter's official SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._batch_size = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        with OpenRouter(api_key=self._api_key) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                try:
                    response = client.embeddings.generate(
                        model=self._model,
                        input=batch,
                    )
                except PaymentRequiredResponseError as exc:
                    raise ValueError(self._format_payment_error(exc)) from exc
                except ResponseValidationError as exc:
                    # OpenRouter can return an error payload that fails SDK success-schema parsing.
                    if self._is_payment_limit_error(exc):
                        raise ValueError(self._format_payment_error(exc)) from exc
                    raise ValueError(
                        f"OpenRouter embedding response validation failed: {exc}"
                    ) from exc
                except OpenRouterDefaultError as exc:
                    raise ValueError(
                        f"OpenRouter embedding request failed: {exc}"
                    ) from exc
                all_embeddings.extend(self._extract_embeddings(response))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ValueError("OpenRouter returned no embedding for query input.")
        return vectors[0]

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not data:
            raise ValueError("No embedding data received from OpenRouter SDK.")

        embeddings: list[list[float]] = []
        for item in data:
            embedding = getattr(item, "embedding", None)
            if embedding is None and isinstance(item, dict):
                embedding = item.get("embedding")
            if embedding is None:
                raise ValueError("Malformed embedding item returned by OpenRouter SDK.")
            embeddings.append(embedding)
        return embeddings

    @staticmethod
    def _is_payment_limit_error(exc: ResponseValidationError) -> bool:
        body = getattr(exc, "body", "") or ""
        message = str(exc)
        return "Payment required" in message or "Payment required" in body

    @staticmethod
    def _format_payment_error(exc: Exception) -> str:
        body = getattr(exc, "body", "") or ""
        raw_response = getattr(exc, "raw_response", None)
        status = getattr(raw_response, "status_code", None)

        api_message = ""
        if body:
            try:
                payload = json.loads(body)
                api_message = payload.get("error", {}).get("message", "")
            except json.JSONDecodeError:
                api_message = ""

        details = api_message or "Payment required or monthly spend limit reached."
        status_text = f" (status {status})" if status is not None else ""
        return (
            f"OpenRouter embeddings request failed{status_text}: {details} "
            "Please check your OpenRouter credits/spending limits."
        )
