from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingManager:
    def __init__(
        self,
        model: str,
        model_kwargs: dict[str, Any],
        encode_kwargs: dict[str, Any],
        show_progress: bool,
    ) -> None:
        self.model = model
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.show_progress = show_progress

        # E5 models need query prefix
        if "e5" in model.lower():
            self.encode_kwargs["prompt"] = "query: "

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model=self.model,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
            show_progress=self.show_progress,
        )
