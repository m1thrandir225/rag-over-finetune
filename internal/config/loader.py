import json
from pathlib import Path

from dotenv import load_dotenv

from internal.constants import (
    DEFAULT_ENABLED_QUERY_TRANSFORMS,
    DEFAULT_ENV_PATH,
    DEFAULT_HYDE_INCLUDE_ORIGINAL_QUERY,
    DEFAULT_MAX_GENERATED_QUERIES,
    DEFAULT_OLLAMA_URL,
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_QDRANT_URL,
    DEFAULT_QUERY_TRANSFORM_TIMEOUT_MS,
    DEFAULT_RETRIEVAL_K_PER_QUERY,
    DEFAULT_RETRIEVAL_K_TOTAL_BEFORE_RERANK,
    DEFAULT_RETRIEVAL_MERGE_STRATEGY,
    DEFAULT_TRANSFORM_GATE_ENABLED,
)

from .config import (
    ChunkDocumentOptions,
    ChunkLengthOptions,
    Config,
    EmbeddingProvider,
    LLMProvider,
)
from .device import _detect_best_device

_system_prompt_path = Path("prompts/system_prompt.txt")
_template_path = Path("prompts/prompt_template.txt")

_system_prompt_fallback = (
    "Ти си помошник кој одговара на прашања.\n"
    "Користи го САМО дадениот контекст за да одговориш на прашањето.\n"
    "Ако одговорот не е во контекстот, СЕКОГАШ кажи дека немаш доволно информации на темата.\n"
    "Одговарај точно и концизно."
)
_template_fallback = "Контекст:\n{context}\n\nПрашање: {question}\n\nОдговор:"


def _load_prompt_file(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback


class ConfigLoader:
    """
    Config Loader is resonsible for loading the config.json file and creating a frozen Config object
    """

    def __init__(self, path: str, env_path: str = DEFAULT_ENV_PATH) -> None:
        self.path: str = path
        self.env_path: str = env_path
        load_dotenv(self.env_path)

    def _parse_provider(self, provider_str: str) -> LLMProvider:
        """
        Parse provider string to LLMProvider enum
        """
        provider_map = {
            "ollama": LLMProvider.OLLAMA,
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "google": LLMProvider.GOOGLE,
            "openrouter": LLMProvider.OPENROUTER,
        }

        provider = provider_map.get(provider_str.lower())
        if provider is None:
            raise ValueError(
                f"Unknown LLM provider: {provider_str}. Supported: {list(provider_map.keys())}"
            )
        return provider

    def _parse_embedding_provider(self, provider_str: str) -> EmbeddingProvider:
        """
        Parse embedding provider string to EmbeddingProvider enum
        """
        provider_map = {
            "huggingface": EmbeddingProvider.HUGGINGFACE,
            "openai": EmbeddingProvider.OPENAI,
            "openrouter": EmbeddingProvider.OPENROUTER,
        }
        provider = provider_map.get(provider_str.lower())
        if provider is None:
            raise ValueError(
                f"Unknown embedding provider: {provider_str}. "
                f"Supported: {list(provider_map.keys())}"
            )
        return provider

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        """
        Resolve the embedding device.  'auto' triggers detection.
        """
        if device_str.lower() == "auto":
            return _detect_best_device()
        return device_str

    def load_config(self) -> Config:
        try:
            with open(self.path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)

                chunk_doc_options = None
                if "chunk_document_options" in config_data:
                    doc_opts = config_data["chunk_document_options"]
                    chunk_doc_options = ChunkDocumentOptions(
                        json_mode_options=doc_opts.get("json_mode_options", {}),
                        html_mode_options=doc_opts.get("html_mode_options", {}),
                        markdown_mode_options=doc_opts.get("markdown_mode_options", {}),
                    )

                chunk_len_options = None
                if "chunk_length_options" in config_data:
                    len_opts = config_data["chunk_length_options"]
                    chunk_len_options = ChunkLengthOptions(
                        mode=len_opts.get("mode", "char"),
                        char_mode_options=len_opts.get("char_mode_options", {}),
                        token_mode_options=len_opts.get("token_mode_options", {}),
                    )

                provider_str = config_data.get("llm_provider", "ollama")
                llm_provider = self._parse_provider(provider_str)

                embedding_provider_str = config_data.get(
                    "embedding_provider", "huggingface"
                )
                embedding_provider = self._parse_embedding_provider(
                    embedding_provider_str
                )

                max_tokens_raw = config_data.get("llm_max_tokens")

                llm_max_tokens = (
                    int(max_tokens_raw) if max_tokens_raw is not None else None
                )

                default_system_prompt = _load_prompt_file(
                    _system_prompt_path, _system_prompt_fallback
                )
                default_prompt_template = _load_prompt_file(
                    _template_path, _template_fallback
                )

                return Config(
                    embedding_model=config_data["embedding_model"],
                    embedding_provider=embedding_provider,
                    llm_model=config_data["llm_model"],
                    chunk_size=config_data.get("chunk_size", 512),
                    chunk_overlap=config_data.get("chunk_overlap", 50),
                    top_k=config_data.get("top_k", 3),
                    vector_store_provider=config_data.get(
                        "vector_store_provider", "qdrant"
                    ),
                    qdrant_collection_name=config_data.get(
                        "qdrant_collection_name", DEFAULT_QDRANT_COLLECTION_NAME
                    ),
                    qdrant_url=config_data.get("qdrant_url", DEFAULT_QDRANT_URL),
                    qdrant_api_key=config_data.get("qdrant_api_key"),
                    qdrant_prefer_grpc=config_data.get("qdrant_prefer_grpc", False),
                    chunk_mode=config_data.get("chunk_mode", "text"),
                    chunk_document_options=chunk_doc_options,
                    chunk_length_options=chunk_len_options,
                    llm_provider=llm_provider,
                    llm_url=config_data.get("llm_url", DEFAULT_OLLAMA_URL),
                    llm_temperature=config_data.get("llm_temperature", 0.75),
                    llm_max_tokens=llm_max_tokens,
                    system_prompt=default_system_prompt,  # TODO: check default system prompt
                    prompt_template=default_prompt_template,  # TODO: check default prompt template
                    embedding_device=self._resolve_device(
                        config_data.get("embedding_device", "auto")
                    ),
                    normalize_embeddings=config_data.get("normalize_embeddings", True),
                    embedding_batch_size=config_data.get("embedding_batch_size", 256),
                    semantic_breakpoint_type=config_data.get(
                        "semantic_breakpoint_type", "percentile"
                    ),
                    semantic_breakpoint_amount=config_data.get(
                        "semantic_breakpoint_amount"
                    ),
                    # MCP Tools Configuration
                    mcp_enabled=config_data.get("mcp_enabled", False),
                    mcp_servers=config_data.get("mcp_servers", {}),
                    query_transform_mode=config_data.get(
                        "query_transform_mode", "multi_query"
                    ),
                    enabled_transforms=config_data.get(
                        "enabled_transforms", DEFAULT_ENABLED_QUERY_TRANSFORMS
                    ),
                    max_generated_queries=config_data.get(
                        "max_generated_queries", DEFAULT_MAX_GENERATED_QUERIES
                    ),
                    transform_gate_enabled=config_data.get(
                        "transform_gate_enabled", DEFAULT_TRANSFORM_GATE_ENABLED
                    ),
                    k_per_query=config_data.get(
                        "k_per_query", DEFAULT_RETRIEVAL_K_PER_QUERY
                    ),
                    k_total_before_rerank=config_data.get(
                        "k_total_before_rerank", DEFAULT_RETRIEVAL_K_TOTAL_BEFORE_RERANK
                    ),
                    merge_strategy=config_data.get(
                        "merge_strategy", DEFAULT_RETRIEVAL_MERGE_STRATEGY
                    ),
                    hyde_include_original_query=config_data.get(
                        "hyde_include_original_query",
                        DEFAULT_HYDE_INCLUDE_ORIGINAL_QUERY,
                    ),
                    query_transform_timeout_ms=config_data.get(
                        "query_transform_timeout_ms",
                        DEFAULT_QUERY_TRANSFORM_TIMEOUT_MS,
                    ),
                )
        except FileNotFoundError:
            print(f"Error: The config file '{self.path}' was not found.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: The config file '{self.path}' had errors decoding.")
            exit()
        except KeyError as e:
            print(f"Error: Missing required config key: {e}")
            exit()
