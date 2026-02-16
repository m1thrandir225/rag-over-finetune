# rag-over-finetune

## Requirements

- Python 3.13 (sentence-transformers currently is limited to 3.13)
- UV (recommended)

**Note**: Copy `.env.example` to an `.env` file if you plan to use any external LLM's, if not then please provide an URL to your local `Ollama` instance in the `config.json`.

You can ignore the field `llm_url` if you are using an external LLM like ChatGPT.

## Configuration (`config.json`)

The `config.json` file defines models, chunking behavior, vector store settings, and
prompting details. The current fields are:

- `embedding_model`: Model name for the selected embedding provider.
- `embedding_provider`: Embedding backend (`huggingface`, `openai`, `openrouter`). HuggingFace runs locally; OpenAI and OpenRouter use remote APIs.
- `llm_provider`: LLM backend (`ollama`, `openai`, `anthropic`, `google`, `openrouter`).
- `llm_model`: Model name for the selected provider.
- `llm_url`: Base URL for Ollama (ignored for external providers).
- `llm_temperature`: Sampling temperature.
- `llm_max_tokens`: Optional max tokens for output (`null` = provider default).
- `chunk_size`: Target chunk size used by text splitters.
- `chunk_overlap`: Overlap between consecutive chunks.
- `top_k`: Number of chunks retrieved for each query.
- `chroma_collection_name`: Chroma collection name.
- `chroma_persist_dir`: Path where Chroma persists data.
- `chunk_mode`: Chunking mode (`text`, `length`, `document`).
- `chunk_document_options`: Options for document-based chunking.
  - `json_mode_options`, `html_mode_options`, `markdown_mode_options`
- `chunk_length_options`: Options for length-based chunking.
  - `mode`: `char` or `token`
  - `char_mode_options`: `separator`, `is_separator_regex`
  - `token_mode_options`: `encoding_name`
- `system_prompt`: System instructions for the LLM.
- `prompt_template`: Prompt template with `{context}` and `{question}` placeholders.
- `embedding_device`: Device for HuggingFace embeddings only (e.g. `cpu`, `cuda`). Ignored for API providers. Use `auto` for automatic detection.
- `normalize_embeddings`: Whether to normalize embeddings.

### Notes

- `chunk_mode: document` is present but not implemented yet.
- Provider API keys are read from `.env`:
  - `OPENAI_API_KEY` (OpenAI LLM and embeddings)
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `OPENROUTER_API_KEY` (OpenRouter LLM and embeddings; use model IDs like `anthropic/claude-3-opus`, `openai/text-embedding-3-small`)

## Modes

Currently there are two supported modes:

- demo
- interactive

The `demo` mode runs a selected number of queries showcasing the capabilities of the system.

The `interactive` mode is run using the `--interactive` CLI argument and allows a user to ask questions and gives responses back to the user.

## Architecture

The system is organized around a single `RAG` entry point that wires together
configuration, ingestion, processing, storage, retrieval, and generation.

### Component layout

- `main.py`: CLI entry for demo and interactive modes. Loads config and instantiates `RAG`.
- `internal/config`: Loads `config.json` + `.env` and exposes a frozen `Config` object.
- `internal/importer`: Loads raw documents from files/directories (txt, pdf, docx).
- `internal/processing/chunker`: Splits text/documents into chunks (configurable mode).
- `internal/processing/embedding`: Builds embedding model (HuggingFace/OpenAI/OpenRouter) and embeds queries/documents.
- `internal/store/vector`: Manages Chroma vector store for persistence and retrieval.
- `internal/llm`: Builds provider-specific LLM clients (Ollama/OpenAI/Anthropic/Google/OpenRouter).
- `internal/rag`: Coordinates the end-to-end RAG flow and builds the LangChain chain.

### How components interact

1. `main.py` loads `Config` via `ConfigLoader` and creates `RAG(config)`.
2. Ingestion happens through `RAG.add_texts(...)`, `add_file(...)`, or `add_directory(...)`:
   - `DocumentImporter` loads files into `Document` objects.
   - `Chunker` splits documents into chunks.
   - `VectorStoreManager` embeds and stores chunks in Chroma.
3. Query flow uses `RAG.query(question)`:
   - `RAGChain` builds a LangChain pipeline with a retriever from `VectorStoreManager`.
   - The retriever pulls top-k relevant chunks for context.
   - `LLMService` provides the configured LLM to generate the final answer.
4. `RAG` caches the built chain and invalidates it when documents change.

--- 
**NOTE**: WORK IN PROGRESS
