import json

from .config import Config, ChunkDocumentOptions, ChunkLengthOptions


class ConfigLoader:
    """
    Config Loader is resonsible for loading the config.json file and creating a frozen Config object
    """
    def __init__(self, path: str) -> None:
        self.path: str = path

    def load_config(self) -> Config:
        try:
            with open(self.path, "r") as config_file:
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

                return Config(
                    embedding_model=config_data["embedding_model"],
                    llm_model=config_data["llm_model"],
                    chunk_size=config_data.get("chunk_size", 512),
                    chunk_overlap=config_data.get("chunk_overlap", 50),
                    top_k=config_data.get("top_k", 3),
                    chroma_collection_name=config_data.get("chroma_collection_name", "vector_collection"),
                    chroma_persist_dir=config_data.get("chroma_persist_dir", "./chroma_db"),
                    chunk_mode=config_data.get("chunk_mode", "text"),
                    chunk_document_options=chunk_doc_options,
                    chunk_length_options=chunk_len_options,
                    llm_url=config_data.get("llm_url", "http://localhost:11434"),
                    llm_temperature=config_data.get("llm_temperature", 0.75),
                    system_prompt=config_data.get("system_prompt",""), #TODO: add default system prompt
                    prompt_template=config_data.get("prompt_template",""), #TODO: add default prompt template
                    embedding_device=config_data.get("embedding_device", "cpu"), #TODO: add embedding_device detection
                    normalize_embeddings=config_data.get("normalize_embeddings", True),
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