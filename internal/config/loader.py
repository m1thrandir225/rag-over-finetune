import json
from typing import Any

from .config import Config


class ConfigLoader:
    def __init__(self, path: str) -> None:
        self.path: str = path

    def load_config(self) -> Config:
        try:
            with open(self.path, "r") as config_file:
                config_data = json.load(config_file)

                return Config(
                    embedding_model=config_data["embedding_model"],
                    llm_model=config_data["llm_model"],
                )

        except FileNotFoundError:
            print(f"Error: The config file '{self.path}' was not found.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: The config file '{self.path}' had errors decoding.")
            exit()
