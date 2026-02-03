# rag-over-finetune

## Requirements

- Python >= 3.14
- UV

**Note**: Copy `.env.example` to an `.env` file if you plan to use any external LLM's, if not then please provide an URL to your local `Ollama` instance in the `config.json`.

You can ignore the field `llm_url` if you are using an external LLM like ChatGPT.

## Modes

Currently there are two supported modes:
- demo
- interactive

The `demo` mode runs a selected number of queries showcasing the capabilities of the system.

The `interactive` mode is run using the `--interactive` CLI argument and allows a user to ask questions and gives responses back to the user.

**NOTE**: WORK IN PROGRESS
