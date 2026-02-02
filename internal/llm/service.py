import os
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ..config import Config, LLMProvider

# Type alias for supported LLM types
LLMType = Union[ChatOllama, ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]


class LLMService:
    """
    Manages LLM initialization and direct invocation.
    Supports multiple providers: Ollama, OpenAI, Anthropic (Claude), and Google (Gemini).

    Provider is configured via config.json (llm_provider field).
    API keys are loaded from environment variables (.env file):
        - OPENAI_API_KEY for OpenAI
        - ANTHROPIC_API_KEY for Anthropic/Claude
        - GOOGLE_API_KEY for Google/Gemini
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm: LLMType | None = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def llm(self) -> LLMType:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment variable"""
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"Missing API key: {env_var} environment variable is not set. "
                f"Please add it to your .env file."
            )
        return api_key

    def _create_llm(self) -> LLMType:
        """Create the appropriate LLM based on the configured provider"""
        provider = self.config.llm_provider

        if provider == LLMProvider.OLLAMA:
            return self._create_ollama()
        elif provider == LLMProvider.OPENAI:
            return self._create_openai()
        elif provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic()
        elif provider == LLMProvider.GOOGLE:
            return self._create_google()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_ollama(self) -> ChatOllama:
        """
        Create Ollama Chat LLM instance
        """
        return ChatOllama(
            model=self.config.llm_model,
            base_url=self.config.llm_url,
            temperature=self.config.llm_temperature,
        )

    def _create_openai(self) -> ChatOpenAI:
        """
        Create OpenAI LLM instance
        """
        api_key = self._get_api_key("OPENAI_API_KEY")
        kwargs = {
            "model": self.config.llm_model,
            "api_key": api_key,
            "temperature": self.config.llm_temperature,
        }
        if self.config.llm_max_tokens is not None:
            kwargs["max_tokens"] = self.config.llm_max_tokens
        return ChatOpenAI(**kwargs)  # ty: ignore

    def _create_anthropic(self) -> ChatAnthropic:
        """
        Create  Claude LLM instance
        """
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        kwargs = {
            "model": self.config.llm_model,
            "api_key": api_key,
            "temperature": self.config.llm_temperature,
        }
        if self.config.llm_max_tokens is not None:
            kwargs["max_tokens"] = self.config.llm_max_tokens
        return ChatAnthropic(**kwargs)  # ty: ignore

    def _create_google(self) -> ChatGoogleGenerativeAI:
        """
        Create Gemini LLM instance
        """
        api_key = self._get_api_key("GOOGLE_API_KEY")
        kwargs = {
            "model": self.config.llm_model,
            "google_api_key": api_key,
            "temperature": self.config.llm_temperature,
        }
        if self.config.llm_max_tokens is not None:
            kwargs["max_output_tokens"] = self.config.llm_max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    def invoke(self, prompt: str) -> str:
        """
        Invoke the configured LLM with a prompt
        """
        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content  # pyright: ignore
        return response  # pyright: ignore

    def generate_with_context(self, question: str, context: str) -> str:
        """
        Generate a response given a question and a context
        """
        full_template = f"{self.config.system_prompt}\n\n{self.config.prompt_template}"
        prompt = full_template.format(
            context=context,
            question=question,
        )
        return self.invoke(prompt)
