from langchain_community.llms.ollama import Ollama

from ..config import Config

class LLMService:
    """
    Manages LLM initialization and direct invocation of it
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm: Ollama | None = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def llm(self) -> Ollama:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> Ollama:
        return Ollama(
            model=self.config.llm_model,
            base_url=self.config.llm_url,
            temperature=self.config.llm_temperature,
        )

    def invoke(self, prompt: str) -> str:
        """
        Invoke the configured LLM with a prompt
        """
        return self.llm.invoke(prompt)

    def generate_with_context(self, question: str, context: str) -> str:
        """
        Generate a response given a question and a context
        """
        prompt = self.config.prompt_template.format(
            context=context,
            question=question,
        )
        return self.invoke(prompt)