import openai

from llm.openai import OpenAIChat


class VLLMChat(OpenAIChat):
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(model_name, model_path, max_tokens, temperature, seed)
        self.client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
