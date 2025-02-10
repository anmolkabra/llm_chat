import os

import anthropic

from llm.common import CommonLLMChat


class AnthropicChat(CommonLLMChat):
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        """
        Examples of model names:
            "claude-3-5-haiku-20241022"
            "claude-3-5-sonnet-20241022"
        """
        super().__init__(model_name, model_path, max_tokens, temperature, seed)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        return model_name.startswith("claude")

    def _call_api(self, messages_api_format: list[dict]) -> str:
        # https://docs.anthropic.com/en/api/migrating-from-text-completions-to-messages
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages_api_format,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text
