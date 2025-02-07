import os

import openai

from llm.common import CommonLLMChat


class OpenAIChat(CommonLLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-11-20",
    ]

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
        max_retries: int = 3,
        wait_seconds: int = 2,
    ):
        super().__init__(model_name, model_path, max_tokens, temperature, seed, max_retries, wait_seconds)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_api(self, messages_api_format: list[dict]) -> str:
        # https://platform.openai.com/docs/api-reference/introduction
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_api_format,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            seed=self.seed,
        )
        return completion.choices[0].message.content
