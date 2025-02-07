import os

import together

from llm.common import CommonLLMChat


class TogetherChat(CommonLLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "together:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "together:meta-llama/Llama-Vision-Free",
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
        assert model_name.startswith("together:"), "model_name must start with 'together:'"
        super().__init__(model_name, model_path, max_tokens, temperature, seed, max_retries, wait_seconds)
        self.client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))

    def _call_api(self, messages_api_format: list[dict]) -> str:
        # https://github.com/togethercomputer/together-python
        # Remove the "together:" prefix before setting up the client
        completion = self.client.chat.completions.create(
            model=self.model_name[len("together:") :],
            messages=messages_api_format,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content
