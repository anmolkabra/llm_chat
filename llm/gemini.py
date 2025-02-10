import os

import google.generativeai as gemini

from _types import Conversation, ContentTextMessage
from llm.common import CommonLLMChat


class GeminiChat(CommonLLMChat):
    """
    Overrides the common messages format with the Gemini format:
    ```
    [
        {"role": role1, "parts": text1},
        {"role": role2, "parts": text2},
        {"role": role3, "parts": text3},
    ]
    ```
    """
    SUPPORTED_LLM_NAMES = [
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-2.0-flash-exp",
    ]

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(model_name, model_path, max_tokens, temperature, seed)

        gemini.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = gemini.GenerativeModel(self.model_name)

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict]:
        # https://ai.google.dev/gemini-api/docs/models/gemini
        # https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md
        formatted_messages = []
        for message in conv.messages:
            for content in message.content:
                match content:
                    case ContentTextMessage(text=text):
                        role = "model" if message.role == "assistant" else message.role
                        formatted_messages.append({"role": role, "parts": text})
        return formatted_messages

    def _call_api(self, messages_api_format: list[dict]) -> str:
        response = self.client.generate_content(
            contents=messages_api_format,
            generation_config=gemini.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text
