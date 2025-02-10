import ollama

from _types import ContentTextMessage, Conversation
from llm.common import CommonLLMChat


class OllamaChat(CommonLLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "ollama:deepseek-r1:8b",
        "ollama:deepseek-r1:1.5b",
        "ollama:llama3.2:1b",
    ]

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        assert model_name.startswith("ollama:"), "model_name must start with 'ollama:'"
        super().__init__(model_name, model_path, max_tokens, temperature, seed)
        self.ollama_headers: dict = {}
        self.client = ollama.Client(
            host="http://localhost:11434",
            headers=self.ollama_headers,
        )

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict]:
        """
        Converts conv into the following format and calls the Ollama client.
        ```
        [
            {"role": role1, "content": text1},
            {"role": role2, "content": text2},
            {"role": role3, "content": text3},
        ]
        ```
        """
        formatted_messages = []
        for message in conv.messages:
            for content in message.content:
                match content:
                    case ContentTextMessage(text=text):
                        formatted_messages.append({"role": message.role, "content": text})
                    # TODO figure out image parsing
        return formatted_messages

    def _call_api(self, messages_api_format: list[dict]) -> str:
        options = dict(
            temperature=self.temperature,
        )
        # Remove the "ollama:" prefix before setting up the client
        response = self.client.chat(
            model=self.model_name[len("ollama:") :],
            messages=messages_api_format,
            options=options,
        )
        return response.message.content
