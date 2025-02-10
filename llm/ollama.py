import ollama

from _types import ContentTextMessage, Conversation
from llm.common import CommonLLMChat, is_ollama_model


class OllamaChat(CommonLLMChat):
    # NOTE: We don't want to hardcode the supported ollama models here,
    # because this list is used at init time of streamlit UI. If ollama is not installed
    # on the system, the streamlit UI will crash. We still want other models to be available.
    # So we skip checking for supported models here.
    SUPPORTED_LLM_NAMES: list[str] = []

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        assert is_ollama_model(model_name), "model_name must start with 'ollama:'"
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
