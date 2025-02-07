import abc

from tenacity import retry, stop_after_attempt, wait_fixed

import files
from _types import ContentImageMessage, ContentTextMessage, Conversation


class LLMChat(abc.ABC):
    SUPPORTED_LLM_NAMES: list[str] = []

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
        """
        Initialize the LLM chat object.

        Args:
            model_name (str): The model name to use.
            model_path (Optional[str]): Local path to the model.
                Defaults to None.
            max_tokens (int): Maximum number of tokens to generate.
                Defaults to 4096.
            temperature (Optional[float]): Temperature parameter for sampling.
                Defaults to 0.0.
            seed (Optional[int]): Seed for random number generator, passed to the model if applicable.
                Defaults to 0.
            max_retries (Optional[int]): Max number of API calls allowed before giving up.
                Defaults to 3.
            wait_seconds (Optional[int]): Number of seconds to wait between API calls.
                Defaults to 2.
        """
        assert (
            model_name in self.SUPPORTED_LLM_NAMES
        ), f"Model name {model_name} must be one of {self.SUPPORTED_LLM_NAMES}."
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds

        self.model_kwargs = dict(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries,
            wait_seconds=wait_seconds,
        )

    @abc.abstractmethod
    def generate_response(self, conv: Conversation) -> str:
        """
        Generate a response for the conversation.

        Args:
            conv (Conversation): The conversation object.

        Returns:
            str: The response generated by the model.
        """
        pass


class CommonLLMChat(LLMChat):
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
        self.client = None

    @abc.abstractmethod
    def _call_api(self, messages_api_format: list[dict]) -> str:
        """
        Expects messages ready for API. Use `convert_conv_to_api_format` to convert a conversation
        into such a list.
        """
        pass

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict]:
        """
        Converts the conversation object to a common format supported by various LLM providers.
        Common format is:
        ```
        [
            {"role": role1, "content": [{"type": "text", "text": text1},
            {"role": role2, "content": [{"type": "text", "text": text2},
            {"role": role3, "content": [{"type": "text", "text": text3},
        ]
        ```
        """
        formatted_messages = []
        for message in conv.messages:
            for content in message.content:
                match content:
                    case ContentTextMessage(text=text):
                        formatted_messages.append({"role": message.role, "content": [{"type": "text", "text": text}]})
                    case ContentImageMessage(image=image):
                        base64_image = files.pil_to_base64(image)
                        formatted_messages.append(
                            {
                                "role": message.role,
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    }
                                ],
                            }
                        )
        return formatted_messages

    def generate_response(self, conv: Conversation) -> str:
        """
        Generate response for the conversation.
        """
        assert self.client is not None, "Client is not initialized."

        # max_retries and wait_seconds are object attributes, and cannot be written around the generate_response function
        # So we need to wrap the _call_api function with the retry decorator
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(self.wait_seconds))
        def _call_api_wrapper(conv: Conversation) -> str:
            messages_api_format: list[dict] = self._convert_conv_to_api_format(conv)
            return self._call_api(messages_api_format)

        return _call_api_wrapper(conv)
