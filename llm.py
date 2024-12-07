import abc
import os

import anthropic
import openai
import together
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoProcessor, MllamaForConditionalGeneration

import files
from data import ContentImageMessage, ContentTextMessage, Conversation


class LLMChat(abc.ABC):
    SUPPORTED_LLM_NAMES: list[str] = []

    def __init__(self, model_name: str, max_retries: int, wait_seconds: int, temperature: float, seed: int):
        """
        Initialize the LLM chat object.

        Args:
            model_name (str): The model name to use.
            max_retries (int): Max number of API calls allowed before giving up.
            wait_seconds (int): Number of seconds to wait between API calls.
            temperature (float): Temperature parameter for sampling.
            seed (int): Seed for random number generator, passed to the model if applicable.
        """
        assert model_name in self.SUPPORTED_LLM_NAMES, f"Model name {model_name} must be one of {self.SUPPORTED_LLM_NAMES}."
        self.model_name = model_name
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds
        self.temperature = temperature
        self.seed = seed

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
        max_retries: int,
        wait_seconds: int,
        temperature: float,
        seed: int,
    ):
        super().__init__(model_name, max_retries, wait_seconds, temperature, seed)
        self.client = None
    
    @abc.abstractmethod
    def _call_api(self, conv: Conversation) -> str:
        pass

    def convert_conv_to_common_format(self, conv: Conversation) -> list[dict]:
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
            return self._call_api(conv)

        return _call_api_wrapper(conv)


class OpenAIChat(CommonLLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-11-20",
    ]

    def __init__(
        self,
        model_name: str,
        max_retries: int,
        wait_seconds: int,
        temperature: float,
        seed: int,
    ):
        super().__init__(model_name, max_retries, wait_seconds, temperature, seed)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _call_api(self, conv: Conversation) -> str:
        # https://platform.openai.com/docs/api-reference/introduction
        formatted_messages = self.convert_conv_to_common_format(conv)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            seed=self.seed,
        )
        return completion.choices[0].message.content


class TogetherChat(OpenAIChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "together:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "together:meta-llama/Llama-Vision-Free",
    ]

    def __init__(
        self,
        model_name: str,
        max_retries: int,
        wait_seconds: int,
        temperature: float,
        seed: int,
    ):
        assert model_name.startswith("together:"), "model_name must start with 'together:'"
        super().__init__(model_name, max_retries, wait_seconds, temperature, seed)
        self.client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))

    def _call_api(self, conv: Conversation) -> str:
        # https://github.com/togethercomputer/together-python
        formatted_messages = self.convert_conv_to_common_format(conv)

        # Remove the "together:" prefix before setting up the client
        completion = self.client.chat.completions.create(
            model=self.model_name.lstrip("together:"),
            messages=formatted_messages,
            temperature=self.temperature,
            seed=self.seed,
        )
        return completion.choices[0].message.content


class AnthropicChat(CommonLLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
    ]

    def __init__(
        self,
        model_name: str,
        max_retries: int,
        wait_seconds: int,
        temperature: float,
        seed: int,
    ):
        super().__init__(model_name, max_retries, wait_seconds, temperature, seed)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def _call_api(self, conv: Conversation) -> str:
        # https://docs.anthropic.com/en/api/migrating-from-text-completions-to-messages
        formatted_messages = self.convert_conv_to_common_format(conv)
        response = self.client.messages.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
        )
        return response.content[0].text


class LocalLlamaChat(LLMChat):
    SUPPORTED_LLM_NAMES: list[str] = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ]

    def __init__(self, model_path: str, max_retries: int, wait_seconds: int, temperature: float, seed: int):
        """
        Args:
            model_path (str): Huggingface hub model path or local model path.
        """
        super().__init__(model_path, max_retries, wait_seconds, temperature, seed)

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_response(self, conv: Conversation) -> str:
        # Take out images from messages
        images: list[Image.Image] = []
        for message in conv.messages:
            for content in message.content:
                match content:
                    case ContentImageMessage(image=image):
                        images.append(image)

        # Process text and images
        input_text = self.processor.apply_chat_template(conv.messages, add_generation_prompt=True)
        inputs = self.processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate and decode
        outputs = self.model.generate(
            **inputs, temperature=self.temperature, max_new_tokens=1024
        )  # shape (1, output_length)
        decoded_output = self.processor.decode(outputs[0])
        user_assistant_alternate_messages: list[str] = decoded_output.split("assistant<|end_header_id|>")
        latest_assistant_message: str = (
            user_assistant_alternate_messages[-1].strip().rstrip("<|eot_id|>")
            if user_assistant_alternate_messages
            else ""
        )
        return latest_assistant_message


SUPPORTED_LLM_NAMES: list[str] = OpenAIChat.SUPPORTED_LLM_NAMES + TogetherChat.SUPPORTED_LLM_NAMES + AnthropicChat.SUPPORTED_LLM_NAMES + LlamaChat.SUPPORTED_LLM_NAMES + 

def get_llm(model_name: str, model_kwargs: dict) -> LLMChat:
    match model_name:
        case model_name if model_name in OpenAIChat.SUPPORTED_LLM_NAMES:
            return OpenAIChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in TogetherChat.SUPPORTED_LLM_NAMES:
            return TogetherChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in AnthropicChat.SUPPORTED_LLM_NAMES:
            return AnthropicChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in LocalLlamaChat.SUPPORTED_LLM_NAMES:
            # LocalLlama models do not support temperature=0, so we set it to 0.01 or higher
            if "temperature" in model_kwargs:
                model_kwargs["temperature"] = max(0.01, model_kwargs["temperature"])
            return LocalLlamaChat(model_name=model_name, **model_kwargs)
        case _:
            raise ValueError(f"Model name {model_name} must be one of {SUPPORTED_LLM_NAMES}.")
