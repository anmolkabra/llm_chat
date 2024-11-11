import abc
import os

import openai
import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoProcessor, MllamaForConditionalGeneration

from data import Conversation


class LLMChat(abc.ABC):
    def __init__(self, max_retries: int, wait_seconds: int, temperature: float, seed: int):
        self.max_retries = max_retries
        self.wait_seconds = wait_seconds
        self.temperature = temperature
        self.seed = seed

    @abc.abstractmethod
    def generate_response(self, conv: Conversation) -> str:
        pass


class OpenAIChat(LLMChat):
    def __init__(
        self,
        model_name: str,
        max_retries: int,
        wait_seconds: int,
        temperature: float,
        seed: int,
        stream_generations: bool,
    ):
        super().__init__(max_retries, wait_seconds, temperature, seed)
        self.model_name = model_name
        self.stream_generations = stream_generations
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, conv: Conversation) -> str:
        # Wrap retry params inside generate_response
        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(self.wait_seconds))
        def _call_api(conv: Conversation) -> str:
            completion = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=conv.messages,
                temperature=self.temperature,
                seed=self.seed,
                stream=self.stream_generations,
            )
            return completion if self.stream_generations else completion.choices[0].message

        return _call_api(conv)


class LlamaChat(LLMChat):
    def __init__(self, model_path: str, max_retries: int, wait_seconds: int, temperature: float, seed: int):
        super().__init__(max_retries, wait_seconds, temperature, seed)
        self.model_path = model_path

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_response(self, conv: Conversation) -> str:
        # Take out images from messages
        # TODO images
        images: list[Image.Image] = []
        # for message in conv.messages:
        #     for content in message.content:
        #         match content:
        #             case {"type": "image", "image": image}:
        #                 images.append(image)

        # Process text and images
        input_text = self.processor.apply_chat_template(conv.messages, add_generation_prompt=True)
        inputs = self.processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate and decode
        outputs = self.model.generate(**inputs, temperature=self.temperature)
        decoded_output = self.processor.decode(outputs[0])
        split_output = decoded_output.split("assistant<|end_header_id|>")
        return split_output[-1].strip() if split_output else ""
