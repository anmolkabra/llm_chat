import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from _types import ContentImageMessage, Conversation
from llm.common import LLMChat


class HFLlamaChat(LLMChat):
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
            "meta-llama/Llama-3.1-8B-Instruct"
            "meta-llama/Llama-3.2-3B-Instruct"
            "meta-llama/Llama-3.2-11B-Vision-Instruct"
        """
        super().__init__(model_name, model_path, max_tokens, temperature, seed)

        # Use local model if provided
        model_path_to_use = self.model_path or self.model_name
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path_to_use,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path_to_use)

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
            **inputs, temperature=self.temperature, max_new_tokens=self.max_tokens
        )  # shape (1, output_length)
        decoded_output = self.processor.decode(outputs[0])
        user_assistant_alternate_messages: list[str] = decoded_output.split("assistant<|end_header_id|>")
        latest_assistant_message: str = (
            user_assistant_alternate_messages[-1].strip().rstrip("<|eot_id|>")
            if user_assistant_alternate_messages
            else ""
        )
        return latest_assistant_message
