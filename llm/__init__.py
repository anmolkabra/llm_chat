from llm.anthropic import AnthropicChat
from llm.common import LLMChat, is_ollama_model
from llm.gemini import GeminiChat
from llm.hf import HuggingfaceChat
from llm.ollama import OllamaChat
from llm.openai import OpenAIChat
from llm.together import TogetherChat

SUPPORTED_LLM_NAMES: list[str] = (
    AnthropicChat.SUPPORTED_LLM_NAMES
    + GeminiChat.SUPPORTED_LLM_NAMES
    + HuggingfaceChat.SUPPORTED_LLM_NAMES
    + OllamaChat.SUPPORTED_LLM_NAMES
    + OpenAIChat.SUPPORTED_LLM_NAMES
    + TogetherChat.SUPPORTED_LLM_NAMES
)


def get_llm(model_name: str, model_kwargs: dict) -> LLMChat:
    match model_name:
        case model_name if model_name in AnthropicChat.SUPPORTED_LLM_NAMES:
            return AnthropicChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in GeminiChat.SUPPORTED_LLM_NAMES:
            return GeminiChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in HuggingfaceChat.SUPPORTED_LLM_NAMES:
            # TODO only llama models?
            # Huggingface models do not support temperature=0, so we set it to 0.01 or higher
            if "temperature" in model_kwargs:
                model_kwargs["temperature"] = max(0.01, model_kwargs["temperature"])
            return HuggingfaceChat(model_name=model_name, **model_kwargs)
        case is_ollama_model(model_name):
            return OllamaChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in OpenAIChat.SUPPORTED_LLM_NAMES:
            return OpenAIChat(model_name=model_name, **model_kwargs)
        case model_name if model_name in TogetherChat.SUPPORTED_LLM_NAMES:
            return TogetherChat(model_name=model_name, **model_kwargs)
        case _:
            raise ValueError(f"Model name {model_name} must be one of {SUPPORTED_LLM_NAMES}.")
