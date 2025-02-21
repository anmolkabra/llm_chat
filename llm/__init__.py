from llm.anthropic import AnthropicChat
from llm.common import LLMChat
from llm.gemini import GeminiChat
from llm.hf import HFLlamaChat
from llm.ollama import OllamaChat
from llm.openai import OpenAIChat
from llm.together import TogetherChat
from llm.vllm import VLLMChat

SUPPORTED_LLM_SERVERS = [
    "anthropic",
    "gemini",
    "hf-llama",
    "ollama",
    "openai",
    "together",
    "vllm",
]


def get_llm(server: str, model_name: str, model_kwargs: dict) -> LLMChat:
    match server:
        case "anthropic":
            return AnthropicChat(model_name=model_name, **model_kwargs)
        case "gemini":
            return GeminiChat(model_name=model_name, **model_kwargs)
        case "hf-llama":
            # Huggingface Llama models do not support temperature=0, so we set it to 0.01 or higher
            if "temperature" in model_kwargs:
                model_kwargs["temperature"] = max(0.01, model_kwargs["temperature"])
            return HFLlamaChat(model_name=model_name, **model_kwargs)
        case "ollama":
            return OllamaChat(model_name=model_name, **model_kwargs)
        case "openai":
            return OpenAIChat(model_name=model_name, **model_kwargs)
        case "together":
            return TogetherChat(model_name=model_name, **model_kwargs)
        case "vllm":
            return VLLMChat(model_name=model_name, **model_kwargs)
        case _:
            raise ValueError(f"Provider {server} not supported.")
