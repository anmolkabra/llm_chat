from llm.anthropic import AnthropicChat
from llm.common import LLMChat
from llm.gemini import GeminiChat
from llm.hf import HFLlamaChat
from llm.ollama import OllamaChat
from llm.openai import OpenAIChat
from llm.together import TogetherChat


def get_llm(model_name: str, model_kwargs: dict) -> LLMChat:
    # Check if the model name is supported by any of the LLMChat classes
    if AnthropicChat.is_model_supported(model_name):
        return AnthropicChat(model_name=model_name, **model_kwargs)
    elif GeminiChat.is_model_supported(model_name):
        return GeminiChat(model_name=model_name, **model_kwargs)
    elif HFLlamaChat.is_model_supported(model_name):
        # Huggingface Llama models do not support temperature=0, so we set it to 0.01 or higher
        if "temperature" in model_kwargs:
            model_kwargs["temperature"] = max(0.01, model_kwargs["temperature"])
        return HFLlamaChat(model_name=model_name, **model_kwargs)
    elif OllamaChat.is_model_supported(model_name):
        return OllamaChat(model_name=model_name, **model_kwargs)
    elif OpenAIChat.is_model_supported(model_name):
        return OpenAIChat(model_name=model_name, **model_kwargs)
    elif TogetherChat.is_model_supported(model_name):
        return TogetherChat(model_name=model_name, **model_kwargs)
    else:
        raise ValueError(f"Model name {model_name} not supported.")
