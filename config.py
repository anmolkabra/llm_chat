GPT_MODEL_NAMES: list[str] = [
    "gpt-4",
    "gpt-4o-mini",
    "gpt-4o",
]

LLAMA_MODEL_NAMES: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
]

TOGETHER_MODEL_NAMES: list[str] = [
    "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "together:meta-llama/Llama-Vision-Free", # Llama 3.2 11B Vision Instruct Turbo
]

supported_llm_model_names: list[str] = GPT_MODEL_NAMES + LLAMA_MODEL_NAMES + TOGETHER_MODEL_NAMES
