# Chat with an Assistant locally

## Installation

We recommend miniconda3 and python3.12.
Tested on an M3 Macbook Pro with 16GB RAM.

```bash
conda create -n llm_chat python=3.12
conda activate llm_chat
pip install -r requirements.txt
```

You'll need to populate `$OPENAI_API_KEY` and `$HUGGINGFACE_HUB_TOKEN` env vars.

## Run
```bash
huggingface-cli login
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-3B-Instruct --stream_generations" run

# or
./run_chat.sh
```
