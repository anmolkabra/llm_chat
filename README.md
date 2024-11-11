# Chat with an Assistant locally

## Installation

We recommend miniconda3 and python3.12.
Tested on an M3 Macbook Pro with 16GB RAM, and the AIDA cluster.

```bash
conda create -n llm_chat python=3.12
conda activate llm_chat
pip install -r requirements-linux.txt
```

If you want to use OpenAI's GPT models, you'll need to set `$OPENAI_API_KEY`.
If you want to download from Huggingface, you'll need to set `$HUGGINGFACE_HUB_TOKEN`.

## Run

```bash
# If you have already downloaded Llama 3.2 to a local directory e.g. LLAMA_MODEL_PATH=/mnt/beegfs/bulk/mirror/localllama/localLlama-3.2-11B-Vision-Instruct
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Instruct --model_local_path $LLAMA_MODEL_PATH" run

# If not downloaded, ask huggingface to download for you
huggingface-cli login
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Instruct" run

# or
./run_chat.sh
```

## TODOs

- UI reloads the model from memory on every call. Probably because streamlit repaints on each chat input. Should be fixable.
- Want to eventually load Llama 3.2 on my mac, but there's not much hardware/software support yet.
- Image uploads through the UI. Should be doable.
