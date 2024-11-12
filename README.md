# Chat with an Assistant locally

## Installation

We recommend miniconda3 and python3.12.
- OpenAI GPT models work on both mac and linux.
- Llama models currently work on the AIDA cluster. Resources needed for `Llama-3.2-11B-Vision-Instruct`: 1 A100 and >25GB GPU RAM.

```bash
conda create -n llm_chat python=3.12
conda activate llm_chat
pip install -r requirements-linux.txt
```

If you want to use OpenAI's GPT models, you'll need to set `$OPENAI_API_KEY`.
If you want to download from Huggingface, you'll need to set `$HUGGINGFACE_HUB_TOKEN`.

## Run

First, request a node with an A100 (for the Llama 3.2 11B model) and note the compute node's ID, e.g. `c0021`.
```bash
# If you have already downloaded Llama 3.2 to a local directory e.g. LLAMA_MODEL_PATH=/mnt/beegfs/bulk/mirror/localllama/localLlama-3.2-11B-Vision-Instruct
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Vision-Instruct --model_local_path $LLAMA_MODEL_PATH" run

# If not downloaded, ask huggingface to download for you
huggingface-cli login
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Vision-Instruct" run
```

Now open a new terminal and run `./scripts/forward_streamlit_port_slurm_to_mac.sh c0021`.
Then open http://localhost:8501 in your browser.
This script forwards port 8501 from the compute node c0021 -> AIDA head node -> your laptop.
You might need to need to change the `aida` in the SSH cmd in that script to `user@aida.cac.cornell.edu`.

## TODOs

- Want to eventually load Llama 3.2 on my mac, but there's not much hardware/software support yet.
- Image uploads through the UI. Should be doable.
