# Chat with an Assistant locally

See `llm.py` for a list of supported LLMs names.

## Installation

We recommend miniconda3 and python3.12.
- OpenAI, Together.ai, Anthropic APIs work on both mac and linux.
- Local Llama models currently work on the AIDA cluster. Resources needed for `Llama-3.2-11B-Vision-Instruct`: 1 A100 and >25GB GPU RAM.

```bash
conda create -n llm_chat python=3.12
conda activate llm_chat
pip install -r requirements-linux.txt
# or 
pip install -r requirements-mac.txt
```


## Run

1. **OpenAI GPTx, e.g. gpt-4o**. If you want to use OpenAI's GPT models, you'll need to set `$OPENAI_API_KEY`.
Then, on your laptop,
```bash
make RUN_FLAGS="--model_name gpt-4o-mini-2024-07-18" run
```

2. **TogetherAI models**. If you want to use LLMs hosted by TogetherAI, you'll need to set `$TOGETHER_API_KEY`.
Then, on your laptop,
```bash
make RUN_FLAGS="--model_name together:meta-llama/Llama-Vision-Free" run
```

3. **Anthropic models**. If you want to use Anthropic's Claude models, you'll need to set `$ANTHROPIC_API_KEY`.
Then, on your laptop,
```bash
make RUN_FLAGS="--model_name claude-3-5-sonnet-20241022" run
```

4. **Llama-3.2**.
First, request a node with an A100 (for the Llama 3.2 11B model) and note the compute node's ID, e.g. `c0021`.
Then, on the compute node,
```bash
# If you have already downloaded Llama 3.2 to a local directory
export LLAMA_MODEL_PATH=/mnt/beegfs/bulk/mirror/localllama/localLlama-3.2-11B-Vision-Instruct
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Vision-Instruct --model_local_path $LLAMA_MODEL_PATH" run

# If not downloaded, ask huggingface to download for you
huggingface-cli login
make RUN_FLAGS="--model_name meta-llama/Llama-3.2-11B-Vision-Instruct" run
```

Now on your local laptop, open a new terminal and run `./scripts/forward_streamlit_port_slurm_to_mac.sh c0021`.
Then open http://localhost:8501 in your browser.
This script forwards port 8501 from the compute node c0021 -> AIDA head node -> your laptop.
You might need to need to change the `aida` in the SSH cmd in that script to `user@aida.cac.cornell.edu`.

## TODOs

- Want to eventually load Llama 3.2 on my mac, but there's not much hardware/software support yet.
- Image uploads through the UI. Should be doable.
