# Chat with an Assistant locally

See `llm.py` for a list of supported LLMs names.

## Installation

We recommend miniconda3 and python3.12.
- OpenAI, Together.ai, Anthropic APIs work on both mac and linux.
- Ollama models are also supported, and you can add your own models. See Ollama models section.
- Local Llama models currently work on the AIDA cluster. Resources needed for `Llama-3.2-11B-Vision-Instruct`: 1 A100 and >25GB GPU RAM.

```bash
conda create -n llm_chat python=3.12
conda activate llm_chat
pip install -r requirements-linux.txt
# or 
pip install -r requirements-mac.txt
```


## Run

It is super easy to add your own LLM provider by making simple changes to `llm.py`.
Currently the code supports the following LLM providers out-of-the-box:
- OpenAI
- Together.ai
- Anthropic
- Ollama
- LocalLlama through Huggingface

See the list of models supported for these LLM providers in `llm.py` in the `<LLM_PROVIDER>.SUPPORTED_LLM_NAMES` list.
You can add your to the list and the UI will recognize that as an option.

### OpenAI models
You'll need to set `$OPENAI_API_KEY` environment variable.
Then, on your machine,
```bash
./run_chat.sh --model_name gpt-4o-mini-2024-07-18
```

### Together.ai models
You'll need to set `$TOGETHER_API_KEY` environment variable.
Then, on your machine,
```bash
./run_chat.sh --model_name together:meta-llama/Llama-Vision-Free
```

### Anthropic models
You'll need to set `$ANTHROPIC_API_KEY` environment variable.
Then, on your machine,
```bash
./run_chat.sh --model_name claude-3-5-sonnet-20241022
```

### Ollama models
[Ollama](https://ollama.com/) provides a way to locally run LLMs on your machine (mac, linux, windows).
After downloading the software and setting up the CLI interface, you can download an LLM, say Llama 3.2 with 1B parameters with:
```bash
ollama pull llama3.2:1b
```

Then use the same model name with `"ollama:"` prefix to run in the UI:
```bash
./run_chat.sh --model_name ollama:llama3.2:1b
```

### Local Llama through Huggingface

**Note:** Only multimodal models like Llama3.2 supported for now.
First, request a node with an A100 (for the Llama 3.2 11B model) and note the compute node's ID, e.g. `c0021`.
Then, on the compute node,
```bash
# If you have already downloaded Llama 3.2 to a local directory
export LLAMA_MODEL_PATH=/mnt/beegfs/bulk/mirror/localllama/localLlama-3.2-11B-Vision-Instruct
./run_chat.sh --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --model_local_path $LLAMA_MODEL_PATH

# If not downloaded, ask huggingface to download for you
huggingface-cli login
./run_chat.sh --model_name meta-llama/Llama-3.2-11B-Vision-Instruct
```

Now on your local machine (not on the cluster), open a new terminal and run `./scripts/forward_streamlit_port_slurm_to_mac.sh c0021`.
Then open http://localhost:8501 in your browser.
This script forwards port 8501 from the compute node c0021 -> AIDA head node -> your laptop.
You might need to need to change the `aida` in the SSH cmd in that script to `user@aida.cac.cornell.edu`.

## TODOs

- Image uploads through the UI. Should be doable.
