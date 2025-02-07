# Instructions for AIDA cluster

## GPU allocation suggestions

For loading these Huggingface models in `bfloat16` format, we recommend the following allocation.
How do we calculate these allocations?
`bfloat16` is 2 bytes per parameter, and so a 1B parameter model requires 2B of GPU RAM.
We suggest a bit more.

| LLM Name                        | Required GPU RAM | Recommended GPU Allocation |
|---------------------------------|------------------|----------------------------|
| Llama-3.2-11B-Vision-Instruct   | >25GB            | 1 A100 or H100             |
| Llama-3.2-90B-Vision-Instruct   | >190GB           | 4 A100 or H100             |

## Running the Chat Interface

First, request a node with an A100 (for the Llama 3.2 11B model) and note the compute node's ID, e.g. `c0021`.
Then, on the compute node,
```bash
# If you have already downloaded Llama 3.2 to a local directory
export LLAMA_MODEL_PATH=/mnt/beegfs/bulk/mirror/hf_models/llama-3.2-11b
./run_chat.sh --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --model_local_path $LLAMA_MODEL_PATH

# If not downloaded, ask huggingface to download for you
# By default huggingface will download in your local cache https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
huggingface-cli login
./run_chat.sh --model_name meta-llama/Llama-3.2-11B-Vision-Instruct
```

Now on your local machine (not on the cluster), open a new terminal and run `./scripts/forward_streamlit_port_slurm_to_mac.sh aida c0021`.
Here `aida` is a Host in my local machine's `~/.ssh/config`, or you can use any other Host cluster that you've setup in your SSH config.
Otherwise, specify the `username@remote` such as `username@aida.cac.cornell.edu`.

Then open http://localhost:8501 in your browser.
This script forwards port 8501 from the compute node c0021 -> AIDA head node -> your laptop.
