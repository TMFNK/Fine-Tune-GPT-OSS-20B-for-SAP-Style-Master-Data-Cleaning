# Fine-Tune-GPT-OSS-20B-for-SAP-Style-Master-Data-Cleaning

## TLDR

Fine-tune a 20B model on synthetic data to clean SAP-style master records into validated JSON. No cloud, no GPU and runs entirely locally on a Mac.

## Library dependencies

- `unsloth` (LoRA fine-tuning)
- `unsloth-zoo` (GPU optimizations)
- `transformers==4.56.2` (HF interface)
- `trl` (training tools)
- `llama-cpp-python` (local deployment)

## Pipeline overview

1. **Synthetic data generation** – based on `convention_spec.py`
2. **LoRA fine-tuning** – GPT-OSS-20B (0.6% params trained)
3. **GGUF export** – local inference with `llama.cpp`
4. **Fast deployment** – `llama-gguf-cli` serves an OpenAI-compatible endpoint

## Getting started

```bash
# Clone repo
git clone https://github.com/TMFNK/Fine-Tune-GPT-OSS-20B-for-SAP-Style-Master-Data-Cleaning.git

# Install dependencies
uv pip install -r requirements.txt

# Generate training data
make data

# Fine-tune (run on Colab T4)
jupyter notebook train.ipynb

# Export GGUF model
make gguf

# Run local inference
./llama.cpp/llama-gguf-cli -m output/gpt-oss-sap-cleaner-q8_0.gguf
```

## Models

- **Base**: [GPT-OSS-20B](https://huggingface.co/unsloth/gpt-oss-20b)
- **LoRA adapter**: `qwen3-0.6b-cleaner`

## Legal

MIT (see LICENSE file)

## License

MIT
