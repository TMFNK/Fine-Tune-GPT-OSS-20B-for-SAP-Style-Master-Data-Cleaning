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
make demo   # or directly:
# uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf --sample 0
```

## Repository layout

```tree
convention_spec.py        Deterministic "teacher" normalisation rules (stdlib only)
scripts/gen_data.py       Synthetic messy->clean JSONL generation (make data)
data/train|valid|test.jsonl  800 generated samples (640/80/80)
train.ipynb               Unsloth LoRA fine-tuning notebook (Colab T4)
inference_demo.py         Local GGUF inference demo
projects/                 Implementation roadmap (MOC)
```

See [`projects/gpt-oss-20b-finetune-roadmap.md`](projects/gpt-oss-20b-finetune-roadmap.md)
for the step-by-step implementation status.

## Models

- **Base**: [GPT-OSS-20B](https://huggingface.co/unsloth/gpt-oss-20b)
- **LoRA adapter**: `gpt-oss-sap-cleaner`
- **GGUF (local)**: `output/gpt-oss-sap-cleaner-q8_0.gguf` (~600 MB)

## Bringing your own messy data

The synthetic `data/*.jsonl` splits are for training and CI. In practice you
clean **your own messy SAP records**: dump them as JSON lines and run them through

```bash
uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf \
    --input '{"name1":" muster handels ","country":"Germany","amount":"1.234,56"}'
```

or pipe a whole file through the `llama.cpp` server (`make serve`). The model
also handles fields the deterministic `convention_spec` rules don't cover, so
real-world records that the rules can't parse are where the fine-tuned model
earns its keep.

## License

MIT (see LICENSE file)
