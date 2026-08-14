# Fine-Tune-GPT-OSS-20B-for-SAP-Style-Master-Data-Cleaning

DESCRIPTION:
Fine-tune a local Small Language Model (SLM) on synthetic data to clean messy
SAP-style master data, and run the whole thing locally on a Mac. No cloud, no GPU,
no per-token bill. The intelligence lives in a model you own, and the only thing
it costs to run is the electricity.

This project is a beginner-friendly, end-to-end tutorial that walks you through
every step — from generating synthetic training data to running a clean record
through your fine-tuned model on your own machine. No machine-learning expertise
is assumed; you just copy the commands.

PREREQUISITES:

1. Python 3.8+ must be installed
2. uv package manager must be installed
3. A Mac with Apple Silicon (M1/M2/M3/M4, anything from 2020 on). 8 GB RAM is enough
4. About 5 GB of free disk space, plus an internet connection for the one-time downloads
5. 30 to 45 minutes of time

SETUP FOR NEW USERS:

1. Install uv: `pip install uv` (or follow the official instructions at https://docs.astral.sh/uv/)
2. Clone this repo: `git clone https://github.com/TMFNK/Fine-Tune-GPT-OSS-20B-for-SAP-Style-Master-Data-Cleaning.git`
3. Navigate into the repo: `cd Fine-Tune-GPT-OSS-20B-for-SAP-Style-Master-Data-Cleaning`
4. Install dependencies (full stack, includes unsloth for training + llama-cpp for inference):
   `uv pip install -r requirements.txt`
   (Or for inference-only: `uv pip install -r requirements-local.txt`)
5. Generate synthetic training data: `make data`
6. Open the training notebook in Google Colab: `jupyter notebook train.ipynb`
   (or follow the Colab instructions below)

FILE REQUIREMENTS:

- This repo requires no external input files; all data is synthetic and generated
  on-the-fly by `make data`. The generated files are: data/train.jsonl, data/valid.jsonl,
  data/test.jsonl (640 train + 80 valid + 80 test samples).
- For training, a Google Colab T4 GPU is recommended (free tier). Local inference
  after training runs on any Mac with Apple Silicon.
- The convention_spec.py file defines the deterministic "teacher" rules that both
  generate the synthetic data and serve as the ground-truth eval reference.

USAGE:
uv run python -m scripts.gen_data --out data --seed 42 # generate synthetic JSONL
uv run python -m scripts.eval --data data/test.jsonl --baseline # baseline rule accuracy # Then follow the Colab training, GGUF export, and demo steps below

    # Colab T4 training (free tier):
    # Open train.ipynb in Google Colab and run all cells (max_steps=30, r=8).
    # After training, run `make gguf` to export the GGUF model.

    # Local inference (after GGUF export):
    uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf --sample 0

    # Or serve the model locally and eval:
    make serve           # starts llama.cpp server
    make eval            # scores fine-tuned model against held-out test set

OUTPUT:

- data/train.jsonl (640 samples) — training split
- data/valid.jsonl (80 samples) — validation split
- data/test.jsonl (80 samples) — held-out test split
- output/lora_adapter/ — LoRA adapter weights (a few MB)
- output/gpt-oss-sap-cleaner-q8_0.gguf — quantised GGUF model (~600 MB) for local inference
- Optional: eval-results.json — JSON summary of model accuracy

DEPENDENCIES (automatically installed by uv):

- pandas>=2.0.0
- openpyxl>=3.1.0
- unsloth>=2025.3
- unsloth-zoo>=2025.3
- transformers==4.56.2
- trl==0.22.2
- torch>=2.5
- peft>=0.14
- accelerate>=1.2
- datasets>=3.2
- bitsandbytes>=0.45
- sentencepiece>=0.2.0
- jupyter>=1.1
- llama-cpp-python>=0.3.0 (inference-only stack: requirements-local.txt)

TROUBLESHOOTING:

- `command not found: make`, `brew` or `git`: install Xcode Command Line Tools
  (`xcode-select --install`) or use `uv run` equivalents.
- `command not found: mlx_lm...`: run `make setup` again to ensure MLX is available.
- `Address already in use` on port 8080: a server is still running in another
  Terminal. Find it and press `Ctrl + C`, or use another port with
  `make serve PORT=8081` (then `make eval PORT=8081`1).
- The download is slow: the pulls are one-time and cache after first run.
- Out of memory during `make train`: stop any running server first, and if needed
  lower the batch size with `make train BATCH=2`.
- Connection refused or `Cannot reach the model server`: the model server is not
  running yet. Make sure `make baseline-serve` or `make serve` is running in the
  other Terminal and has printed its "listening" line, then run the command again.
- You closed the Terminal mid-way: nothing is lost. Open a new one, `cd` into the
  project folder, and continue from the step you were on. Finished steps do not need
  to be redone; downloads and generated files are still there.

## About

Built by mbitai, freelance data and AI engineering for German businesses, with a
focus on practical, privacy-first machine learning that runs where your data already
lives. This repo is part of that portfolio and a worked example: local, tiny, open,
and GDPR-friendly by design.

For commercial licensing without AGPL obligations, or help applying this to your own
master data, contact www.mbitai.com.

## License

MIT (see LICENSE file). All sample data is synthetic and invented.
