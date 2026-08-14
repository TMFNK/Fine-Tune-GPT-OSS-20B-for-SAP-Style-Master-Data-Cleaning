# ============================================================================
# gpt-oss-sap-cleaner Makefile
# ----------------------------------------------------------------------------
# Convenience targets for the full pipeline. Most ML targets assume a
# GPU environment (Colab T4). Data generation (make data) runs anywhere.
# ============================================================================
 
SHELL := /bin/bash
.PHONY: help install data train eval fuse gguf serve demo clean

PROJECT    := gpt-oss-sap-cleaner
ALIAS      := gpt-oss-sap-cleaner
GGUF_MODEL := output/$(ALIAS)-q8_0.gguf

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies (full stack)
	uv pip install -r requirements.txt

local: ## Install only the local-inference stack (llama-cpp-python)
	uv pip install -r requirements-local.txt

data: ## Generate synthetic messy->clean JSONL splits (train/valid/test)
	uv run python -m scripts.gen_data

train: ## Fine-tune GPT-OSS-20B (run inside train.ipynb on Colab T4)
	@echo "Open train.ipynb in Google Colab and run all cells (max_steps=30, r=8)."

eval: ## Score the cleaner on the held-out test set (baseline by default)
	uv run python -c " \
		import json; \
		from pathlib import Path; \
		import sys; \
		baseline_matches = 0; \
		total_records = 0; \
		for line in Path('data/test.jsonl').open(): \
			item = json.loads(line); \
			total_records += 1; \
			preprocessor = {k: v for k, v in item['messy'].items() if k != 'status'}; \
			if item['clean'].get('status') == preprocessor.get('status', 'active'): \
				baseline_matches += 1; \
		accuracy = baseline_matches / total_records if total_records > 0 else 0; \
		print(f'Baseline accuracy (pre-processing rules): {accuracy:.2%}') \
	"

fuse: ## Merge LoRA adapter into base model
	@echo "TODO: merge LoRA checkpoint into base model (see train.ipynb)."

gguf: fuse ## Export GGUF from merged model
	@echo "Converting to GGUF -> $(GGUF_MODEL)"
	@echo "Run: unsloth.save_pretrained_gguf(...) or convert_hf_to_gguf.py"

serve: ## Spin up llama.cpp OpenAI-compatible HTTP server (llama-cpp-python)
	python -m llama_cpp.server --model $(GGUF_MODEL) --n_gpu_layers 0

demo: ## Run one clean record through the GGUF model
	uv run inference_demo.py --gguf $(GGUF_MODEL)

clean: ## Remove generated data and models
	rm -rf data/*.jsonl output/ checkpoints/

.PHONY: help install local data train eval fuse gguf serve demo clean