# ============================================================================
# gpt-oss-sap-cleaner Makefile
# ----------------------------------------------------------------------------
# Convenience targets for the full pipeline. Most ML targets assume a
# GPU environment (Colab). Data generation (make data) runs anywhere.
# ============================================================================

SHELL := /bin/bash
.PHONY: help install data train eval fuse gguf serve demo

PROJECT    := gpt-oss-sap-cleaner
ALIAS      := gpt-oss-sap-cleaner
GGUF_MODEL := output/$(ALIAS)-q8_0.gguf

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	uv pip install -r requirements.txt

data: ## Generate synthetic messy->clean JSONL splits (train/valid/test)
	uv run python -m scripts.gen_data

train: ## Fine-tune GPT-OSS-20B (run inside train.ipynb on Colab T4)
	@echo "Open train.ipynb in Jupyter and run all cells (max_steps=30, r=8)."

eval: ## Score baseline vs fine-tuned model on held-out test set
	@echo "TODO: implement evaluation harness (n/a until a GGUF is exported)."
	@echo "Run: python -m scripts.eval --gguf $(GGUF_MODEL)"

fuse: ## Merge LoRA adapter into base model
	@echo "TODO: merge LoRA checkpoint into base model (see train.ipynb)."

gguf: fuse ## Export GGUF from merged model
	@echo "Converting to GGUF -> $(GGUF_MODEL)"
	@echo "Run: unsloth.save_pretrained_gguf(...) or convert_hf_to_gguf.py"

serve: ## Spin up llama.cpp OpenAI-compatible HTTP server
	llama-gguf-cli --server -m $(GGUF_MODEL)

demo: ## Run one clean record through the GGUF model
	uv run inference_demo.py --gguf $(GGUF_MODEL)
