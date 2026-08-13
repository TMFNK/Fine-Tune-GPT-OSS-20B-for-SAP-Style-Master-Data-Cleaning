# Fine-Tune GPT-OSS-20B for SAP-Style Master-Data Cleaning — Implementation Roadmap

**Project**: `gpt-oss-sap-cleaner`
**Base Model**: OpenAI GPT-OSS-20B (via Unsloth on HuggingFace)
**Target Hardware**: Google Colab T4 (free tier) → export GGUF for local inference
**Business Fit**: Privacy-first, on-premise data cleaning for German SMEs (SAP master data)

## Status

| Step | Status |
|------|--------|
| Scaffold + `.gitignore` + `requirements.txt` + `Makefile` | ✅ Done |
| `convention_spec.py` (deterministic teacher rules) | ✅ Done |
| `scripts/gen_data.py` (synthetic JSONL generation) | ✅ Done |
| `train.ipynb` (Unsloth LoRA notebook) | ✅ Done |
| `inference_demo.py` (GGUF local demo) | ✅ Done |
| `scripts/eval.py` + `make eval` (baseline + model scoring) | ✅ Done |
| CI (data determinism, self-tests, baseline eval) | ✅ Done |
| Fuse + GGUF export (`make fuse` / `make gguf`) | ⏳ Pending training |
| Live training run on Colab T4 | ⏳ Pending |

---

## Directory Layout (scaffold)

```
gpt-oss-sap-cleaner/
├── .gitignore                  # __pycache__/, *.gguf, ~/.cache/huggingface/
├── README.md                   # Project overview, links
├── requirements.txt            # Python dependencies
├── Makefile                    # Convenience targets
├── convention_spec.py          # Deterministic "teacher" rule file
├── scripts/
│   └── gen_data.py             # Creates messy↔clean JSONL splits
├── data/
│   ├── train.jsonl             # 640 samples (80%)
│   ├── valid.jsonl             # 80 samples  (10%)
│   └── test.jsonl              # 80 samples  (10%)
├── train.ipynb                 # Colab notebook (Unsloth)
├── inference_demo.py           # GGUF demo script
└── projects/
    └── gpt-oss-20b-finetune-roadmap.md   ←  THIS FILE
```

## Makefile Targets

| Target | Purpose |
|--------|---------|
| `make install` | Install Python deps (`unsloth`, `transformers==4.56.2`, `trl==0.22.2`, etc.) |
| `make data` | Generate synthetic messy↔clean JSONL pairs |
| `make train` | Run fine-tuning in Jupyter notebook (`max_steps≈30`, `r=8`) |
| `make eval` | Score baseline vs fine-tuned model on held-out test set |
| `make fuse` | Merge LoRA adapter into base model |
| `make gguf` | Export GGUF file (`[alias]-q8_0.gguf`, ~600MB) |
| `make serve` | Spin up `llama.cpp` HTTP server |
| `make demo` | Run one clean record through the model |

## Step 1 — Convention Specification (`convention_spec.py`)

Deterministic normalisation logic for SAP master data. Run the self-test with:

```bash
uv run convention_spec.py
```

Handled fields: `name1`, `legalForm`, `country`, `iban`, `currency`, `status`,
`validFrom`, `amount`. Each normalizer returns `(clean_value, changed)` so the
record builds a `changes` list and a deterministic `confidence` score.

## Step 2 — Synthetic Data Generation (`scripts/gen_data.py`)

```bash
uv run python -m scripts.gen_data --out data --seed 42 --train 640 --valid 80 --test 80
```

Output: `data/train|valid|test.jsonl`, each line `{"messy": {...}, "clean": {...}}`.
Every `clean` is the exact output of `convention_spec.normalize_record(messy)`, so the
training target is guaranteed consistent (verified: 0 mismatches across 800 samples).

## Step 3 — Colab Training Notebook (`train.ipynb`)

- Load `unsloth/gpt-oss-20b` in 4-bit, `max_seq_length=1024`
- LoRA `r=8`, `lora_alpha=16`, targeting q/k/v/o/gate/up/down projections
- `SFTTrainer` with `train_on_responses_only`, `max_steps=30`
- Saves adapter to `output/lora_adapter`

## Step 4 — Export GGUF

```bash
make fuse
make gguf   # → output/gpt-oss-sap-cleaner-q8_0.gguf
```

## Step 5 — Inference Demo (`inference_demo.py`)

```bash
uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf
```

Sample output shape:

```json
{
  "name1": "Muster Handels",
  "legalForm": "GmbH",
  "city": "Musterstadt",
  "country": "DE",
  "iban": "DE89370400440532013000",
  "currency": "EUR",
  "status": "active",
  "validFrom": "2024-03-01",
  "amount": 1234.56,
  "confidence": 0.97,
  "changes": ["country: 'Germany' -> 'DE'", "..."]
}
```

## GitHub Push Checklist

- [x] `requirements.txt` + `.gitignore` finalized
- [x] `convention_spec.py` initial mappings in place
- [x] `scripts/gen_data.py` generates valid JSONL
- [x] `train.ipynb` commits cleanly
- [ ] `inference_demo.py` runs against exported GGUF (needs a trained model)
- [x] `README.md` links to project resources

## Estimated Timeline

| Milestone | Time on free T4 |
|-----------|-----------------|
| Data generation | 5 min |
| Training (30 steps, LoRA) | 15 min |
| Fuse + GGUF export | 10 min |
| Demo & verification | 5 min |
| **Total** | ~35 min |