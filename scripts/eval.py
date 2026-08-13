"""
eval.py — Score the cleaner on the held-out test set
======================================================

DESCRIPTION:
- Loads data/test.jsonl (ground-truth {messy, clean} pairs).
- Produces a prediction for each messy record using one of two backends:
    * --baseline (default): the deterministic `convention_spec.normalize_record`
      rule engine — a self-consistent "teacher" reference.
    * --gguf <file>: the fine-tuned GGUF model via llama.cpp.
- Reports per-field accuracy, full-record exact-match rate, and change-list
  coverage, plus a JSON summary file (--json-out).

PREREQUISITES:
1. Python 3.8+
2. Baseline mode needs no extra deps (stdlib only).
3. Model mode needs a trained GGUF + `uv pip install -r requirements-local.txt`.

SETUP FOR NEW USERS:
1. Clone the repo and cd into it.
2. `uv pip install -r requirements-local.txt`   (only for --gguf mode)

USAGE:
    # Baseline (no model needed):
    uv run python -m scripts.eval --data data/test.jsonl --baseline

    # Fine-tuned model:
    uv run python -m scripts.eval --data data/test.jsonl \
        --gguf output/gpt-oss-sap-cleaner-q8_0.gguf --json-out eval-results.json

OUTPUT:
- A human-readable report on stdout.
- Optional JSON summary (--json-out) with the same metrics.

DEPENDENCIES:
- stdlib only (baseline mode); llama-cpp-python for --gguf mode.

TROUBLESHOOTING:
- "No module named 'scripts'": run as a module from the repo root,
  i.e. `uv run python -m scripts.eval`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Canonical fields that the model/teacher is expected to produce.
CANONICAL_FIELDS: List[str] = [
    "name1", "legalForm", "city", "country", "iban",
    "currency", "status", "validFrom", "amount",
]


def load_samples(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load {messy, clean} pairs; returns a list of records."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `make data` first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def ground_truth(clean: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the meta keys (confidence/changes) to compare canonical fields."""
    return {k: v for k, v in clean.items() if k not in ("confidence", "changes")}


def field_accuracy(pred: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, float]:
    """Per-field accuracy over CANONICAL_FIELDS (unseen keys ignored)."""
    acc: Dict[str, float] = {}
    for field in CANONICAL_FIELDS:
        if field not in truth:
            continue
        correct = pred.get(field) == truth[field]
        acc[field] = 1.0 if correct else 0.0
    return acc


def summarize(samples: List[Dict[str, Any]],
              predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all samples."""
    n = len(samples)
    per_field_hits: Dict[str, int] = {f: 0 for f in CANONICAL_FIELDS}
    exact_matches = 0
    change_coverage_total = 0.0
    change_coverage_count = 0

    for sample, pred in zip(samples, predictions):
        truth = ground_truth(sample["clean"])
        acc = field_accuracy(pred, truth)
        for field, score in acc.items():
            per_field_hits[field] += int(score)

        if all(pred.get(f) == truth.get(f) for f in CANONICAL_FIELDS if f in truth):
            exact_matches += 1

        # Change coverage: fraction of ground-truth changes the pred matched.
        truth_changes = set(sample["clean"].get("changes", []))
        if truth_changes:
            pred_changes = set(pred.get("changes", []))
            change_coverage_total += len(truth_changes & pred_changes) / len(truth_changes)
            change_coverage_count += 1

    per_field = {
        field: round(hits / n, 4) if n else 0.0
        for field, hits in per_field_hits.items()
    }
    return {
        "n_samples": n,
        "exact_match_rate": round(exact_matches / n, 4) if n else 0.0,
        "field_accuracy": per_field,
        "mean_change_coverage": round(change_coverage_total / change_coverage_count, 4)
        if change_coverage_count else 0.0,
    }


def print_report(metrics: Dict[str, Any], label: str) -> None:
    print(f"\n=== Eval report [{label}] ===")
    print(f"samples            : {metrics['n_samples']}")
    print(f"exact-match rate   : {metrics['exact_match_rate']:.2%}")
    print(f"mean change coverage: {metrics['mean_change_coverage']:.2%}")
    print("field accuracy     :")
    for field, acc in metrics["field_accuracy"].items():
        print(f"  {field:<10} {acc:.2%}")


# ---------------------------------------------------------------------------
# Prediction backends
# ---------------------------------------------------------------------------
def predict_baseline(messy: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic rule-based prediction via convention_spec."""
    from convention_spec import normalize_record

    return normalize_record(messy)


def predict_model(messy: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """Model prediction via inference_demo.clean_record (lazy llama import)."""
    from inference_demo import clean_record

    parsed, _ = clean_record(messy, llm, max_tokens=512, temperature=0.0)
    return parsed or {}


def load_llm(gguf_path: str) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            f"llama_cpp not installed ({exc}). Run `uv pip install -r requirements-local.txt`."
        ) from exc
    return Llama(model_path=gguf_path, n_ctx=2048, verbose=False)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--baseline", action="store_true",
                        help="Use the rule-based convention_spec as the predictor.")
    parser.add_argument("--gguf", help="Path to the fine-tuned GGUF model.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--json-out", help="Write metrics JSON to this file.")
    args = parser.parse_args(argv)

    if args.gguf and args.baseline:
        parser.error("--gguf and --baseline are mutually exclusive.")

    samples = load_samples(Path(args.data), args.max_samples)

    if args.baseline or not args.gguf:
        label = "baseline (convention_spec)"
        predictions = [predict_baseline(s["messy"]) for s in samples]
    else:
        label = f"model ({Path(args.gguf).name})"
        llm = load_llm(args.gguf)
        predictions = [predict_model(s["messy"], llm) for s in samples]

    metrics = summarize(samples, predictions)
    print_report(metrics, label)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
