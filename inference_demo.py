"""
inference_demo.py — GGUF local inference demo for gpt-oss-sap-cleaner
======================================================================

DESCRIPTION:
- Loads the exported GGUF model (llama.cpp) and cleans a messy SAP-style
  master-data record into validated JSON.
- Builds the same system/user chat prompt used during fine-tuning.
- Parses the assistant's JSON reply and (optionally) adds a deterministic
  `confidence` + `changes` from convention_spec as a cross-check.

PREREQUISITES:
1. Python 3.8+
2. A quantised GGUF model, e.g. output/gpt-oss-sap-cleaner-q8_0.gguf
3. `uv pip install -r requirements.txt` (needs llama-cpp-python)

SETUP FOR NEW USERS:
1. Export the GGUF (see train.ipynb / `make gguf`).
2. Install deps: `uv pip install -r requirements.txt`

USAGE:
    uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf \
        --input '{"name1":" muster handels ","country":"Germany","amount":"1.234,56"}'

    # Use a held-out test sample from data/test.jsonl:
    uv run inference_demo.py --gguf output/gpt-oss-sap-cleaner-q8_0.gguf --sample 0

    # Show help without loading the model:
    uv run inference_demo.py --help

OUTPUT:
- Prints the messy input, the raw assistant reply, and the parsed clean JSON
  (with confidence/changes when enabled).

DEPENDENCIES:
- llama-cpp-python>=0.3.0 (GGUF inference)
- stdlib (json, argparse, pathlib)

TROUBLESHOOTING:
- "No module named 'llama_cpp'": `uv pip install llama-cpp-python` or run
  `make install` first.
- No --gguf given: the script prints usage only (does not require llama_cpp).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You clean messy SAP-style master data into valid JSON. "
    "Return only the JSON object, no commentary."
)


def build_messages(messy: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the system/user message list used for inference."""
    user = "Clean this record: " + json.dumps(messy, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of JSON from the model reply (tolerates code fences)."""
    text = content.strip()
    # Strip triple-backtick fences if the model wrapped the JSON.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def clean_record(
    messy: Dict[str, Any],
    llm: Any,
    max_tokens: int = 512,
    temperature: float = 0.0,
    cross_check: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run one record through the model and return (parsed_json, raw_reply)."""
    messages = build_messages(messy)
    resp = llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    content = resp["choices"][0]["message"]["content"]
    parsed = extract_json(content)

    if cross_check and parsed is not None:
        try:
            from convention_spec import normalize_record

            ref = normalize_record(messy)
            parsed.setdefault("confidence", ref.get("confidence"))
            parsed.setdefault("changes", ref.get("changes"))
        except ImportError:
            pass  # convention_spec is optional at runtime
    return parsed, content


def _load_test_sample(index: int) -> Dict[str, Any]:
    path = Path("data/test.jsonl")
    if not path.exists():
        raise FileNotFoundError("data/test.jsonl missing — run `make data` first.")
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[index % len(lines)])["messy"]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", help="Path to the GGUF model file.")
    parser.add_argument("--input", help="Messy record as a JSON string.")
    parser.add_argument("--sample", type=int, help="Use data/test.jsonl row index.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-cross-check", action="store_true",
                        help="Do not add confidence/changes from convention_spec.")
    args = parser.parse_args(argv)

    if not args.gguf:
        parser.print_help()
        sys.exit(0)

    if args.input:
        messy = json.loads(args.input)
    elif args.sample is not None:
        messy = _load_test_sample(args.sample)
    else:
        messy = _load_test_sample(0)

    try:
        from llama_cpp import Llama  # imported lazily so --help works w/o pkg
    except ImportError as exc:  # pragma: no cover - env dependent
        parser.error(f"llama_cpp not installed ({exc}). Run `make install`.")

    print(f"GGUF model: {args.gguf}")
    llm = Llama(model_path=args.gguf, n_ctx=2048, verbose=False)

    print("\n--- messy input ---")
    print(json.dumps(messy, ensure_ascii=False, indent=2))

    parsed, raw = clean_record(
        messy, llm,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        cross_check=not args.no_cross_check,
    )

    print("\n--- raw model reply ---")
    print(raw)

    print("\n--- parsed clean record ---")
    if parsed is None:
        print("(could not parse JSON from model reply)")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
