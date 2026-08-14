"""
gen_data.py — Generate synthetic messy->clean JSONL training splits
====================================================================

DESCRIPTION:
- Builds a catalog of canonical (already-clean) SAP-style master-data records.
- Applies random "corruption" tactics to each clean record to produce a messy
  version, then re-normalises the messy version with convention_spec to obtain
  the canonical target (including confidence + changes).
- Writes three JSONL files (train/valid/test) where each line contains:
      {"messy": {...}, "clean": {...}}
  It only uses the Python standard library.

PREREQUISITES:
1. Python 3.8+
2. No third-party packages required (runs anywhere, no ML stack).

SETUP FOR NEW USERS:
1. Clone the repo and cd into it.
2. `uv pip install -r requirements.txt`   (or just: uv run scripts/gen_data.py)

USAGE:
    uv run python -m scripts.gen_data \
        --out data \
        --seed 42 \
        --train 640 --valid 80 --test 80

OUTPUT:
- data/train.jsonl  (default 640 samples)
- data/valid.jsonl  (default 80 samples)
- data/test.jsonl   (default 80 samples)
Each line: {"messy": {...}, "clean": {...}}
Console summary of split sizes and a corruption-tactic histogram.

DEPENDENCIES (stdlib only):
- json, random, argparse, pathlib

TROUBLESHOOTING:
- "No module named 'convention_spec'": run as a module from the repo root so the
  project root is on sys.path, i.e. `uv run python -m scripts.gen_data`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from convention_spec import make_clean_template, normalize_record

# ---------------------------------------------------------------------------
# Catalog: varied canonical field values
# ---------------------------------------------------------------------------
NAMES: List[str] = [
    "Muster Handels", "Alpen Metall", "Rhein Bau Kontor", "Nordsee Logistik",
    "Bavaria Solutions", "Spree Digital", "Elbe Textil", "Donau Energie",
    "Main Zuliefer", "Kölner Feinwerk", "Hamburg Container", "Sächsische Präzision",
    "Badische Chemie", "Ostsee Handel", "Weser Maritime", "Tauro Export",
]

CITIES: List[str] = [
    "Musterstadt", "München", "Berlin", "Hamburg", "Köln", "Frankfurt",
    "Stuttgart", "Leipzig", "Dresden", "Nürnberg", "Düsseldorf", "Bremen",
]

LEGAL_FORMS: List[str] = [
    "GmbH", "AG", "GbR", "OHG", "UG", "KG", "e.K.", "GmbH & Co. KG",
]

COUNTRIES: List[str] = ["DE", "DE", "DE", "DE", "AT", "CH"]

CURRENCIES: List[str] = ["EUR", "EUR", "EUR", "USD", "CHF"]

#: ISO date seeds used to build validFrom variants.
_BASE_DATES = ["2024-03-01", "2023-11-15", "2025-01-20", "2022-07-04"]


def _dbn(value: int) -> str:
    """Return a German public IBAN test string derived from an integer."""
    return f"DE89{value:0>18}"


def _build_clean_record(rng: random.Random, index: int) -> Dict[str, Any]:
    """Return one canonical record with varied field values."""
    name = NAMES[index % len(NAMES)]
    city = CITIES[(index * 3) % len(CITIES)]
    legal_form = LEGAL_FORMS[(index * 5) % len(LEGAL_FORMS)]
    country = COUNTRIES[(index * 7) % len(COUNTRIES)]
    currency = CURRENCIES[(index * 11) % len(CURRENCIES)]
    amount = round(rng.uniform(100.0, 99999.0), 2)
    valid_from = _BASE_DATES[index % len(_BASE_DATES)]

    record = make_clean_template()
    record["name1"] = name
    record["city"] = city
    record["legalForm"] = legal_form
    record["country"] = country
    record["currency"] = currency
    record["amount"] = amount
    record["validFrom"] = valid_from
    record["iban"] = _dbn(index)
    return record


# ---------------------------------------------------------------------------
# Corruption tactics (each returns a modified COPY of the record)
# ---------------------------------------------------------------------------
def _corrupt_spaces(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    out["name1"] = f"  {out['name1']}  "
    if rng.random() < 0.5:
        out["city"] = f" {out['city']} "
    return out


def _corrupt_country(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    mapping = {"DE": "Germany", "AT": "Austria", "CH": "Switzerland"}
    out["country"] = mapping.get(out["country"], out["country"].lower())
    return out


def _corrupt_iban(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    raw = out["iban"]
    grouped = " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))
    out["iban"] = grouped.lower() if rng.random() < 0.5 else grouped
    return out


def _corrupt_currency(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    symbol = {"EUR": "€", "USD": "$", "CHF": "CHF"}.get(out["currency"])
    out["currency"] = symbol if symbol else out["currency"].lower()
    return out


def _corrupt_status(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    out["status"] = "aktiv" if out["status"] == "active" else "inaktiv"
    return out


def _corrupt_date(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    year, month, day = out["validFrom"].split("-")
    sep = "." if rng.random() < 0.5 else "/"
    out["validFrom"] = f"{day}{sep}{month}{sep}{year}"
    return out


def _corrupt_amount(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = dict(rec)
    amt = out["amount"]
    int_part = f"{int(amt):,}".replace(",", ".")
    dec = f"{abs(amt - int(amt)):.2f}".split(".")[1]
    out["amount"] = f"{int_part},{dec}"
    return out


_TACTICS: List[Any] = [
    _corrupt_spaces,
    _corrupt_country,
    _corrupt_iban,
    _corrupt_currency,
    _corrupt_status,
    _corrupt_date,
    _corrupt_amount,
]


def _make_messy(clean: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Apply a random non-empty subset of corruption tactics."""
    messy = dict(clean)
    n_tactics = rng.randint(1, len(_TACTICS))
    chosen = rng.sample(_TACTICS, n_tactics)
    for tactic in chosen:
        messy = tactic(messy, rng)
    return messy


def generate_samples(n: int, seed: int) -> List[Dict[str, Any]]:
    """Return ``n`` {messy, clean} pairs; each clean is the normalised target."""
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    for i in range(n):
        clean_base = _build_clean_record(rng, i)
        messy = _make_messy(clean_base, rng)
        clean_target = normalize_record(messy)
        samples.append({"messy": messy, "clean": clean_target})
    return samples


def write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=640)
    parser.add_argument("--valid", type=int, default=80)
    parser.add_argument("--test", type=int, default=80)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    try:
        train = generate_samples(args.train, seed=args.seed)
        valid = generate_samples(args.valid, seed=args.seed + 1)
        test = generate_samples(args.test, seed=args.seed + 2)
    except Exception as e:
        print(f"Error generating samples: {e}")
        print(f"Ensure 'convention_spec.py' is in the Python path (e.g., run as: uv run python -m scripts.gen_data)")
        raise

    try:
        write_jsonl(train, out_dir / "train.jsonl")
        write_jsonl(valid, out_dir / "valid.jsonl")
        write_jsonl(test, out_dir / "test.jsonl")
    except (IOError, OSError) as e:
        print(f"Error writing JSONL files: {e}")
        print(f"Ensure write permissions for output directory: {out_dir}")
        raise

    print(f"train: {len(train)} samples -> {out_dir / 'train.jsonl'}")
    print(f"valid: {len(valid)} samples -> {out_dir / 'valid.jsonl'}")
    print(f"test : {len(test)} samples -> {out_dir / 'test.jsonl'}")
    print(f"seed : {args.seed}")


if __name__ == "__main__":
    main()
