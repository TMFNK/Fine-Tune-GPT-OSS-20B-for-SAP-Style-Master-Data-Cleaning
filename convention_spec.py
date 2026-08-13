"""
convention_spec.py — Deterministic normalisation rules for SAP-style master data
=================================================================================

DESCRIPTION:
- Defines the canonical output schema for a cleaned master-data record.
- Provides pure, deterministic "teacher" logic that converts messy values into
  a validated, canonical form (used for both synthetic-data generation and as a
  reference for judging the fine-tuned model's output).
- Every normaliser returns (clean_value, changed: bool) so callers can build a
  human-readable "changes" list and a confidence estimate.
- Deliberately depends ONLY on the Python standard library: it can run on any
  machine (local Mac, CI, or Colab) without installing the ML stack.

PREREQUISITES:
1. Python 3.8+ (tested on 3.14)
2. No third-party packages required.

SETUP FOR NEW USERS:
1. Clone the repo and `cd` into it.
2. (Optional, for a clean env) `uv venv && source .venv/bin/activate`
3. Run the self-test: `uv run convention_spec.py`

USAGE:
    from convention_spec import normalize_record

    clean = normalize_record({"name1": "  muster handels ", "country": "DE"})
    # -> {...all fields normalized..., "confidence": 0.9, "changes": [...]}

    # Or run built-in self-tests:
    uv run convention_spec.py

OUTPUT:
- normalize_record(record) -> dict with every known field normalised, plus a
  "confidence" float in [0.5, 1.0] and a "changes" list of strings.

DEPENDENCIES:
- stdlib only (re, datetime, typing)

TROUBLESHOOTING:
- If `uv run convention_spec.py` is slow on first use, it is just uv creating
  the ephemeral environment; rerun for instant execution.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical field registry
# ---------------------------------------------------------------------------
#: Canonical output fields produced by this module.
FIELDS: Dict[str, str] = {
    "name1": "trimmed & title-cased",
    "city": "trimmed & title-cased",
    "legalForm": "GmbH / AG / GbR / OHG / UG / e.K.",
    "country": "ISO-3166-1 alpha-2 (DE / AT / CH / ...)",
    "iban": "uppercased, no spaces (DE...22)",
    "currency": "ISO-4217 (EUR / USD / ...)",
    "status": "active / inactive",
    "validFrom": "ISO 8601 date (YYYY-MM-DD)",
    "amount": "float (1234.56)",
}

#: Per-key alias maps used by the country / legal-form / currency / status
#: normalisers. Keys are lower-cased during lookup.
COUNTRY_ALIASES: Dict[str, str] = {
    "de": "DE", "germany": "DE", "deutschland": "DE", "deutsch": "DE",
    "at": "AT", "austria": "AT", "österreich": "AT", "osterreich": "AT",
    "ch": "CH", "switzerland": "CH", "schweiz": "CH", "suisse": "CH",
}

LEGAL_FORM_ALIASES: Dict[str, str] = {
    "gmbh": "GmbH", "mbh": "GmbH", "ges.m.b.h": "GmbH",
    "ag": "AG", "aktiengesellschaft": "AG",
    "gbr": "GbR", "gesellschaft bürgerlichen rechts": "GbR",
    "ohg": "OHG", "offene handelsgesellschaft": "OHG",
    "ug": "UG", "e.k": "e.K.", "e.k.": "e.K.", "kg": "KG",
    "gmbh & co. kg": "GmbH & Co. KG",
}

CURRENCY_ALIASES: Dict[str, str] = {
    "€": "EUR", "eur": "EUR", "euro": "EUR",
    "$": "USD", "usd": "USD", "us-dollar": "USD", "us dollar": "USD",
    "chf": "CHF", "sfr": "CHF", "£": "GBP", "gbp": "GBP",
}

STATUS_ALIASES: Dict[str, str] = {
    "aktiv": "active", "aktiv.": "active", "active": "active",
    "yes": "active", "y": "active",
    "inaktiv": "inactive", "inaktiv.": "inactive", "inactive": "inactive",
    "no": "inactive", "n": "inactive",
}

#: German decimal-string separator pattern: 1.234,56 or 1,234.56
_AMOUNT_DE = re.compile(r"^\s*(-?)(\d{1,3}(?:\.\d{3})*)(?:,(\d+))?\s*$")
_AMOUNT_INTL = re.compile(r"^\s*(-?)(\d{1,3}(?:,\d{3})*)(?:\.(\d+))?\s*$")

#: Date patterns accepted, in priority order -> (format)
_DATE_FORMATS: List[Tuple[str, str]] = [
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),
    (r"^\d{2}\.\d{2}\.\d{4}$", "%d.%m.%Y"),
    (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
]


# ---------------------------------------------------------------------------
# Field normalisers  (return (clean_value, changed: bool))
# ---------------------------------------------------------------------------
def _norm_name1(value: Any) -> Tuple[str, bool]:
    text = str(value).strip()
    collapsed = re.sub(r"\s+", " ", text)
    cleaned = collapsed.title()
    return cleaned, cleaned != text


def _norm_city(value: Any) -> Tuple[str, bool]:
    text = str(value).strip()
    collapsed = re.sub(r"\s+", " ", text)
    cleaned = collapsed.title()
    return cleaned, cleaned != text


def _norm_legal_form(value: Any) -> Tuple[str, bool]:
    raw = str(value).strip().lower()
    if not raw:
        return "", False
    cleaned = LEGAL_FORM_ALIASES.get(raw, raw.title())
    return cleaned, cleaned != str(value).strip()


def _norm_country(value: Any) -> Tuple[str, bool]:
    raw = str(value).strip()
    key = raw.lower()
    cleaned = COUNTRY_ALIASES.get(key, raw.upper() if len(raw) == 2 else raw)
    return cleaned, cleaned != raw


def _norm_iban(value: Any) -> Tuple[str, bool]:
    raw = str(value)
    cleaned = re.sub(r"\s+", "", raw).upper()
    return cleaned, cleaned != raw


def _norm_currency(value: Any) -> Tuple[str, bool]:
    raw = str(value).strip()
    key = raw.lower()
    cleaned = CURRENCY_ALIASES.get(key, raw.upper())
    return cleaned, cleaned != raw


def _norm_status(value: Any) -> Tuple[str, bool]:
    raw = str(value).strip()
    key = raw.lower()
    cleaned = STATUS_ALIASES.get(key, raw.lower())
    return cleaned, cleaned != raw


def _norm_valid_from(value: Any) -> Tuple[str, bool]:
    raw = str(value).strip()
    for pattern, fmt in _DATE_FORMATS:
        if re.match(pattern, raw):
            try:
                cleaned = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                return raw, False
            return cleaned, cleaned != raw
    return raw, False


def _norm_amount(value: Any) -> Tuple[Any, bool]:
    if isinstance(value, (int, float)):
        return float(value), False
    raw = str(value).strip()

    m = _AMOUNT_DE.match(raw)  # German: 1.234,56
    if m:
        sign, int_part, dec_part = m.groups()
        digits = int_part.replace(".", "")
        num = float(f"{sign}{digits}.{dec_part or '0'}")
        return round(num, 2), True

    m = _AMOUNT_INTL.match(raw)  # ISO: 1,234.56
    if m:
        sign, int_part, dec_part = m.groups()
        digits = int_part.replace(",", "")
        num = float(f"{sign}{digits}.{dec_part or '0'}")
        return round(num, 2), True

    return raw, False


_NORMALIZERS: Dict[str, Any] = {
    "name1": _norm_name1,
    "city": _norm_city,
    "legalForm": _norm_legal_form,
    "country": _norm_country,
    "iban": _norm_iban,
    "currency": _norm_currency,
    "status": _norm_status,
    "validFrom": _norm_valid_from,
    "amount": _norm_amount,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a messy master-data record into canonical form.

    Every known field is cleaned via its normaliser. Unknown keys are passed
    through untouched. A ``confidence`` float and a ``changes`` list are
    appended to the result.
    """
    clean: Dict[str, Any] = {}
    changes: List[str] = []

    for key, value in record.items():
        normalizer = _NORMALIZERS.get(key)
        if normalizer is None or value is None:
            clean[key] = value
            continue
        try:
            cleaned, changed = normalizer(value)
        except (TypeError, ValueError):
            clean[key] = value
            continue

        clean[key] = cleaned
        if changed:
            changes.append(f"{key}: '{value}' -> '{cleaned}'")

    # Deterministic confidence: lower it with each change, floor at 0.5.
    n_changes = len(changes)
    confidence = max(0.5, round(1.0 - 0.05 * n_changes, 2))
    clean["confidence"] = confidence
    clean["changes"] = changes
    return clean


def make_clean_template() -> Dict[str, Any]:
    """Return a canonical, already-clean record (template used by gen_data)."""
    return {
        "name1": "Muster Handels",
        "legalForm": "GmbH",
        "city": "Musterstadt",
        "country": "DE",
        "iban": "DE89370400440532013000",
        "currency": "EUR",
        "status": "active",
        "validFrom": "2024-03-01",
        "amount": 1234.56,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _run_self_tests() -> None:
    n_pass = 0
    n_fail = 0

    def check(desc: str, got: Any, expected: Any) -> None:
        nonlocal n_pass, n_fail
        if got == expected:
            n_pass += 1
        else:
            n_fail += 1
            print(f"  [FAIL] {desc}: got {got!r}, expected {expected!r}")

    check("name1 trim+title", _norm_name1("  muster handels  ")[0], "Muster Handels")
    check("city trim+title", _norm_city("  hamburg  ")[0], "Hamburg")
    check("legalForm mbH", _norm_legal_form("mbH")[0], "GmbH")
    check("legalForm gbr", _norm_legal_form("gbr")[0], "GbR")
    check("country Germany", _norm_country("Germany")[0], "DE")
    check("country de", _norm_country("de")[0], "DE")
    check("iban case+space", _norm_iban("de89 3704 0044 0532 0130 00")[0],
          "DE89370400440532013000")
    check("currency EUR", _norm_currency("EUR")[0], "EUR")
    check("currency eur", _norm_currency("eur")[0], "EUR")
    check("currency euro symbol", _norm_currency("€")[0], "EUR")
    check("status aktiv", _norm_status("aktiv")[0], "active")
    check("status inaktiv", _norm_status("inaktiv")[0], "inactive")
    check("validFrom DD.MM.YYYY", _norm_valid_from("01.03.2024")[0], "2024-03-01")
    check("validFrom DD/MM/YYYY", _norm_valid_from("01/03/2024")[0], "2024-03-01")
    check("validFrom ISO", _norm_valid_from("2024-03-01")[0], "2024-03-01")
    check("amount DE 1.234,56", _norm_amount("1.234,56")[0], 1234.56)
    check("amount DE plain 12,5", _norm_amount("12,5")[0], 12.5)
    check("amount INT 1,234.56", _norm_amount("1,234.56")[0], 1234.56)
    check("amount float passthrough", _norm_amount(1234.56)[0], 1234.56)

    # Full-record test
    messy = {
        "name1": "  muster handels ",
        "legalForm": "mbH",
        "country": "Germany",
        "iban": "de89 3704 0044 0532 0130 00",
        "currency": "€",
        "status": "aktiv",
        "validFrom": "01.03.2024",
        "amount": "1.234,56",
    }
    clean = normalize_record(messy)
    expected = {
        "name1": "Muster Handels",
        "legalForm": "GmbH",
        "country": "DE",
        "iban": "DE89370400440532013000",
        "currency": "EUR",
        "status": "active",
        "validFrom": "2024-03-01",
        "amount": 1234.56,
    }
    for key, val in expected.items():
        check(f"record.{key}", clean.get(key), val)

    check("record.changes count", len(clean["changes"]), 8)
    check("record.confidence", clean["confidence"], 0.6)  # 8 * 0.05 off 1.0 -> 0.6

    print(f"\nconvention_spec self-tests: {n_pass} passed, {n_fail} failed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
