import json
import math
import re
import unicodedata
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration Parameters ---
TOTAL_TRUE_ALERTS = 100 # Target total number of TRUE alerts required
TOTAL_FALSE_ALERTS = 200 # Target total number of FALSE alerts required
TOTAL_ROWS = TOTAL_TRUE_ALERTS + TOTAL_FALSE_ALERTS
ROWS_PER_CALL = 10    # Number of rows generated per LLM call

# --- LLM Provider Configuration ---
MODEL_PROVIDER: Literal["gemini", "ollama"] = "ollama"   # Switch between cloud Gemini and a local Ollama model
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"   # LangChain Gemini model name
OLLAMA_MODEL_NAME = "gemma4:31b-cloud"   # Local Ollama model name (must be pulled beforehand)
OLLAMA_BASE_URL = "http://localhost:11434"   # Local Ollama server URL


def _sanitize_filename_component(value: str) -> str:
    """Replace special characters with underscores for filename components."""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return sanitized or "model"


def _configured_model_name() -> str:
    if MODEL_PROVIDER == "gemini":
        return GEMINI_MODEL_NAME
    if MODEL_PROVIDER == "ollama":
        return OLLAMA_MODEL_NAME
    return MODEL_PROVIDER


def _build_output_filename(now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%d%m%Y_%H%M")
    model_name = _sanitize_filename_component(_configured_model_name())
    return f"{timestamp}_{model_name}_compliance_master_dataset.xlsx"

# --- Missing-value configuration ---
# Target fraction of records where each field must be empty (None), split
# independently per decision outcome. Edit these values to customize the
# % missing values for city / country / dob on TRUE vs FALSE alerts.
# "client_*" = the client's data, "hit_*" = the screening hit's data.
MISSING_RATE_CONFIG = {
    "client_city":    {"true": 0.12, "false": 0.60},
    "client_country": {"true": 0.47, "false": 0.40},
    "hit_city":       {"true": 0.22, "false": 0.50},
    "hit_country":    {"true": 0.12, "false": 0.60},
    "client_dob":     {"true": 0.50, "false": 0.50},
    "hit_dob":        {"true": 0.50, "false": 0.50},
}

# --- Enums & Schemas ---
DecisionReason = Literal[
    "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING",
    "NAME_MISMATCH",
    "DOB_MISMATCH_OR_INVALID",
    "GEOGRAPHIC_MISMATCH",
]

class AlertRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    client_name: str
    hit_name: str
    matching_text: str  # exact token(s) in hit_name vs client_name that fired the engine alert
    client_dob: Optional[str] = None
    hit_dob: Optional[str] = None
    client_country: Optional[str] = None
    client_city: Optional[str] = None
    hit_country: Optional[str] = None
    hit_city: Optional[str] = None
    decision: bool
    decision_reason: DecisionReason
    thinking: str

class AlertDatasetBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alerts: List[AlertRecord]

# --- Missing-value helpers ---
_MISSING_PLACEHOLDER_TOKENS = {
    "", "na", "n/a", "none", "null", "nan", "nil", "-", "--", "_",
    "unknown", "not available", "not provided", "missing", "unknown city", "unknown country",
}

TRUE_REASON_OK = "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH"
TRUE_REASON_DOB_MISSING = "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH"
TRUE_REASON_CITY_MISSING = "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING"


def _is_effectively_missing(value) -> bool:
    """True when a raw model value should be treated as empty."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in _MISSING_PLACEHOLDER_TOKENS:
        return True
    return False

def _build_missing_quotas(
    batch_true_count: int, batch_false_count: int, config: dict
) -> str:
    """Render per-batch empty-field quotas injected into the LLM prompt.

    Updated to align strictly with the revised PEP screening decision reasons.
    """
    lines = [
        "<missing_value_quotas>",
        f"This batch has approximately {batch_true_count} TRUE and {batch_false_count} FALSE alerts.",
        'Empty = null/omitted field. Never use placeholder text like "N/A", "-", or "unknown".',
        "Leave each field empty in EXACTLY the stated number of records per decision type:",
    ]

    for decision_label, decision_key, decision_total in (
        ("TRUE", "true", batch_true_count),
        ("FALSE", "false", batch_false_count),
    ):
        lines.append(f"- {decision_label} alerts (total={decision_total}):")
        for field in config:
            target_count = round(decision_total * config[field][decision_key])
            if target_count == 0:
                lines.append(f"    - {field}: 0 (do not leave empty)")
            else:
                lines.append(f"    - {field}: leave empty in {target_count}")

    lines += [
        "- Keep every emptied field consistent with `decision_reason`:",
        "    * TRUE alerts: missing DOB or geographic fields are acceptable; maintain allowed reasons like `EXACT_NAME_MATCH` or `VALID_FUZZY_MATCH`.",
        "    * FALSE alerts: never empty the specific field that drives the false decision (e.g., keep DOBs populated on `DOB_MISMATCH`).",
        "- Briefly note every emptied field in `thinking` (e.g., `hit_city unavailable - bypassing geography step`).",
        "</missing_value_quotas>",
    ]

    return "\n".join(lines)


def _null_field(record: dict, field: str, reason_override: Optional[str] = None) -> None:
    """Set a field to None, optionally remapping decision_reason, and document it."""
    record[field] = None
    if reason_override is not None:
        record["decision_reason"] = reason_override
    note = f" [auto-null: {field}]"
    thinking = record.get("thinking")
    record["thinking"] = f"{thinking.rstrip()} {note}" if thinking else note.lstrip()


def _override_reason(decision_key: str, field: str, record: dict) -> Optional[str]:
    """Return a remapped decision_reason when nulling `field` would contradict it."""
    if decision_key != "true":
        return None
    reason = record["decision_reason"]
    if reason != TRUE_REASON_OK:
        return None
    if field in ("client_dob", "hit_dob", "client_country", "hit_country"):
        return TRUE_REASON_DOB_MISSING
    if field in ("client_city", "hit_city"):
        return TRUE_REASON_CITY_MISSING
    return None


def _pad_candidates(records: List[dict], decision_key: str, field: str) -> List[dict]:
    """Records eligible to have `field` nulled without contradicting their reason."""
    present = [r for r in records if r.get(field) is not None]
    if decision_key == "true":
        if field in ("client_dob", "hit_dob"):
            def dob_priority(r):
                if r["decision_reason"] == TRUE_REASON_DOB_MISSING:
                    return 0  # reason already implies DOB missing; just remove the stray value
                if r["decision_reason"] == TRUE_REASON_OK:
                    return 1  # remap handled by _override_reason
                return 2
            return sorted(present, key=dob_priority)
        if field in ("client_city", "hit_city"):
            def city_priority(r):
                if r["decision_reason"] == TRUE_REASON_CITY_MISSING:
                    return 0
                if r["decision_reason"] == TRUE_REASON_OK and r.get("client_dob") is not None and r.get("hit_dob") is not None:
                    return 1  # DOB stays OK so the CITY_MISSING remap stays valid
                return 2
            return sorted(present, key=city_priority)
        if field in ("client_country", "hit_country"):
            # No dedicated TRUE reason references a missing country; prefer records
            # whose reason the raw model already uses for missing geography.
            return sorted(
                present,
                key=lambda r: 0 if r["decision_reason"] in (TRUE_REASON_DOB_MISSING, TRUE_REASON_CITY_MISSING) else 1,
            )
    else:
        if field in ("client_dob", "hit_dob"):
            return [r for r in present if r["decision_reason"] in ("NAME_MISMATCH", "GEOGRAPHIC_MISMATCH")]
        return [r for r in present if r["decision_reason"] == "NAME_MISMATCH"]
    return []


def enforce_missing_rates(records: List[dict], config: dict = MISSING_RATE_CONFIG) -> List[dict]:
    """Deterministically enforce missing-rate targets per field per decision.

    - Normalizes blank/placeholder strings to None, so they count as missing.
    - When a field/decision is under its target, nulls surplus present values on
      records where the empty field stays consistent with `decision_reason`.
    - Runs in-place per generated batch; returns the same list for chaining.
    """
    fields = list(config.keys())

    # 1) Normalize placeholder-like strings / NaN to None so they count as missing.
    for record in records:
        for field in fields:
            value = record.get(field)
            if value is not None and _is_effectively_missing(value):
                record[field] = None

    true_recs = [r for r in records if bool(r["decision"]) is True]
    false_recs = [r for r in records if bool(r["decision"]) is False]

    # 2) Pad under-target fields, decision group by decision group.
    for decision_key, group in (("true", true_recs), ("false", false_recs)):
        if not group:
            continue
        for field in fields:
            rate = config[field][decision_key]
            if rate <= 0:
                continue
            target = round(len(group) * rate)
            already_missing = sum(1 for r in group if r.get(field) is None)
            shortfall = max(0, target - already_missing)
            if shortfall == 0:
                continue
            for record in _pad_candidates(group, decision_key, field):
                if shortfall == 0:
                    break
                if record.get(field) is None:
                    continue
                _null_field(record, field, _override_reason(decision_key, field, record))
                shortfall -= 1

    return records


def print_missing_qa(records: List[dict], config: dict = MISSING_RATE_CONFIG) -> None:
    """Print target vs realized missing-value % per field per decision."""
    if not records:
        print("No records to evaluate for missing-value QA.")
        return
    df = pd.DataFrame(records)
    fields = list(config.keys())
    print("\n=== Missing-value QA (target % vs realized %) ===")
    print(f"{'field':<16}{'decision':<8}{'target%':>9}{'realized%':>10}  status")
    print("-" * 52)
    for field in fields:
        for decision in ("true", "false"):
            mask = df["decision"].astype(bool) == (decision == "true")
            n = int(mask.sum())
            if n == 0:
                continue
            target_count = round(n * config[field][decision])
            target = (target_count / n) * 100
            realized = float(df.loc[mask, field].isna().mean() * 100)
            tolerance = max(1.0, 0.5 * 100.0 / n)  # half a record + rounding slack
            diff = realized - target
            if diff > tolerance:
                status = "OVER"
            elif diff < -tolerance:
                status = "UNDER"
            else:
                status = "OK"
            print(f"{field:<16}{decision:<8}{target:>8.1f}%{realized:>9.1f}%  {status}")
    print("=" * 52)


def _normalize_token(token: str) -> str:
    """Lowercase and strip diacritics/punctuation while keeping letters of any script."""
    token = unicodedata.normalize("NFD", token)
    token = "".join(c for c in token if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", token).lower()


def _normalized_tokens(name: str) -> set:
    """Loose word-token set from a name (any script, diacritics/punctuation-stripped)."""
    tokens = {_normalize_token(t) for t in re.split(r"[\s,.;:|()~\-–—/\\]+", name)}
    return {t for t in tokens if t}


def print_matching_text_qa(records: List[dict]) -> None:
    """Warn when `matching_text` tokens cannot be traced to either source name.

    Warning-only: transliteration / homoglyph / multi-script matches can
    legitimately differ at string level, so misses are surfaced, never fatal.
    """
    if not records:
        print("No records to evaluate for matching-text QA.")
        return

    issues = 0
    for idx, record in enumerate(records, start=1):
        name_tokens = _normalized_tokens(record.get("client_name") or "")
        name_tokens |= _normalized_tokens(record.get("hit_name") or "")
        matching_text = record.get("matching_text")
        if not matching_text or not str(matching_text).strip():
            print(f"Row {idx}: matching_text is empty "
                  f"({record.get('client_name')!r} vs {record.get('hit_name')!r}).")
            issues += 1
            continue
        for token in _normalized_tokens(str(matching_text)):
            if token not in name_tokens:
                print(f"Row {idx}: token {token!r} from matching_text {str(matching_text)!r} not found in "
                      f"client {record.get('client_name')!r} or hit {record.get('hit_name')!r}.")
                issues += 1
    if issues == 0:
        print("Matching-text QA: all matching_text tokens traced to client/hit names across all rows.")
    else:
        print(f"Matching-text QA complete: {issues} token(s) untraced (warnings only).")


class BatchGenerationError(RuntimeError):
    """Raised when a batch cannot be produced in a schema-valid form after all repairs/retries."""


def _extract_json_payload(text: str) -> Optional[str]:
    """Extract the JSON object or array from a raw model completion.

    Handles markdown fences, reasoning preamble, and trailing prose by scanning
    balanced braces/brackets rather than trusting the full string to be JSON.
    """
    if not text:
        return None
    candidate = text.strip()
    start = candidate.find("{")
    array_start = candidate.find("[")
    if start != -1 and (array_start == -1 or start < array_start):
        open_ch, close_ch = "{", "}"
        idx = start
    elif array_start != -1:
        open_ch, close_ch = "[", "]"
        idx = array_start
    else:
        return None
    depth = 0
    in_str = False
    i = idx
    while i < len(candidate):
        ch = candidate[i]
        if in_str:
            if ch == "\\" and i + 1 < len(candidate):
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        if depth == 0:
            return candidate[idx : i + 1]
        i += 1
    return None


def _coerce_bool(value) -> Optional[bool]:
    """Coerce a model decision value to bool, or None if unparseable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


_TRUE_REASONS = {
    "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING",
}
_FALSE_REASONS = {"NAME_MISMATCH", "DOB_MISMATCH_OR_INVALID", "GEOGRAPHIC_MISMATCH"}


def _repair_alert(raw: dict) -> dict:
    """Coerce one raw model alert into a clean dict that Pydantic strictly accepts."""
    # Only keep keys the schema knows, so extra='forbid' never fails.
    known = {
        "client_name", "hit_name", "matching_text",
        "client_dob", "hit_dob", "client_country", "client_city",
        "hit_country", "hit_city", "decision", "decision_reason", "thinking",
    }
    cleaned = {k: raw[k] for k in raw if k in known}

    for field in ("client_name", "hit_name", "matching_text", "client_dob", "hit_dob",
                  "client_country", "client_city", "hit_country", "hit_city", "thinking"):
        if cleaned.get(field) is None:
            cleaned[field] = ""
        elif not isinstance(cleaned[field], str):
            cleaned[field] = str(cleaned[field])

    # decision: coerce; default from reason when missing.
    decision = _coerce_bool(cleaned.get("decision"))
    if decision is None:
        reason = str(cleaned.get("decision_reason") or "")
        decision = reason in _TRUE_REASONS and reason not in _FALSE_REASONS
    cleaned["decision"] = decision

    # decision_reason: default / fix per decision.
    reason = str(cleaned.get("decision_reason") or "").strip()
    if not reason:
        reason = "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH" if decision else "NAME_MISMATCH"
    if decision and reason not in _TRUE_REASONS:
        reason = "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH"
    if not decision and reason not in _FALSE_REASONS:
        reason = "NAME_MISMATCH"
    cleaned["decision_reason"] = reason

    # matching_text: synthesize from the first shared token if the model left it blank.
    mt = str(cleaned.get("matching_text") or "").strip()
    if not mt:
        client_tokens = set(_normalized_tokens(cleaned.get("client_name") or ""))
        hit_tokens = set(_normalized_tokens(cleaned.get("hit_name") or ""))
        shared = client_tokens & hit_tokens
        if shared:
            mt = " | ".join(f"{t} ~ {t.upper()}" for t in sorted(shared)[:2])
    cleaned["matching_text"] = mt

    if not cleaned.get("thinking"):
        cleaned["thinking"] = "Auto-repaired record: generated by screening-engine synthesis."

    return cleaned


def _repair_alerts(raw_payload: object) -> List[dict]:
    """Coerce a raw model payload (dict/{"alerts":[...]}/[...]) into clean alert dicts."""
    if not isinstance(raw_payload, dict):
        raw_alerts = raw_payload
    else:
        if isinstance(raw_payload.get("alerts"), list):
            raw_alerts = raw_payload["alerts"]
        else:
            list_values = [v for v in raw_payload.values() if isinstance(v, list)]
            if list_values:
                raw_alerts = list_values[0]
            else:
                raw_alerts = [raw_payload]
    if isinstance(raw_alerts, dict):
        raw_alerts = [raw_alerts]
    if not isinstance(raw_alerts, list):
        raise ValueError(f"Expected a list/dict alert payload, got {type(raw_alerts).__name__}.")
    return [_repair_alert(item) for item in raw_alerts if isinstance(item, dict)]
def _validate_alerts(cleaned: List[dict]) -> List[dict]:
    """Strictly validate cleaned alert dicts via Pydantic; raises on failure."""
    if not cleaned:
        raise ValueError("No valid alert objects recovered from model output.")
    batch = AlertDatasetBatch.model_validate({"alerts": cleaned})
    return [alert.model_dump() for alert in batch.alerts]


def _synthesize_missing_text_from_error(error: BaseException) -> str:
    """Build a compact, actionable repair hint string from a Pydantic validation error."""
    lines = []
    if hasattr(error, "errors"):
        for err in error.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = err.get("msg", "")
            lines.append(f"- {loc}: {msg}")
    return "\n".join(lines[:12])


def _invoke_batch_with_repair(llm, prompt: str, max_retries: int = 1) -> List[dict]:
    """Run the model with structured-output; on parse failure, repair/validate the raw text.

    Preferred path: `with_structured_output(schema, include_raw=True)` so a parse
    miss puts a `raw` message + `parsing_error` in the result instead of raising.
    Falls back to plain `.invoke()` for providers/methods whose servers reject the
    structured format request.
    """
    structured_llm = llm.with_structured_output(
        AlertDatasetBatch, include_raw=True, method="json_schema"
    )

    def _extract_raw(out) -> str:
        if isinstance(out, dict):
            raw = out.get("raw")
            try:
                return raw.content if hasattr(raw, "content") else str(raw or "")
            except Exception:
                return str(out) if "raw" not in out else str(raw)
        return str(out)

    last_error = None
    raw_text = ""
    for attempt in range(max_retries + 1):
        # 1st try: structured path (raw + parsing_error), never raising on parse.
        out = structured_llm.invoke(prompt)
        parsing_error = out.get("parsing_error") if isinstance(out, dict) else None
        raw_text = _extract_raw(out) or ""
        if parsing_error is None:
            parsed = out.get("parsed")
            if isinstance(parsed, AlertDatasetBatch):
                return [alert.model_dump() for alert in parsed.alerts]
        # Fall back to repair+validate on the raw completion text.
        payload_src = _extract_json_payload(raw_text)
        if payload_src is not None:
            try:
                repaired = _repair_alerts(json.loads(payload_src))
                return _validate_alerts(repaired)
            except Exception as repair_err:
                last_error = repair_err
        else:
            last_error = parsing_error or ValueError("No JSON payload found in model output.")
        # Self-correction retry: append the parse feedback and ask again.
        if attempt >= max_retries:
            break
        fix_hint = _synthesize_missing_text_from_error(last_error) or str(last_error)
        prompt += (
            "\n\n<repair_instructions>\n"
            "Your previous response failed schema validation. "
            f"Fix these specific issues and resend the complete valid JSON only:\n{fix_hint}\n"
            "- Use EXACTLY the schema keys, no extras.\n"
            "- Include exactly the requested number of alert objects, ALL fields populated"
            " (no missing `matching_text` or `thinking`).\n"
            "</repair_instructions>"
        )
    if not raw_text:
        last_error = last_error or ValueError("Model returned empty output.")
    raise BatchGenerationError(f"Batch generation failed after repairs: {last_error}")


def generate_batch(
    batch_size: int,
    target_true_ratio: float,
    missing_config: dict = MISSING_RATE_CONFIG,
) -> List[dict]:
    batch_true_count = round(batch_size * target_true_ratio)
    batch_false_count = batch_size - batch_true_count
    missing_quotas_section = _build_missing_quotas(batch_true_count, batch_false_count, missing_config)

    prompt = f"""
<system>
You are an expert compliance client alert review analyst and data synthesis specialist. Generate a realistic, high-fidelity synthetic dataset of exactly {batch_size} alerts originating from a financial crime screening engine (Sanctions, PEPs, RCAs). 

This dataset will be used to fine-tune and train a pretrained LLM. Therefore, the data distribution, global demographic representativeness, formatting chaos, and step-by-step reasoning must perfectly mirror the cognitive process and edge cases handled by a real-world human analyst.

CRITICAL COMPLIANCE DIRECTIVES:
1. **Human-Only Focus:** All `client_*` and `hit_*` entities are strictly human individuals (no corporate entities, shell companies, or trusts).
2. **Avoid Type II Errors (False Negatives):** Legitimate identity variations (such as cultural name inversions, maiden/married names, cross-lingual aliases, dropped prefixes, and diacritic normalization) are **valid matches** that require human investigation, not automatic dismissals. False name mismatches (`NAME_MISMATCH`) must *only* be triggered when comparing genuinely distinct individuals who share accidental commonalities.
3. **EU AI Act Article 10 Alignment:** Ensure the dataset is globally representative across diverse cultures, linguistic backgrounds, scripts, and naming customs to prevent proxy bias or demographic discrimination.

For this specific batch, aim for approximately {batch_true_count} TRUE alerts and {batch_false_count} FALSE alerts.
</system>

<persona_context>
- `client_*` fields represent human retail banking customers and wealth management clients globally, reflecting diverse international demographics, age groups, and socioeconomic backgrounds.
- `hit_*` fields represent human individuals found on high-risk watchlists: Politically Exposed Persons (PEPs), sanctioned individuals, or financial criminals, featuring diverse global origins and titles (e.g., political, military, or religious honorifics where culturally appropriate) without skewing risk solely to specific regions.
</persona_context>

<naming_convention>
- `client_name` and `hit_name` must reflect realistic global naming customs for individuals (e.g., Western last/first structures, patronymics, matronymics, mononyms, multi-part surnames, and localized name orders). Do not force all human names into a rigid Western template.
- Go crazy with name formats across records: mix native scripts and diacritics, uppercase/lowercase/mixed-case, title prefixes, extra or missing middle components, and inconsistent spacing, punctuation, and capitalization.
- Ensure true alerts leverage realistic variations (diacritic normalization, standard initial expansions, common aliases, or cultural structures) without failing name logic, while false `NAME_MISMATCH` alerts leverage the specific distinct-person collision families below.
</naming_convention>

<dob_logic>
- Since the DOB difference of < 1 year is acceptable for true alerts, dates do not need to be identical always. Minor offsets, day/month swaps, or formatting differences resulting in a delta under 1 year are valid for true alerts.
- Only a clear major disparity (\\ge 1 year difference) triggers a DOB mismatch failure (`decision: false`, `decision_reason: "DOB_MISMATCH_OR_INVALID"`).
- Completely missing, invalid, or unparsable values gracefully bypass DOB checks to proceed down the cascade.
- Go crazy with DOB formats for BOTH `client_dob` and `hit_dob`. Vary representation across records: ISO (`YYYY-MM-DD`), day-first (`DD-MM-YYYY`), US (`MM/DD/YYYY`), dotted (`DD.MM.YYYY`), slash/two-digit years (`15/04/82`), written English (`12 April 1982`, `Apr 12, 1982`), ordinal forms (`12th April 1982`), month-year or year-only partials (`April 1982`, `1982`), age-style values (`43 yrs`), and `circa`/approximate markers (`circa 1982`).
- Keep the format chaos decision-safe: the same calendar date expressed in different layouts must still resolve correctly (matching date in different formats -> true-compatible; real year gaps >= 1 year -> `DOB_MISMATCH_OR_INVALID`; unparsable/partial -> graceful bypass).
</dob_logic>

<city_country_logic>
- `client_country`/`client_city` and `hit_country`/`hit_city` capture the geography associated with the human client and the screening hit. A location can be expressed at country and/or city granularity.
- Go crazy with city/country formats for BOTH client and hit geography. Vary representation across records: native/local scripts and diacritics (e.g., `Москва` vs `Moscow`), endonyms vs exonyms (`München` vs `Munich`), country aliases (`USA` vs `United States` vs `U.S.A.`), historical or alternate city names (`Mumbai` vs `Bombay`), ISO-style codes, mixed case, and city-plus-region/country suffixes (`Paris, FR`).
- Keep the format chaos decision-safe: the same place expressed in different spellings or layouts is still a geographic MATCH; a genuinely different country or city -> `decision: false`, `decision_reason: "GEOGRAPHIC_MISMATCH"`. Missing or invalid values bypass geography per Step 2 of `<cascading_logic>` (note it briefly in `thinking` and proceed, do not fail).
</city_country_logic>

<screening_engine_match_logic>
- Every alert (TRUE and FALSE) MUST populate `matching_text`: the exact word(s)/token(s) that the screening engine matched when it fired the alert.
- The engine does NOT know the verdict when it fires: it tokenizes and normalizes both names and stores the evidence that tripped a match.
- `matching_text` format: a compact token-pair list, one pair per fired token, written as `"<client_token> ~ <hit_token>"`, joined by ` | ` when the engine over-fires on several tokens (e.g. `"Garcia ~ GARCIA | Lopez ~ LOPEZ"`).
- The tokens MUST be traceable to the names: the client-side token appears in `client_name` and the hit-side token appears in `hit_name`.
- Even when later adjudicated FALSE (e.g. `NAME_MISMATCH`), `matching_text` must still state what fired the engine, since the analyst reviews it after the fact.
</screening_engine_match_logic>

<false_name_match_scenarios>
*Note: Unsafe evasion/normalization families (like maiden names, nicknames, prefix-stripping, and CJK inversions) have been intentionally excluded to prevent training the model to clear true risks.*

CRITICAL GENERATION RULE: When creating a FALSE alert with the reason `NAME_MISMATCH`, you MUST physically generate `client_name` and `hit_name` to be visibly distinct strings. You must inject a distinguishing token (e.g., a different middle name, an added surname, or altered spelling) into one of the names. THEY CANNOT BE IDENTICAL. 

The engine fires on fuzzy token similarity. Use **only** these approved false-positive families for records resulting in `NAME_MISMATCH`:

Family F1 - Common-name partial/middle mismatch: The engine over-matches on common first and last names, but there is a clear distinguishing component physically present in the strings, such as completely different middle names (e.g., "John David Smith" vs "John Michael Smith"). 
Family F2 - Phonetic / spelling variants of distinct people: Names spelled similarly but clearly belonging to completely different individuals with different token counts or structures.
Family F3 - Transliteration across scripts for separate individuals: Cross-script name collisions where the underlying entities are distinct and visibly different in length or composition.
Family F4 - Token-order permutation of separate individuals: An order-insensitive index fires on swapped name components, resulting in a culturally invalid or clearly distinct name structure.
Family F5 - Partial / compound-name substring over-fire: A shared surname fragment trips the engine on an unrelated compound-surname individual (e.g., "Carlos Ruiz" vs "Carlos Ruiz-Zafón").
Family F6 - Generation markers (Father vs. Son): Jr/Sr/III suffixes match separate legal entities who share a base name.
</false_name_match_scenarios>

<true_name_match_scenarios>
*Note: Use these approved true-positive families to generate realistic variations of the same human entity resulting in an exact match or valid variation approval.*

Family T1 - Transliteration & Romanization: Cross-script mapping differences pointing to the same individual due to phonetic translation.
Family T2 - Component Omission / Addition: Missing middle names, dropped secondary maternal surnames, or added patronymics standard to the entity's culture.
Family T3 - Cultural Permutation & Inversion: Valid reordering based on local naming conventions, such as Asian Surname-Given inversion.
Family T4 - Typographical & Diacritic Noise: Minor spelling variations, stripped accents, or OCR errors falling within standard edit-distance thresholds.
</true_name_match_scenarios>

<cascading_logic>
When evaluating each record, you MUST generate the `thinking` field FIRST before determining the final decision or reason. Follow this exact sequence and document your evaluation in `thinking` using **1-2 short, crisp sentences**:
1. Step 1 (DOB): Evaluate `client_dob` and `hit_dob` per the `<dob_logic>` rules above. If acceptable (< 1 year diff or missing/invalid), proceed to Step 2. Missing DOB is NEVER a reason to fail an alert.
2. Step 2 (Geography): Check country/city fields. If missing or invalid, do not fail. Bypass geography, note it briefly in `thinking`, and proceed to Step 3. Missing geography is NEVER a reason to fail an alert.
3. Step 3 (Name): Check `client_name` vs `hit_name` accounting for formatting and strict token order. 
   - TRUE MATCH: If the names are an exact match (ignoring case), a standard initial expansion (e.g., "K. Sharma" -> "Kiran Sharma"), or utilize a valid identity variation (Family T1-T4), you MUST explicitly cite the true match family code in your `thinking`. Proceed to assign a true decision reason. **You CANNOT default to FALSE just because DOB/Geography are missing on an exact name match.**
   - FALSE MATCH: If and only if the names have visible, physical string discrepancies (different middle names, differing compound structures, invalid permutations) utilizing an approved false-match family, explicitly cite the false match family code in your `thinking` string. Proceed to assign decision: false, decision_reason: "NAME_MISMATCH".
   - ALWAYS populate `matching_text` with the token(s) that fired the engine — even on FALSE `NAME_MISMATCH` verdicts.
</cascading_logic>

{missing_quotas_section}

<constraints>
- Allowed True Reasons: "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING", "INDETERMINATE_DEFAULT_TRUE"
- Allowed False Reasons: "NAME_MISMATCH", "DOB_MISMATCH_OR_INVALID", "GEOGRAPHIC_MISMATCH"
- Demographics: Sample diversely across global human backgrounds over your generation lifecycle.
- Native Scripts & Data Quality: Include native characters/diacritics where appropriate and realistic dirty data.
- `matching_text` is required on EVERY row (never empty, never "N/A"); it may hold a single token-pair or several pairs joined by ` | `.
- Thinking Field Style: Keep `thinking` values punchy, concise, and professional (1-2 sentences max).
- ABSOLUTE FAIL-SAFE: If `client_name` and `hit_name` are identical strings (ignoring capitalization), the reason CANNOT BE `NAME_MISMATCH`. If you choose `NAME_MISMATCH`, you are violating system instructions unless you have generated distinct text for the two names.
</constraints>
"""
    
    # Initialize the configured LangChain chat model with structured output enforcement
    if MODEL_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.5,
            top_p=0.9,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif MODEL_PROVIDER == "ollama":
        llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.5,
            top_p=0.9,
        )
    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER!r}. Expected 'gemini' or 'ollama'.")

    # Invoke the model; structured output with tolerant repair on parse failure
    return _invoke_batch_with_repair(llm, prompt)

def main():
    total_calls = math.ceil(TOTAL_ROWS / ROWS_PER_CALL)
    global_true_ratio = TOTAL_TRUE_ALERTS / TOTAL_ROWS
    master_dataset: List[dict] = []
    successful_calls = 0

    print(f"Starting batch generation: Target = {TOTAL_ROWS} rows (True: {TOTAL_TRUE_ALERTS}, False: {TOTAL_FALSE_ALERTS}) | Batch Size = {ROWS_PER_CALL} | Total Calls = {total_calls}")
    
    for current_call in range(1, total_calls + 1):
        remaining_needed = TOTAL_ROWS - len(master_dataset)
        current_batch_size = min(ROWS_PER_CALL, remaining_needed)
        
        if current_batch_size <= 0:
            break

        try:
            batch_records = generate_batch(current_batch_size, global_true_ratio)
            if not batch_records:
                raise ValueError("generate_batch returned an empty alert list.")
            batch_records = enforce_missing_rates(batch_records)
            master_dataset.extend(batch_records)
            successful_calls += 1
            completed_pct = (len(master_dataset) / TOTAL_ROWS) * 100
            print(f"Progress: {len(master_dataset)}/{TOTAL_ROWS} records ({completed_pct:.1f}%)")
        except Exception as e:
            print(f"Error on call {current_call}: {e}. Skipping iteration...")
            continue

    # Enforce exact global missing-rate targets on the compiled dataset, then QA.
    enforce_missing_rates(master_dataset)
    print_missing_qa(master_dataset)
    print_matching_text_qa(master_dataset)

    if not master_dataset:
        print(f"No records were generated (all {total_calls} batch calls failed).")
        print("Refusing to overwrite compliance_master_dataset.xlsx with an empty dataset.")
        return

    # Export to Excel
    output_filename = _build_output_filename()
    df = pd.DataFrame(master_dataset).sample(frac=1)
    df.to_excel(output_filename, index=False, engine='openpyxl')
        
    actual_true = sum(1 for r in master_dataset if r['decision'] is True)
    actual_false = sum(1 for r in master_dataset if r['decision'] is False)
    print(f"Generation complete. Compiled {len(master_dataset)} records (True: {actual_true}, False: {actual_false}) "
          f"into Excel file: {output_filename}. ({successful_calls}/{total_calls} batch calls succeeded, "
          f"{total_calls - successful_calls} skipped).")

if __name__ == "__main__":
    main()