import json
import math
import re
import unicodedata
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration Parameters ---
TOTAL_TRUE_ALERTS = 40 # Target total number of TRUE alerts required
TOTAL_FALSE_ALERTS = 100 # Target total number of FALSE alerts required
TOTAL_ROWS = TOTAL_TRUE_ALERTS + TOTAL_FALSE_ALERTS
ROWS_PER_CALL = 10      # Number of rows generated per LLM call
MODEL_NAME = "gemini-3.5-flash-lite"   # LangChain Gemini model name

# --- Missing-value configuration ---
# Target fraction of records where each field must be empty (None), split
# independently per decision outcome. Edit these values to customize the
# % missing values for city / country / dob on TRUE vs FALSE alerts.
# "client_*" = the client's data, "hit_*" = the screening hit's data.
MISSING_RATE_CONFIG = {
    "client_city":    {"true": 0.12, "false": 0.00},
    "client_country": {"true": 0.07, "false": 0.00},
    "hit_city":       {"true": 0.22, "false": 0.00},
    "hit_country":    {"true": 0.12, "false": 0.00},
    "client_dob":     {"true": 0.02, "false": 0.00},
    "hit_dob":        {"true": 0.03, "false": 0.00},
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


def _build_missing_quotas(batch_true_count: int, batch_false_count: int, config: dict) -> str:
    """Render per-batch empty-field quotas injected into the LLM prompt."""
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
        "    * TRUE + DOB field emptied -> prefer `EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH`.",
        "    * TRUE + city field emptied -> prefer `EXACT_NAME_MATCH_DOB_OK_CITY_MISSING`.",
        "    * FALSE alerts: never empty the field that drives the false reason",
        "      (keep DOBs populated on `DOB_MISMATCH_OR_INVALID`; keep country/city populated on `GEOGRAPHIC_MISMATCH`).",
        "- Briefly note every emptied field in `thinking` (e.g. `hit_city unavailable - bypassing geography`).",
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


def generate_batch(batch_size: int, target_true_ratio: float, model_name: str, missing_config: dict = MISSING_RATE_CONFIG) -> List[dict]:
    batch_true_count = round(batch_size * target_true_ratio)
    batch_false_count = batch_size - batch_true_count
    missing_quotas_section = _build_missing_quotas(batch_true_count, batch_false_count, missing_config)

    prompt = f"""
<system>
You are an expert compliance data engineer and data synthesis specialist. Generate a realistic, high-fidelity synthetic dataset of exactly {batch_size} alerts originating from a financial crime screening engine (Sanctions, PEPs, RCAs).
For this specific batch, aim for approximately {batch_true_count} TRUE alerts and {batch_false_count} FALSE alerts.
</system>

<naming_convention>
- `client_name` strictly follows the format: `Lastname, Middlename, Firstname` (or `Lastname, Firstname` if middle name is absent).
- `hit_name` may or may not follow this format.
- Go crazy with name matches for BOTH false and true alerts. Vary how `client_name` and `hit_name` relate to each other across records: exact repeats, token-order swaps, partial/fuzzy matches, initials or abbreviations, added/omitted middle names, maiden vs married surnames, aliases and known-as names, transliterations across scripts, hyphenated and multi-part surnames (e.g., "de la Cruz" vs "dela Cruz"), and near-miss typo variants that produce `NAME_MISMATCH` false alerts — alongside legitimate close variations that still count as TRUE matches.
- Also go crazy with formats: mix native scripts and diacritics, uppercase/lowercase/mixed-case, title prefixes (Mr., Dr., Sheikh, etc.), extra or missing middle components, and inconsistent spacing, punctuation, and capitalization across records.
</naming_convention>

<dob_logic>
- Since the DOB difference of < 1 year is acceptable for true alerts, dates do not need to be identical always. Minor offsets, day/month swaps, or formatting differences resulting in a delta under 1 year are valid for true alerts.
- Only a clear major disparity ($\\ge 1$ year difference) triggers a DOB mismatch failure (`decision: false`, `decision_reason: "DOB_MISMATCH_OR_INVALID"`).
- Completely missing, invalid, or unparsable values gracefully bypass DOB checks to proceed down the cascade.
- Go crazy with DOB formats for BOTH `client_dob` and `hit_dob`. Vary representation across records: ISO (`YYYY-MM-DD`), day-first (`DD-MM-YYYY`), US (`MM/DD/YYYY`), dotted (`DD.MM.YYYY`), slash/two-digit years (`15/04/82`), written English (`12 April 1982`, `Apr 12, 1982`), ordinal forms (`12th April 1982`), month-year or year-only partials (`April 1982`, `1982`), age-style values (`43 yrs`), and `circa`/approximate markers (`circa 1982`).
- Keep the format chaos decision-safe: the same calendar date expressed in different layouts must still resolve correctly (matching date in different formats -> true-compatible; real year gaps >= 1 year -> `DOB_MISMATCH_OR_INVALID`; unparsable/partial -> graceful bypass).
</dob_logic>

<city_country_logic>
- `client_country`/`client_city` and `hit_country`/`hit_city` capture the geography associated with the client and the screening hit. A location can be expressed at country and/or city granularity.
- Go crazy with city/country formats for BOTH client and hit geography. Vary representation across records: native/local scripts and diacritics (e.g., `Москва` vs `Moscow`), endonyms vs exonyms (`München` vs `Munich`), country aliases (`USA` vs `United States` vs `U.S.A.`), historical or alternate city names (`Mumbai` vs `Bombay`, `Saint Petersburg` vs `St. Petersburg`), ISO-style codes, mixed case and inconsistent capitalization, and city-plus-region/country suffixes (`Paris, FR`, `Dubai, AE`).
- Keep the format chaos decision-safe: the same place expressed in different spellings or layouts is still a geographic MATCH; a genuinely different country or city -> `decision: false`, `decision_reason: "GEOGRAPHIC_MISMATCH"`. Missing or invalid values bypass geography per Step 2 of `<cascading_logic>` (note it briefly in `thinking` and proceed, do not fail).
</city_country_logic>

<screening_engine_match_logic>
- Every alert (TRUE and FALSE) MUST populate `matching_text`: the exact word(s)/token(s) that the screening engine matched when it fired the alert.
- The engine does NOT know the verdict when it fires: it tokenizes and normalizes both names (strip titles/initials where applicable, split on spaces/commas/hyphens, lower-case, tolerate diacritics and phonetic variants) and stores the evidence that tripped a match.
- `matching_text` format: a compact token-pair list, one pair per fired token, written as `"<client_token> ~ <hit_token>"`, joined by ` | ` when the engine over-fires on several tokens (e.g. `"Garcia ~ GARCIA | Lopez ~ LOPEZ"`).
- The tokens MUST be traceable to the names: the client-side token appears in `client_name` and the hit-side token appears in `hit_name` (allowing for case/diacritic/small-spelling variants and transliterations across scripts).
- Even when later adjudicated FALSE (e.g. `NAME_MISMATCH`), `matching_text` must still state what fired the engine, since the analyst reviews it after the fact.
</screening_engine_match_logic>

<false_name_match_scenarios>
The engine fires on fuzzy token similarity, so FALSE name matches mirror the exact false-positive families a real screening engine produces. Study these families and sample across ALL of them over the generation lifecycle:

Family A - Common-name over-match:
- client `Smith, Jane Adelaide` / hit `SMITH, Robert` / matching_text `"Smith ~ SMITH"` - shared extremely common surname, different people -> `NAME_MISMATCH`
- client `Mohammed, Ali Hassan` / hit `ALI, Batyr` / matching_text `"Ali ~ ALI"` - single common given name over-fires
- client `Ho, Wei Ming` / hit `HO CHI MINH` / matching_text `"Ho ~ HO"` - two-letter surname over-fires on a famous namesake

Family B - Phonetic / spelling variants:
- client `Smyth, James` / hit `SMITH, James` / matching_text `"Smyth ~ SMITH"` - spelled differently, pronounced alike, unrelated people
- client `Yusuf, Daud` / hit `YOUSEF, Daoud` / matching_text `"Yusuf ~ YOUSEF"` - transliteration variants of a common Arabic name
- client `Müller, Hans` / hit `MUELLER, Josef` / matching_text `"Müller ~ MUELLER"` - umlaut-expanded surname, different people

Family C - Transliteration across scripts:
- client `Usama, Rashid` / hit `USSAMA, Osama` / matching_text `"Usama ~ USSAMA"` - romanized Arabic variants
- client `Petrov, Dmitri` / hit `ДМИТРИЙ ПЕТРОВ` / matching_text `"Petrov ~ ПЕТРОВ"` - Latin vs Cyrillic forms
- client `Wong, Li Wei` / hit `WONG, Wei` / matching_text `"Wong ~ WONG | Wei ~ Wei"` - romanized CJK surname+given over-fire

Family D - Diacritics / accent stripped:
- client `García, Ana María` / hit `GARCIA, Luis` / matching_text `"García ~ GARCIA"` - accent dropped at match time
- client `Weiß, Kurt` / hit `WEISS, Petra` / matching_text `"Weiß ~ WEISS"` - sharp-s vs ss

Family E - Abbreviations / initials expanded:
- client `Smith, J. R.` / hit `SMITH, JOHN ROBERT` / matching_text `"Smith, J. R. ~ SMITH, JOHN ROBERT"` - surname + initials match the expanded hit
- client `de la Torre, F.` / hit `TORRE, FERNANDO` / matching_text `"de la Torre, F. ~ TORRE, FERNANDO"` - particle + initial over-matches

Family F - Token-order permutation:
- client `Haddad, Karim` / hit `KARIM HADDAD` / matching_text `"Haddad, Karim ~ KARIM HADDAD"` - order-insensitive name index fires
- client `Garcia Lopez, Maria` / hit `LOPEZ GARCIA, Carlos` / matching_text `"Garcia ~ GARCIA | Lopez ~ LOPEZ"` - both surnames shared in reordered form

Family G - Partial / compound-name substring:
- client `Ahmadinejad, Reza` / hit `AHMAD, Reza` / matching_text `"Ahmad ~ AHMAD"` - leading substring of a longer surname trips the engine
- client `Cruz-Ramirez, Elena` / hit `RAMIREZ, Jorge` / matching_text `"Ramirez ~ RAMIREZ"` - hyphenated compound shares one surname token

Family H - Maiden / married / apostrophe-split surnames:
- client `Lopez, Rosa` (maiden Garcia-Lopez) / hit `GARCIA, Rosa` / matching_text `"Rosa ~ ROSA"` - shared wrapper given name + linked surnames
- client `O'Brien, Siobhan` / hit `BRIEN, Michael` / matching_text `"Brien ~ BRIEN"` - apostrophe-split surname over-fires

Family I - Nicknames & cross-lingual given-name equivalences:
- client `Alejandro, Miguel` / hit `ALEX, Miguel` / matching_text `"Miguel ~ MIGUEL"` - same given name, different people
- client `John, Robert` / hit `IVAN, Robert` / matching_text `"Robert ~ ROBERT"` - cross-lingual synonyms (John/Ivan) are NOT matches but the shared second name fires

Family J - Asian name-order / structure confusion:
- client `Kim, Jong Il` / hit `JONG IL KIM` / matching_text `"Kim, Jong Il ~ JONG IL KIM"` - the engine reorders tokens and normalizes structure
- client `Nguyen, Van Anh` / hit `ANH, NGUYEN` / matching_text `"Nguyen ~ NGUYEN | Van Anh ~ ANH"` - swapped surname/given pairs over-fire

Family K - Generation markers stripped:
- client `Ford, Harrison, Jr.` / hit `FORD, HARRISON, III` / matching_text `"Ford, Harrison ~ FORD, HARRISON"` - Jr/III stripped, still a different person

Family L - Articles / connectives / prefixes normalized away:
- client `Al-Assad, Bashar` / hit `ASSAD, Batyr` / matching_text `"Al-Assad ~ ASSAD"` - Al-/El-/bin/ibn prefixes removed at match time
- client `van der Berg, Jan` / hit `BERG, Klaus` / matching_text `"Berg ~ BERG"` - `van der` connective dropped
</false_name_match_scenarios>

<cascading_logic>
When evaluating each record, follow this exact sequence and document your evaluation in the `thinking` field in **1-2 short, crisp sentences**:
1. Step 1 (DOB): Evaluate `client_dob` and `hit_dob` per the `<dob_logic>` rules above. If acceptable (< 1 year diff or missing/invalid), proceed to Step 2.
2. Step 2 (Geography): Check country/city fields. If missing or invalid, do not fail. Bypass geography, note it briefly in `thinking`, and proceed to Step 3.
3. Step 3 (Name): Check `client_name` vs `hit_name` accounting for formatting and strict token order. 
   - Exact structured match -> use available valid elements for a true decision with code (`EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH`, `EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH`, or `EXACT_NAME_MATCH_DOB_OK_CITY_MISSING`).
   - Mismatch or reverse order -> decision: false, decision_reason: "NAME_MISMATCH".
   - ALWAYS populate `matching_text` with the token(s) that fired the engine — even on FALSE `NAME_MISMATCH` verdicts.
</cascading_logic>

{missing_quotas_section}

<constraints>
- Allowed True Reasons: "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING"
- Allowed False Reasons: "NAME_MISMATCH", "DOB_MISMATCH_OR_INVALID", "GEOGRAPHIC_MISMATCH"
- Demographics: Sample diversely across global backgrounds over your generation lifecycle.
- Native Scripts & Data Quality: Include native characters/diacritics where appropriate and realistic dirty data.
- `matching_text` is required on EVERY row (never empty, never "N/A"); it may hold a single token-pair or several pairs joined by ` | `.
- Thinking Field Style: Keep `thinking` values punchy, concise, and professional (1-2 sentences max).
</constraints>
"""
    
    # Initialize the LangChain Google GenAI model with structured output enforcement
    llm = ChatGoogleGenerativeAI(
    model=model_name, 
    temperature=0.7, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    structured_llm = llm.with_structured_output(AlertDatasetBatch)
    
    # Invoke the model directly; LangChain handles parsing into the Pydantic object
    validated_batch = structured_llm.invoke(prompt)
    
    return [alert.model_dump() for alert in validated_batch.alerts]

def main():
    total_calls = math.ceil(TOTAL_ROWS / ROWS_PER_CALL)
    global_true_ratio = TOTAL_TRUE_ALERTS / TOTAL_ROWS
    master_dataset: List[dict] = []
    
    print(f"Starting batch generation: Target = {TOTAL_ROWS} rows (True: {TOTAL_TRUE_ALERTS}, False: {TOTAL_FALSE_ALERTS}) | Batch Size = {ROWS_PER_CALL} | Total Calls = {total_calls}")
    
    for current_call in range(1, total_calls + 1):
        remaining_needed = TOTAL_ROWS - len(master_dataset)
        current_batch_size = min(ROWS_PER_CALL, remaining_needed)
        
        if current_batch_size <= 0:
            break

        try:
            batch_records = generate_batch(current_batch_size, global_true_ratio, MODEL_NAME)
            batch_records = enforce_missing_rates(batch_records)
            master_dataset.extend(batch_records)
            completed_pct = (len(master_dataset) / TOTAL_ROWS) * 100
            print(f"Progress: {len(master_dataset)}/{TOTAL_ROWS} records ({completed_pct:.1f}%)")
        except Exception as e:
            print(f"Error on call {current_call}: {e}. Retrying iteration...")
            continue

    # Enforce exact global missing-rate targets on the compiled dataset, then QA.
    enforce_missing_rates(master_dataset)
    print_missing_qa(master_dataset)
    print_matching_text_qa(master_dataset)

    # Export to Excel
    output_filename = "compliance_master_dataset.xlsx"
    df = pd.DataFrame(master_dataset).sample(frac=1)
    df.to_excel(output_filename, index=False, engine='openpyxl')
        
    actual_true = sum(1 for r in master_dataset if r['decision'] is True)
    actual_false = sum(1 for r in master_dataset if r['decision'] is False)
    print(f"Generation complete. Compiled {len(master_dataset)} records (True: {actual_true}, False: {actual_false}) into Excel file: {output_filename}.")

if __name__ == "__main__":
    main()