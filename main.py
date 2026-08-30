import json
import math
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import ollama

# --- Configuration Parameters ---
TOTAL_TRUE_ALERTS = 60 # Target total number of TRUE alerts required
TOTAL_FALSE_ALERTS = 40  # Target total number of FALSE alerts required
TOTAL_ROWS = TOTAL_TRUE_ALERTS + TOTAL_FALSE_ALERTS
ROWS_PER_CALL = 10      # Number of rows generated per LLM call
MODEL_NAME = "gemma4:31b-cloud"   # Local Ollama model name

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

def generate_batch(batch_size: int, target_true_ratio: float, model_name: str) -> List[dict]:
    batch_true_count = round(batch_size * target_true_ratio)
    batch_false_count = batch_size - batch_true_count

    prompt = f"""
<system>
You are an expert compliance data engineer and data synthesis specialist. Generate a realistic, high-fidelity synthetic dataset of exactly {batch_size} alerts originating from a financial crime screening engine (Sanctions, PEPs, RCAs).
For this specific batch, aim for approximately {batch_true_count} TRUE alerts and {batch_false_count} FALSE alerts.
</system>

<output_format>
Return a JSON object containing an "alerts" array with precisely {batch_size} objects adhering strictly to the schema. Do not include any markdown wrappers other than standard JSON formatting, and do not include any introductory or concluding conversational filler.
</output_format>

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

<cascading_logic>
When evaluating each record, follow this exact sequence and document your evaluation in the `thinking` field in **1-2 short, crisp sentences**:
1. Step 1 (DOB): Evaluate `client_dob` and `hit_dob` per the `<dob_logic>` rules above. If acceptable (< 1 year diff or missing/invalid), proceed to Step 2.
2. Step 2 (Geography): Check country/city fields. If missing or invalid, do not fail. Bypass geography, note it briefly in `thinking`, and proceed to Step 3.
3. Step 3 (Name): Check `client_name` vs `hit_name` accounting for formatting and strict token order. 
   - Exact structured match -> use available valid elements for a true decision with code (`EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH`, `EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH`, or `EXACT_NAME_MATCH_DOB_OK_CITY_MISSING`).
   - Mismatch or reverse order -> decision: false, decision_reason: "NAME_MISMATCH".
</cascading_logic>

<constraints>
- Allowed True Reasons: "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING"
- Allowed False Reasons: "NAME_MISMATCH", "DOB_MISMATCH_OR_INVALID", "GEOGRAPHIC_MISMATCH"
- Demographics: Sample diversely across global backgrounds over your generation lifecycle.
- Native Scripts & Data Quality: Include native characters/diacritics where appropriate and realistic dirty data.
- Thinking Field Style: Keep `thinking` values punchy, concise, and professional (1-2 sentences max).
</constraints>

"""
    
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=AlertDatasetBatch.model_json_schema(),
        options={"temperature": 0.7}
    )
    
    parsed_data = json.loads(response['message']['content'])
    validated_batch = AlertDatasetBatch.model_validate(parsed_data)
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
            master_dataset.extend(batch_records)
            completed_pct = (len(master_dataset) / TOTAL_ROWS) * 100
            print(f"Progress: {len(master_dataset)}/{TOTAL_ROWS} records ({completed_pct:.1f}%)")
        except Exception as e:
            print(f"Error on call {current_call}: {e}. Retrying iteration...")
            continue

    # Export to Excel
    output_filename = "compliance_master_dataset.xlsx"
    df = pd.DataFrame(master_dataset)
    df.to_excel(output_filename, index=False, engine='openpyxl')
        
    actual_true = sum(1 for r in master_dataset if r['decision'] is True)
    actual_false = sum(1 for r in master_dataset if r['decision'] is False)
    print(f"Generation complete. Compiled {len(master_dataset)} records (True: {actual_true}, False: {actual_false}) into Excel file: {output_filename}.")

if __name__ == "__main__":
    main()