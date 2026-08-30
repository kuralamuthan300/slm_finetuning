import json
import math
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import ollama

# --- Configuration Parameters ---
TOTAL_TRUE_ALERTS = 600  # Target total number of TRUE alerts required
TOTAL_FALSE_ALERTS = 400  # Target total number of FALSE alerts required
TOTAL_ROWS = TOTAL_TRUE_ALERTS + TOTAL_FALSE_ALERTS
ROWS_PER_CALL = 10      # Number of rows generated per LLM call
MODEL_NAME = "deepseek-r1:1.5b"   # Local Ollama model name

# --- Enums & Schemas ---
ScreeningType = Literal["Sanctions", "Embargoes", "PEP", "RCA"]
HitType = Literal["embargo", "non-embargo"]
DecisionReason = Literal[
    "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH",
    "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING",
    "EMBARGO_COUNTRY_MATCH",
    "NAME_MISMATCH",
    "DOB_MISMATCH_OR_INVALID",
    "GEOGRAPHIC_MISMATCH",
    "EMBARGO_WILDCARD_HIT"
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
    screening_type: ScreeningType
    hit_type: HitType
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
You are an expert compliance data engineer and data synthesis specialist. Generate a realistic, high-fidelity synthetic dataset of exactly {batch_size} alerts originating from a financial crime screening engine (Sanctions, Embargoes, PEPs, RCAs).
For this specific batch, aim for approximately {batch_true_count} TRUE alerts and {batch_false_count} FALSE alerts.
</system>

<output_format>
Return a JSON object containing an "alerts" array with precisely {batch_size} objects adhering strictly to the schema. Do not include any markdown wrappers other than standard JSON formatting, and do not include any introductory or concluding conversational filler.
</output_format>

<naming_convention>
- `client_name` strictly follows the format: `Lastname, Middlename, Firstname` (or `Lastname, Firstname` if middle name is absent).
- `hit_name` may or may not follow this format.
- Strict Matching Rule: Token or order mismatches (e.g., comparing `alexander, kumar` vs `kumar, alexander`) must be flagged as a strict mismatch (`decision: false`, `decision_reason: "NAME_MISMATCH"`).
</naming_convention>

<dob_logic>
- Since the DOB difference of < 1 year is acceptable for true alerts, dates do not need to be identical always. Minor offsets, day/month swaps, or formatting differences resulting in a delta under 1 year are valid for true alerts.
- Only a clear major disparity ($\ge 1$ year difference) triggers a DOB mismatch failure (`decision: false`, `decision_reason: "DOB_MISMATCH_OR_INVALID"`).
- Completely missing, invalid, or unparsable values gracefully bypass DOB checks to proceed down the cascade.
</dob_logic>

<cascading_logic>
When evaluating each record, follow this exact sequence and document your evaluation in the `thinking` field in **1-2 short, crisp sentences**:
1. Embargo Check: If `hit_type` is "embargo" (where `hit_name` is "*"), check if `client_country` or `client_city` matches `hit_country`. 
   - Match -> decision: true, decision_reason: "EMBARGO_COUNTRY_MATCH".
   - No match/missing -> decision: false, decision_reason: "EMBARGO_WILDCARD_HIT". Skip name checks.
2. Step 1 (DOB): Evaluate `client_dob` and `hit_dob` per the `<dob_logic>` rules above. If acceptable (< 1 year diff or missing/invalid), proceed to Step 2.
3. Step 2 (Geography): Check country/city fields. If missing or invalid, do not fail. Bypass geography, note it briefly in `thinking`, and proceed to Step 3.
4. Step 3 (Name): Check `client_name` vs `hit_name` accounting for formatting and strict token order. 
   - Exact structured match -> use available valid elements for a true decision with code (`EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH`, `EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH`, or `EXACT_NAME_MATCH_DOB_OK_CITY_MISSING`).
   - Mismatch or reverse order -> decision: false, decision_reason: "NAME_MISMATCH".
</cascading_logic>

<constraints>
- Allowed True Reasons: "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_MISSING_COUNTRY_MATCH", "EXACT_NAME_MATCH_DOB_OK_CITY_MISSING", "EMBARGO_COUNTRY_MATCH"
- Allowed False Reasons: "NAME_MISMATCH", "DOB_MISMATCH_OR_INVALID", "GEOGRAPHIC_MISMATCH", "EMBARGO_WILDCARD_HIT"
- Demographics: Sample diversely across global backgrounds over your generation lifecycle.
- Native Scripts & Data Quality: Include native characters/diacritics where appropriate and realistic dirty data.
- Thinking Field Style: Keep `thinking` values punchy, concise, and professional (1-2 sentences max).
</constraints>

<few_shot_examples>
Example 1 (True Match with minor DOB variance < 1 year):
{{
  "client_name": "Moreau, Jean-Luc",
  "hit_name": "Moreau, Jean-Luc",
  "client_dob": "1982-04-12",
  "hit_dob": "1982-04-15",
  "client_country": "France",
  "client_city": "Paris",
  "hit_country": "France",
  "hit_city": "Paris",
  "screening_type": "Sanctions",
  "hit_type": "non-embargo",
  "decision": true,
  "decision_reason": "EXACT_NAME_MATCH_DOB_OK_COUNTRY_MATCH",
  "thinking": "Exact name match with acceptable DOB variance (< 1 year) and matching geography."
}}

Example 2 (Name Mismatch due to order):
{{
  "client_name": "Kumar, Alexander",
  "hit_name": "Alexander, Kumar",
  "client_dob": "1990-01-01",
  "hit_dob": "1990-01-01",
  "client_country": "India",
  "client_city": "Bengaluru",
  "hit_country": "India",
  "hit_city": "Bengaluru",
  "screening_type": "PEP",
  "hit_type": "non-embargo",
  "decision": false,
  "decision_reason": "NAME_MISMATCH",
  "thinking": "Token order mismatch between names (Kumar, Alexander vs Alexander, Kumar)."
}}
</few_shot_examples>
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