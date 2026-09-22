#!/usr/bin/env python3
"""
generate_synthetic_screening_data.py

High-performance synthetic screening alert data generator for Small Language Model (SLM) training.
Simulates individual name-matching alerts from an AML/Sanctions client screening engine using a
hybrid Programmatic (RapidFuzz) / Ollama structured generation architecture. The split is
configurable and defaults to 30% Programmatic / 70% Ollama (--fuzzy_percentage /
--ollama_percentage); the reserved exact-match and cross-gender categories are carved out of the
decision budgets, so the requested total is always produced exactly.

Output CSV Schema:
    client_name, hit_name, matching_text, disposition_code, disposition_label, thinking, decision, created_by
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

# ==============================================================================
# PYDANTIC SCHEMA & CONSTANTS
# ==============================================================================

class ScreeningAlert(BaseModel):
    matching_text: str = Field(description="The exact overlapping or matching name tokens")
    disposition_code: str = Field(description="Fixed vocabulary disposition code describing the name-level variation (e.g. DISP_MINOR_TYPO)")
    thinking: str = Field(description="Step-by-step human cognitive reasoning analyzing name similarities, typos, abbreviations, or conflicts without mathematical formulas")
    decision: str = Field(description="Screening resolution: 'escalate to analyst' or 'disqualify'")
    created_by: str = Field(default="rule_based", description="Data generation method: 'rule_based' (RapidFuzz) or 'ollama' (LLM-generated)")


DECISION_TP = "escalate to analyst"
DECISION_FP = "disqualify"

# ==============================================================================
# FIXED VOCABULARY DISPOSITION CODES
# All codes describe name-level phenomena only (structure, script, token order,
# or linguistic variation). No format, encoding, or transport artifacts.
# ==============================================================================

DISPOSITION_CODES: Dict[str, str] = {
    # True Positive codes — same individual, name recorded differently
    "DISP_MINOR_TYPO":                  "Single character typo or keyboard slip",
    "DISP_TOKEN_ORDER_SWAP":            "Name tokens in different sequence",
    "DISP_MIDDLE_NAME_ABBREVIATED":     "Middle name shortened to initial",
    "DISP_MIDDLE_INITIAL_EXPANDED":     "Middle initial expanded to full name",
    "DISP_MIDDLE_NAME_OMITTED":         "Middle name absent from one record",
    "DISP_TRANSLITERATION_VARIANT":     "Same name, different transliteration standard",
    "DISP_DIACRITIC_STRIPPED":          "Diacritics removed from name characters",
    "DISP_COMPOUND_NAME_RESTRUCTURED":  "Hyphenated name split or merged",
    "DISP_EXACT_NAME_MATCH":            "All name tokens match exactly, character for character",
    # False Positive codes — distinct individual
    "DISP_SURNAME_CONFLICT":            "Surnames are entirely different names",
    "DISP_MIDDLE_NAME_CONFLICT":        "Middle names are irreconcilably different",
    "DISP_PHONETIC_SURNAME_DIVERGENCE": "Surnames phonetically shifted but distinct",
    "DISP_GIVEN_NAME_GENDER_VARIANT":   "Given name differs only by gender inflection (e.g. Daniel/Daniela)",
}

# NOTE: DISPOSITION_FP_CODES must be declared BEFORE DISPOSITION_TP_CODES so that the
# TP list can be derived by exclusion. Adding a new False Positive code above only
# requires adding it here — it can never leak into the True Positive vocabulary.
DISPOSITION_FP_CODES = [
    "DISP_SURNAME_CONFLICT",
    "DISP_MIDDLE_NAME_CONFLICT",
    "DISP_PHONETIC_SURNAME_DIVERGENCE",
    "DISP_GIVEN_NAME_GENDER_VARIANT",
]
DISPOSITION_TP_CODES = [k for k in DISPOSITION_CODES if k not in set(DISPOSITION_FP_CODES)]

# Final CSV column order — shared by the in-memory frame, the incremental writer and the
# exported dataset, so the three can never disagree.
OUTPUT_COLUMNS: List[str] = [
    "client_name",
    "hit_name",
    "matching_text",
    "disposition_code",
    "disposition_label",
    "thinking",
    "decision",
    "created_by",
]

OLLAMA_SYSTEM_PROMPT = """You are a senior AML/Sanctions name-screening compliance analyst.
Your job is to review screening alerts by comparing a Client Name against a Candidate Hit Name exactly as a human brain would evaluate them.

COGNITIVE REVIEW PRINCIPLES:
1. Think like a human reviewer: evaluate first names, middle names, patronymics, surnames, and initials token by token.
2. ABSOLUTELY PROHIBITED: Do not mention mathematical metrics, algorithms, Levenshtein distances, similarity percentages, or token ratios.
3. Apply intuitive human name reasoning:
   - Identify clerical typos (e.g. keyboard slips, adjacent letter swaps like 'Sami' vs 'Saim').
   - Identify token order differences (e.g. family name first vs given name first in different jurisdiction databases).
   - Identify middle name abbreviations (e.g. 'Viktorovich' vs 'V.') or omissions.
   - Identify transliteration differences (e.g. 'Mohammed' vs 'Muhammad' vs 'Mukhammad').
   - Identify diacritic stripping (e.g. 'Müller' vs 'Muller', 'José' vs 'Jose').
   - Identify compound name splits or merges (e.g. 'Jean-Paul' vs 'Jean Paul').
   - Identify given names that differ only by a gendered inflection (e.g. 'Daniel' vs 'Daniela', 'Khalid' vs 'Khalida').
     These are two different personal names describing people of DIFFERENT genders — never a typo, an abbreviation or a transliteration,
     and the mismatch in recorded gender rules out the same natural person.
   - Identify fully identical names, where every token matches character for character with no variation of any kind.
   - Identify irreconcilably different surnames indicating two distinct persons.
   - Identify irreconcilably different middle names indicating two distinct persons.
4. Output raw JSON strictly adhering to the schema:
   - "matching_text": Exact overlapping name words or matched tokens between client and hit.
   - "disposition_code": EXACTLY one code from this controlled vocabulary:
       DISP_MINOR_TYPO               — Single character typo or keyboard slip
       DISP_TOKEN_ORDER_SWAP         — Name tokens in different sequence
       DISP_MIDDLE_NAME_ABBREVIATED  — Middle name shortened to initial
       DISP_MIDDLE_INITIAL_EXPANDED  — Middle initial expanded to full name
       DISP_MIDDLE_NAME_OMITTED      — Middle name absent from one record
       DISP_TRANSLITERATION_VARIANT  — Same name, different transliteration standard
       DISP_DIACRITIC_STRIPPED       — Diacritics removed from name characters
       DISP_COMPOUND_NAME_RESTRUCTURED — Hyphenated name split or merged
       DISP_EXACT_NAME_MATCH         — All name tokens match exactly, character for character
       DISP_SURNAME_CONFLICT         — Surnames are entirely different names
       DISP_MIDDLE_NAME_CONFLICT     — Middle names are irreconcilably different
       DISP_PHONETIC_SURNAME_DIVERGENCE — Surnames phonetically shifted but distinct
       DISP_GIVEN_NAME_GENDER_VARIANT — Given name differs only by gender inflection (e.g. Daniel/Daniela)
   - "thinking": Detailed step-by-step human cognitive reasoning explaining the name comparison.
   - "decision": Strictly 'escalate to analyst' (True Positive) or 'disqualify' (False Positive)."""

# Known non-individual indicators to filter out corporate entities
NON_INDIVIDUAL_KEYWORDS = {
    "INC", "INCORPORATED", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION",
    "CO", "COMPANY", "BANK", "FOUNDATION", "GROUP", "HOLDINGS", "ENTERPRISES",
    "SERVICES", "TRADING", "SA", "S.A.", "AG", "GMBH", "PLC", "B.V.", "BV",
    "MINISTRY", "DEPARTMENT", "AGENCY", "SOCIETY", "ASSOCIATION", "FUND",
    "TRUST", "COMMITTEE", "PARTY", "MOVEMENT", "BRIGADE", "FRONT", "ARMY"
}

# Rich Multi-Ethnic Global Name Pools
GLOBAL_NAME_POOLS: Dict[str, List[str]] = {
    "East Asian (Chinese, Japanese, Korean)": [
        "Wei Zhang", "Ming Chen", "Jun Wang", "Xiao Liu", "Mei Huang",
        "Jin Zhao", "Hiroshi Tanaka", "Kenji Sato", "Daiki Suzuki", "Sakura Takahashi",
        "Yuto Watanabe", "Min-jun Kim", "Seo-jun Lee", "Ji-woo Park", "Ha-eun Choi", "Sun-woo Jeong",
        "Li Wei Chen", "Katsumi Takahashi", "Hyun-woo Shin"
    ],
    "South Asian (Indian, Pakistani, Bangladeshi)": [
        "Aarav Sharma", "Rohan Patel", "Priya Verma", "Ananya Singh", "Rajesh Kumar",
        "Vikram Rao", "Suresh Gupta", "Sunita Chatterjee", "Amit Banerjee", "Pooja Mukherjee",
        "Deepak Nair", "Tariq Khan", "Imran Ahmed", "Zainab Ali", "Farooq Hussain",
        "Harpreet Kaur", "Prashant Kulkarni", "Lakshmi Sundaram"
    ],
    "Middle Eastern & North African (MENA)": [
        "Tariq Al-Mansoor", "Omar Al-Fassi", "Ziad Al-Hashimi", "Khalid Al-Sabah", "Karim El-Baz",
        "Nour Haddad", "Layla Kabbaj", "Mariam Trabelsi", "Fatima Qasim", "Yasmin Mansour",
        "Mostafa Al-Masri", "Hamza Bin Rashid", "Ibrahim Al-Ghamdi", "Samiha Oglu",
        "Samir Al-Khouri", "Bilal Benali", "Habib Bourguiba"
    ],
    "African (West, East, Southern African)": [
        "Adebayo Okafor", "Olumide Mensah", "Chidiebere Mwangi", "Kwame Kamau", "Kofi Osei",
        "Babatunde Diallo", "Tendai Traore", "Amara Keita", "Zola Balogun", "Nnamdi Nwachukwu",
        "Sipho Dlamini", "Thabo Mokoena", "Farai Achebe", "Jengo Toure",
        "Kagiso Rabada", "Sekou Toure", "Wanjiku Njoroge"
    ],
    "Slavic & Eastern European": [
        "Dmitry Ivanov", "Alexei Petrov", "Sergei Smirnov", "Mikhail Volkov", "Nikolai Kuznetsov",
        "Ivan Popov", "Vladimir Sokolov", "Elena Morozova", "Svetlana Pavlova", "Tatiana Kozlova",
        "Jan Kowalski", "Luka Horvat", "Milos Stankovic", "Dragan Novak",
        "Bogdan Popescu", "Andrzej Wisniewski", "Zofia Dabrowska"
    ],
    "European (Germanic, Romance, Nordic)": [
        "Alexander Mueller", "Thomas Schmidt", "Lucas Schneider", "Pierre Dubois", "Matteo Rossi",
        "Marco Ferrari", "Julian Moreau", "Sophie Laurent", "Erik Lindqvist", "Lars Hansen",
        "Sebastian Weber", "Claire Fontaine", "Andrea Bianchi", "Frederik Nielsen",
        "Jean-Paul Belmondo", "Henrik Vanger", "Astrid Lindgren"
    ],
    "Hispanic & Latin American": [
        "Carlos Rodriguez", "Miguel Angel Hernandez", "Mateo Garcia", "Santiago Martinez",
        "Alejandro Lopez", "Sofia Gonzalez", "Valentina Perez", "Camila Sanchez", "Isabella Ramirez",
        "Javier Torres", "Diego Flores", "Lucia Gomez", "Andres Castillo", "Maria Elena Morales",
        "Gabriel Marquez", "Juana De La Cruz", "Federico Silva"
    ]
}

ALTERNATIVE_MIDDLE_NAMES = [
    # Anglo-European
    "Alexander", "Robert", "James", "William", "David", "Edward", "Thomas",
    "Michael", "Joseph", "Charles", "Francis", "Raymond", "Anthony", "Patrick",
    "Lawrence", "Arthur", "George", "Pierre", "Jean", "Lucas",
    # MENA & Islamic
    "Hassan", "Omar", "Ali", "Hussein", "Ibrahim", "Tariq", "Khalid",
    "Rashid", "Hamza", "Zayn", "Nour", "Karim", "Ziad",
    # South Asian
    "Kumar", "Pratap", "Chandra", "Dev", "Mohan", "Nath", "Prakash",
    # East Asian
    "Wei", "Ming", "Jun", "Chen", "Hiro", "Ken", "Min", "Woo",
    # African
    "Kwame", "Kofi", "Babatunde", "Chidi", "Amara", "Tendai"
]

ALTERNATIVE_SURNAMES = [
    # Global multi-ethnic surnames
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
    "Rodriguez", "Hernandez", "Lopez", "Gonzalez", "Garcia", "Martinez", "Perez",
    "Wang", "Li", "Zhang", "Chen", "Liu", "Yang", "Huang", "Wu", "Zhou",
    "Tanaka", "Sato", "Suzuki", "Takahashi", "Watanabe", "Ito", "Nakamura",
    "Kim", "Lee", "Park", "Choi", "Jeong", "Kang", "Yoon",
    "Sharma", "Patel", "Verma", "Singh", "Kumar", "Rao", "Gupta", "Chatterjee",
    "Al-Mansoor", "Al-Hashimi", "Al-Sabah", "Al-Fassi", "Haddad", "Mansour", "Kabbaj",
    "Okafor", "Mensah", "Mwangi", "Kamau", "Osei", "Diallo", "Traore", "Balogun",
    "Ivanov", "Petrov", "Smirnov", "Volkov", "Kuznetsov", "Popov", "Kowalski",
    "Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Dubois", "Moreau",
    "Rossi", "Ferrari", "Bianchi", "Russo", "Hansen", "Lindqvist", "Nielsen"
]

PHONETIC_PAIRS = {
    # Anglo / European
    "smith": ["Smyth", "Schmidt", "Smithe"],
    "johnson": ["Jansen", "Johnsen", "Jonson"],
    "williams": ["Williamson", "Willis", "Wilhelm"],
    "brown": ["Braun", "Browne", "Braune"],
    "davis": ["Davies", "Davison", "Davidson"],
    "miller": ["Mueller", "Muller", "Millar"],
    "wilson": ["Wilkinson", "Willson", "Wills"],
    "clark": ["Clarke", "Clerk", "Clarkson"],
    "white": ["Whyte", "Weitz", "Witt"],
    "harris": ["Harrison", "Harry", "Harries"],
    "martin": ["Martins", "Martinet", "Morton"],
    "thompson": ["Thomson", "Tompson", "Tompkins"],
    "mueller": ["Muller", "Moeller", "Miller"],
    "schmidt": ["Schmitt", "Schmid", "Smith"],
    "dubois": ["Du Bois", "Duboys", "Duboix"],
    "rossi": ["Rosi", "Russo", "De Rossi"],
    "hansen": ["Hanson", "Hanssen", "Jensen"],
    # MENA
    "mohammed": ["Muhammad", "Mohamed", "Mahmud", "Mohammad"],
    "muhammad": ["Mohammed", "Mohamed", "Mahmoud"],
    "ali": ["Aly", "Alley", "Alie"],
    "hassan": ["Hasan", "Hassane", "Hasen"],
    "hussein": ["Hussain", "Husein", "Husain"],
    "khan": ["Kahn", "Cahn", "Cann"],
    "yassin": ["Yasin", "Yacine", "Yassine"],
    "omar": ["Umar", "Omer", "Omarr"],
    "khalid": ["Khaled", "Calid", "Kalid"],
    "al-mansoor": ["Al-Mansur", "El-Mansoor", "Mansoor"],
    "al-fassi": ["Al-Fasi", "El-Fassi", "Fassi"],
    "haddad": ["Hadad", "Haddat", "Haddadi"],
    # East Asian
    "zhang": ["Chang", "Tsang", "Zhan"],
    "wang": ["Wong", "Vang", "Huang"],
    "chen": ["Chan", "Chun", "Tan"],
    "kim": ["Gim", "Kym", "Keem"],
    "lee": ["Li", "Rhee", "Yi"],
    "park": ["Bak", "Pak", "Bark"],
    "tanaka": ["Tanacca", "Tanake"],
    "sato": ["Satou", "Satoh", "Satto"],
    # South Asian
    "sharma": ["Sarma", "Sharmah", "Sharman"],
    "patel": ["Patell", "Patil", "Patle"],
    "singh": ["Sinh", "Sing", "Singhji"],
    "rao": ["Rau", "Row", "Raov"],
    # African
    "okafor": ["Okafore", "Ocafur", "Okafur"],
    "mwangi": ["Muangi", "Mwangee", "Mwangy"],
    "mensah": ["Mensa", "Menseh", "Menzah"],
    # Slavic
    "dmitry": ["Dmitri", "Dmitriy", "Dmitrii"],
    "ivanov": ["Ivanoff", "Iwanow", "Ivanovs"],
    "petrov": ["Petroff", "Petrow", "Petrovic"],
    # Hispanic
    "rodriguez": ["Rodrigues", "Rodrigez", "Rodriquez"],
    "hernandez": ["Hernandes", "Fernandez", "Hernande"],
    "gonzalez": ["Gonzales", "Gonsalez", "Gonzalo"]
}


# ==============================================================================
# NAME NORMALIZATION & FILTERING
# ==============================================================================

def clean_and_normalize_name(raw_name: str) -> Optional[str]:
    """
    Cleans raw screening name text, formats 'LASTNAME, Firstname Middlename'
    into natural order 'Firstname Middlename Lastname', strips non-alphabetical
    cruft, and rejects institutional/corporate names.
    """
    if not isinstance(raw_name, str):
        return None
    name = raw_name.strip().strip('"').strip("'")
    if not name:
        return None

    # Check for non-individual keywords
    tokens_upper = re.findall(r"\b[A-Za-z]+\b", name.upper())
    if any(tok in NON_INDIVIDUAL_KEYWORDS for tok in tokens_upper):
        return None

    # Handle 'LASTNAME, Firstname Middlename' structure
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        surname = parts[0]
        given = parts[1]
        name = f"{given} {surname}"

    # Remove unwanted prefixes/titles like Dr., Shaykh, Sheik, Mr., Mrs.
    name = re.sub(r"\b(Dr\.|Shaykh|Sheik|Sheikh|Mr\.|Mrs\.|Ms\.|Prof\.)\s*", "", name, flags=re.IGNORECASE)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Convert words to Title Case if they are all-uppercase or all-lowercase
    words = name.split()
    capitalized = []
    for w in words:
        # Preserve hyphenated parts e.g. 'AL-ZOMOR' -> 'Al-Zomor'
        if "-" in w:
            w = "-".join(sub.capitalize() for sub in w.split("-"))
        elif w.lower() in ("bin", "ibn", "al", "el", "de", "del", "van", "von", "da"):
            w = w.capitalize()
        elif w.isupper() or w.islower():
            w = w.capitalize()
        capitalized.append(w)
    name = " ".join(capitalized)

    # Must contain at least two tokens and some letters
    tokens = name.split()
    if len(tokens) < 2:
        return None

    return name

def load_source_names(
    csv_path: Optional[str] = "sdn.csv",
    name_column: Optional[str] = None,
    augment_diversity: bool = True
) -> List[str]:
    """
    Loads source names from input CSV, detecting the appropriate column,
    and optionally augments the pool with diverse global multi-ethnic names across
    East Asian, South Asian, African, MENA, Slavic, European, and Hispanic traditions.
    """
    cleaned_names: List[str] = []
    seen = set()

    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            target_col = None

            if name_column and name_column in df.columns:
                target_col = name_column
            else:
                candidates = ["name", "hitname", "client_name", "full_name", "individual_name"]
                for col in df.columns:
                    if col.strip().lower() in candidates:
                        target_col = col
                        break
                if not target_col:
                    target_col = df.columns[0]

            raw_names = df[target_col].dropna().astype(str).tolist()
            for r in raw_names:
                norm = clean_and_normalize_name(r)
                if norm and norm.lower() not in seen:
                    seen.add(norm.lower())
                    cleaned_names.append(norm)
        except Exception as e:
            print(f"[Warning] Could not load CSV at '{csv_path}': {e}. Using global name pools.")

    # Augment with rich multi-ethnic global name varieties
    if augment_diversity or not cleaned_names:
        injected_count = 0
        for region, names in GLOBAL_NAME_POOLS.items():
            for n in names:
                if n.lower() not in seen:
                    seen.add(n.lower())
                    cleaned_names.append(n)
                    injected_count += 1

    if not cleaned_names:
        raise ValueError(f"No valid individual names found in input source.")

    return cleaned_names

# ==============================================================================
# PIPELINE A: PROGRAMMATIC MUTATIONS & ZERO-TOKEN CoT ENGINE
# ==============================================================================

KEYBOARD_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'hukmn',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wazxde', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx'
}

def apply_minor_typo(name: str) -> Tuple[str, str, str, str]:
    """
    Applies a Levenshtein distance <= 2 typo mutation (character transposition,
    substitution, or single omission).
    Returns (hit_name, matching_text, disposition_code, thinking).

    Three label-consistency guards apply, because the result must stay a *typo*:
      - transposing two identical characters (e.g. the double 'n' of 'Anna') must not
        leave the client name untouched,
      - the result must never be the curated gendered counterpart of the given name
        (e.g. 'Martin' -> 'Martina'), which belongs to the gender-variant category,
        not to DISP_MINOR_TYPO, and
      - the accepted hit is verified with rapidfuzz's Levenshtein distance so that the
        documented <= 2 edit contract actually holds for the emitted pair.
    """
    tokens = name.split()
    chosen_token_idx = random.randint(0, len(tokens) - 1)
    orig_token = tokens[chosen_token_idx]
    mutated_token = ""
    hit_name = ""
    desc = ""

    for _ in range(8):
        if len(orig_token) > 3:
            token_chars = list(orig_token)
            mutation_kind = random.choice(["transpose", "substitute", "omit"])
            idx = random.randint(1, len(token_chars) - 2)

            if mutation_kind == "transpose":
                if token_chars[idx] == token_chars[idx + 1]:
                    # swapping identical characters is a no-op — redraw
                    continue
                token_chars[idx], token_chars[idx + 1] = token_chars[idx + 1], token_chars[idx]
                desc = f"adjacent character transposition ('{orig_token[idx]}{orig_token[idx+1]}' -> '{token_chars[idx]}{token_chars[idx+1]}')"
            elif mutation_kind == "substitute":
                char = token_chars[idx].lower()
                replacement = random.choice(KEYBOARD_NEIGHBORS.get(char, "aeiou"))
                token_chars[idx] = replacement.upper() if orig_token[idx].isupper() else replacement
                desc = f"single typographical substitution ('{orig_token[idx]}' -> '{token_chars[idx]}')"
            else: # omit
                removed = token_chars.pop(idx)
                desc = f"single typographical omission of character '{removed}'"
            mutated_token = "".join(token_chars)
        else:
            # Fallback substitution
            mutated_token = orig_token + "n"
            desc = "single typographical character append"

        candidate_tokens = tokens[:]
        candidate_tokens[chosen_token_idx] = mutated_token
        candidate_hit = " ".join(candidate_tokens)
        if candidate_hit.lower() == name.lower() or is_gender_variant_pair(name, candidate_hit):
            continue  # label-consistency guard — redraw
        if Levenshtein.distance(name.lower(), candidate_hit.lower()) > 2:
            continue  # edit-distance contract of this generator — redraw
        hit_name = candidate_hit
        break

    if not hit_name:
        # Deterministic last resort applied to the surname token: it can neither
        # duplicate the client name nor form a curated gendered counterpart.
        fallback_tokens = tokens[:]
        fallback_tokens[-1] = fallback_tokens[-1] + "e"
        mutated_token = fallback_tokens[-1]
        hit_name = " ".join(fallback_tokens)
        desc = "single typographical character append"

    # Overlapping text: unaffected tokens
    unaffected = [t for i, t in enumerate(name.split()) if i != chosen_token_idx]
    matching_text = " ".join(unaffected) if unaffected else name[:3]

    thinking = (
        f"Comparing the name strings: the primary identifying words '{matching_text}' match identically. "
        f"The candidate hit has a minor clerical difference in word '{orig_token}' (written as '{mutated_token}'), "
        f"which reflects an {desc}. This is a classic typographical error resulting from fast typing or clerical transcription. "
        f"Because the underlying names match closely with no conflicting family or personal names, "
        f"the candidate is highly likely the same individual. Recommendation: Escalate to analyst."
    )
    disposition_code = "DISP_MINOR_TYPO"
    return hit_name, matching_text, disposition_code, thinking

def apply_token_transposition(name: str) -> Tuple[str, str, str, str]:
    """
    Swaps token ordering (e.g., First Last -> Last, First or Given Middle Last -> Last Given Middle).
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    if len(tokens) == 2:
        hit_name = f"{tokens[1]} {tokens[0]}"
        desc = "reversal of first name and family name order"
    else:
        # 3 or more tokens
        tokens_perm = tokens.copy()
        last = tokens_perm.pop()
        tokens_perm.insert(0, last)
        hit_name = " ".join(tokens_perm)
        desc = "placement of the family name at the beginning of the name"

    matching_text = " ".join(tokens)

    thinking = (
        f"Evaluating word order: both records contain the exact same set of names ('{matching_text}'). "
        f"The variation is purely due to {desc}. In global screening databases, names are frequently recorded "
        f"in family-name-first or given-name-first convention depending on the jurisdiction or database format. "
        f"All individual name parts are fully accounted for with zero missing or conflicting names. "
        f"This refers to the identical individual. Recommendation: Escalate to analyst."
    )
    disposition_code = "DISP_TOKEN_ORDER_SWAP"
    return hit_name, matching_text, disposition_code, thinking

def apply_middle_truncation_expansion(name: str) -> Tuple[str, str, str, str]:
    """
    Truncates or expands middle name/initial.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    if len(tokens) >= 3:
        mid = tokens[1]
        if len(mid) > 2:
            # Truncate
            tokens[1] = f"{mid[0]}."
            hit_name = " ".join(tokens)
            desc = f"shortening of middle name '{mid}' to initial '{tokens[1]}'"
            disposition_code = "DISP_MIDDLE_NAME_ABBREVIATED"
        else:
            # Expand
            expansion = random.choice(ALTERNATIVE_MIDDLE_NAMES)
            tokens[1] = expansion
            hit_name = " ".join(tokens)
            desc = f"expansion of middle initial '{mid}' to full middle name '{expansion}'"
            disposition_code = "DISP_MIDDLE_INITIAL_EXPANDED"
        matching_text = f"{tokens[0]} {tokens[-1]}"
    else:
        # Insert a middle initial
        initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        hit_name = f"{tokens[0]} {initial}. {tokens[1]}"
        desc = f"addition of middle initial '{initial}.'"
        disposition_code = "DISP_MIDDLE_NAME_OMITTED"
        matching_text = f"{tokens[0]} {tokens[1]}"

    thinking = (
        f"Reviewing the core identifiers: the client and candidate hit share the exact same first name and surname ('{matching_text}'). "
        f"The only difference is the {desc}. In banking and legal records, middle names are routinely abbreviated to initials "
        f"or expanded without changing legal identity. There are no contradictory identity markers present. "
        f"This requires manual verification. Recommendation: Escalate to analyst."
    )
    return hit_name, matching_text, disposition_code, thinking

def apply_phonetic_shift(name: str) -> Tuple[str, str, str, str]:
    """
    Substitutes surname or given name with a phonetically shifted variant or sound-alike.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    last_lower = tokens[-1].lower()

    if last_lower in PHONETIC_PAIRS:
        shifted = random.choice(PHONETIC_PAIRS[last_lower])
        orig = tokens[-1]
        tokens[-1] = shifted
        desc = f"phonetic variation of surname from '{orig}' to '{shifted}'"
    else:
        orig = tokens[-1]
        if orig.endswith("son"):
            shifted = orig[:-3] + "sen"
        elif orig.endswith("y"):
            shifted = orig[:-1] + "ey"
        elif "ff" in orig.lower():
            shifted = re.sub(r"[fF]{2}", "ph", orig)
        else:
            # Must not resample the original surname — that would leave the hit
            # identical to the client name, which is not a screening alert at all.
            orig_lower = orig.lower()
            shifted = random.choice([s for s in ALTERNATIVE_SURNAMES if s.lower() != orig_lower])
        tokens[-1] = shifted
        desc = f"surname substitution from '{orig}' to '{shifted}'"

    hit_name = " ".join(tokens)
    matching_text = " ".join(tokens[:-1])

    thinking = (
        f"Analyzing the match: the first name '{matching_text}' matches across both records. "
        f"However, the candidate has a different family name: {desc}. "
        f"While the two surnames may sound somewhat similar or share phonetic traits, they represent distinct family heritages and legal names. "
        f"A shared common first name with a diverging family name indicates two separate individuals. Recommendation: Disqualify."
    )
    disposition_code = "DISP_PHONETIC_SURNAME_DIVERGENCE"
    return hit_name, matching_text, disposition_code, thinking

def apply_middle_name_conflict(name: str) -> Tuple[str, str, str, str]:
    """
    Substitutes an entirely distinct middle name, retaining first and last names.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    if len(tokens) >= 3:
        orig_mid = tokens[1]
        new_mid = random.choice([m for m in ALTERNATIVE_MIDDLE_NAMES if m.lower() != orig_mid.lower()])
        tokens[1] = new_mid
        hit_name = " ".join(tokens)
        matching_text = f"{tokens[0]} {tokens[-1]}"
        desc = f"conflicting middle name '{new_mid}' replacing '{orig_mid}'"
    else:
        mid = random.choice(ALTERNATIVE_MIDDLE_NAMES)
        hit_name = f"{tokens[0]} {mid} {tokens[1]}"
        matching_text = f"{tokens[0]} {tokens[1]}"
        desc = f"introduction of a distinct middle name '{mid}'"

    thinking = (
        f"Evaluating middle names: the first name and surname ('{matching_text}') match identically. "
        f"However, there is an irreconcilable discrepancy: {desc}. "
        f"These are two entirely separate personal given names—not a nickname, abbreviation, or typing slip. "
        f"When two individuals have completely different middle names, they are distinct people who happen to share a first and last name. Recommendation: Disqualify."
    )
    disposition_code = "DISP_MIDDLE_NAME_CONFLICT"
    return hit_name, matching_text, disposition_code, thinking

def apply_distinct_surname_swap(name: str) -> Tuple[str, str, str, str]:
    """
    Retains first name token(s), but replaces surname with a completely distinct surname.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    orig_surname = tokens[-1]
    new_surname = random.choice([s for s in ALTERNATIVE_SURNAMES if s.lower() != orig_surname.lower()])
    tokens[-1] = new_surname
    hit_name = " ".join(tokens)
    matching_text = " ".join(tokens[:-1])

    thinking = (
        f"Comparing identity tokens: both client and candidate share the common first name '{matching_text}'. "
        f"However, the surnames are completely different: client '{orig_surname}' vs candidate '{new_surname}'. "
        f"In identity screening, the family name is the primary differentiator, and these two surnames have zero relation. "
        f"A match on a common given name alone without a matching surname is a coincidental false positive. Recommendation: Disqualify."
    )
    disposition_code = "DISP_SURNAME_CONFLICT"
    return hit_name, matching_text, disposition_code, thinking

# ==============================================================================
# REAL-WORLD NOISE MUTATIONS (5 NEW — ALL TRUE POSITIVE)
# These simulate name recording differences that genuinely occur in production
# screening datasets. All produce the same individual recorded differently.
# ==============================================================================

# Cross-standard transliteration map: token.lower() → list of known variants
TRANSLITERATION_VARIANTS: Dict[str, List[str]] = {
    # MENA — Arabic name transliteration (BGN/PCGN vs ISO 233 vs informal)
    "muhammad":   ["Mohammed", "Mohamed", "Mukhammad", "Mahomed", "Mehmet"],
    "mohammed":   ["Muhammad", "Mohamed", "Mukhammad", "Mahomed"],
    "omar":       ["Umar", "Omer", "Omarr", "Amr"],
    "hassan":     ["Hasan", "Hassane", "Hasen", "Huseyn"],
    "hussein":    ["Hussain", "Husein", "Husain", "Hossain"],
    "khalid":     ["Khaled", "Kalid", "Calid", "Chalid"],
    "yassin":     ["Yasin", "Yacine", "Yassine"],
    "ali":        ["Aly", "Alie", "Aliy"],
    "tariq":      ["Tarik", "Tareck", "Tarek"],
    "karim":      ["Kareem", "Karime", "Karem"],
    # Slavic — Romanization standards (BGN/PCGN vs GOST vs scholarly)
    "dmitry":     ["Dmitri", "Dmitriy", "Dmitrii", "Dmytro"],
    "aleksei":    ["Alexei", "Aleksey", "Alexey", "Aleksiy"],
    "mikhail":    ["Michael", "Mikail", "Mihail"],
    "nikolai":    ["Nikolay", "Nikolaos", "Nikola"],
    "yury":       ["Yuri", "Iuri", "Juri", "Jurij"],
    "sergei":     ["Sergey", "Serghei", "Serhiy"],
    # East Asian — Romanization (Pinyin vs Wade-Giles vs Cantonese Yale)
    "zhang":      ["Chang", "Tsang", "Cheung"],
    "wang":       ["Wong", "Vang", "Huang"],
    "chen":       ["Chan", "Tan", "Chin"],
    "liu":        ["Lau", "Lew", "Lo"],
    "li":         ["Lee", "Ly", "Lei"],
    # South Asian — transliteration variants
    "priya":      ["Pria", "Preya", "Priyah"],
    "rajesh":     ["Rajish", "Rajech", "Rajec"],
}

# Names with diacritics and their ASCII equivalents
DIACRITIC_PAIRS: List[Tuple[str, str]] = [
    ("Müller", "Muller"), ("Möller", "Moller"), ("Schröder", "Schroder"),
    ("Günter", "Gunter"), ("Jürgen", "Jurgen"), ("Björn", "Bjorn"),
    ("Göran", "Goran"), ("Håkan", "Hakan"), ("Åsa", "Asa"),
    ("José", "Jose"), ("María", "Maria"), ("Álvarez", "Alvarez"),
    ("Gonzáles", "Gonzales"), ("Martínez", "Martinez"), ("García", "Garcia"),
    ("François", "Francois"), ("Hébert", "Hebert"), ("Renée", "Renee"),
    ("Çelik", "Celik"), ("Şahin", "Sahin"), ("Özdemir", "Ozdemir"),
    ("Ağaoğlu", "Agaoglu"), ("Žižek", "Zizek"), ("Šimić", "Simic"),
    ("Dvořák", "Dvorak"), ("Novák", "Novak"), ("Václav", "Vaclav"),
]

# Compound / hyphenated names and their split equivalents
COMPOUND_NAME_PAIRS: List[Tuple[str, str]] = [
    ("Jean-Paul", "Jean Paul"), ("Jean-Marie", "Jean Marie"),
    ("Jean-Pierre", "Jean Pierre"), ("Jean-Claude", "Jean Claude"),
    ("Anne-Marie", "Anne Marie"), ("Marie-Claire", "Marie Claire"),
    ("Al-Mansoor", "Al Mansoor"), ("Al-Fassi", "Al Fassi"),
    ("Al-Hashimi", "Al Hashimi"), ("Al-Masri", "Al Masri"),
    ("Abd-Al-Rahman", "Abd Al Rahman"), ("Ibn-Khalid", "Ibn Khalid"),
    ("Bin-Rashid", "Bin Rashid"), ("Bint-Ali", "Bint Ali"),
    ("Min-jun", "Min Jun"), ("Ha-eun", "Ha Eun"), ("Ji-woo", "Ji Woo"),
    ("Sun-woo", "Sun Woo"), ("Hyun-woo", "Hyun Woo"),
]

# ==============================================================================
# GENDER-INFLECTED GIVEN NAME VARIANTS (FALSE POSITIVE SIGNAL)
# A near-identical pair whose given name is the gendered counterpart of the other
# (e.g. 'Daniel Martin' vs 'Daniela Martin') describes two DIFFERENT individuals.
# Every pair below is formed with a true feminine marker (-a, -e, -ie, -ine, -ina),
# so the two strings look almost identical while the recorded gender does not match.
#
# Deliberately EXCLUDED as ambiguous / noisy signals:
#   - Mario -> Maria, Kamal -> Kamala : near-homophone spelling variants that are at
#     least as likely to be a clerical slip as a genuine gender difference.
#   - Francis -> Frances : pure final-vowel substitution, indistinguishable from a typo.
#   - Andres -> Andrea, Nicholas -> Nicola, Simon -> Simone : the 'feminine' token is a
#     masculine given name in Italian, so the direction would be ambiguous.
#   - Michel -> Michele : accented-only change, and 'Michele' is a masculine name in Italian.
# Known coverage gap: East Asian names are largely gender-neutral and produce no such
# pairs; the South Asian list is intentionally short because its markers are irregular.
# ==============================================================================

GENDER_VARIANT_NAME_PAIRS: List[Tuple[str, str]] = [
    # Anglo / Germanic / European
    ("Daniel", "Daniela"), ("Alexander", "Alexandra"), ("Michael", "Michaela"),
    ("Martin", "Martina"), ("Paul", "Paula"), ("Carl", "Carla"), ("Karl", "Karla"),
    ("Eric", "Erika"), ("Robert", "Roberta"), ("Peter", "Petra"), ("George", "Georgia"),
    ("Stephen", "Stephanie"), ("Joseph", "Josephine"), ("Adrian", "Adriana"),
    ("Julian", "Juliana"), ("Gabriel", "Gabriela"), ("Christian", "Christiana"),
    ("Victor", "Victoria"), ("Jan", "Jana"), ("Louis", "Louise"),
    # Hispanic
    ("Juan", "Juana"), ("José", "Josefa"), ("Luis", "Luisa"), ("Manuel", "Manuela"),
    ("Rafael", "Rafaela"), ("Alejandro", "Alejandra"), ("Carlos", "Carla"),
    ("Julio", "Julia"), ("Emilio", "Emilia"), ("Ramón", "Ramona"),
    ("Antonio", "Antonia"), ("Martín", "Martina"),
    # Arabic / MENA — the productive feminine '-a' marker
    ("Khalid", "Khalida"), ("Rashid", "Rashida"), ("Hamid", "Hamida"), ("Farid", "Farida"),
    ("Samir", "Samira"), ("Kamil", "Kamila"), ("Amin", "Amina"), ("Karim", "Karima"),
    ("Nabil", "Nabila"), ("Majid", "Majida"), ("Nadir", "Nadira"), ("Said", "Saida"),
    ("Adel", "Adela"), ("Munir", "Munira"), ("Salim", "Salima"), ("Halim", "Halima"),
    # Slavic
    ("Ivan", "Ivana"), ("Pavel", "Pavla"), ("Stanislav", "Stanislava"),
    ("Vladislav", "Vladislava"), ("Miroslav", "Miroslava"), ("Jaroslav", "Jaroslava"),
    ("Bohdan", "Bohdana"), ("Alexandr", "Alexandra"), ("Zdenek", "Zdenka"),
    ("Vladimir", "Vladimira"),
    # French
    ("Jean", "Jeanne"), ("Marcel", "Marcelle"), ("Pascal", "Pascale"),
    ("Denis", "Denise"), ("François", "Françoise"),
    # Italian
    ("Paolo", "Paola"), ("Carlo", "Carla"), ("Franco", "Franca"), ("Roberto", "Roberta"),
    ("Alessandro", "Alessandra"), ("Emanuele", "Emanuela"), ("Gabriele", "Gabriella"),
    ("Daniele", "Daniela"), ("Valentino", "Valentina"),
    # South Asian (limited — markers are irregular in this tradition)
    ("Amit", "Amita"), ("Anil", "Anila"),
]

def _fold_token(token: str) -> str:
    """
    Lower-cases a name token and strips its diacritics ('José' -> 'jose'), so that
    ASCII-only sanctions sources still match records that preserve accents.
    """
    import unicodedata
    return unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii").lower()

def _build_gender_variant_lookup(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Flattens the curated (male, female) display-form pairs into a bidirectional
    lowercase token lookup used for gender-variant pair construction and detection.
    Both the accented and the ASCII-folded spelling of each token are registered
    ('José' and 'Jose'), because sanctions sources are frequently ASCII-only while
    other records preserve diacritics. Where several masculine names share one
    feminine form (e.g. Carl/Carlos -> Carla), the FIRST declared pair wins; the
    reverse direction only ever needs one valid counterpart, so precedence stays
    deterministic rather than ambiguous.
    """
    lookup: Dict[str, str] = {}
    for male, female in pairs:
        for token, counterpart in ((male, female), (female, male)):
            lookup.setdefault(token.lower(), counterpart)
            lookup.setdefault(_fold_token(token), counterpart)
    return lookup

def _build_token_genders(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Maps every curated token to 'masculine' or 'feminine' so that the generated
    reasoning can name the gender of the substituted given name explicitly.
    """
    genders: Dict[str, str] = {}
    for male, female in pairs:
        for token, gender in ((male, "masculine"), (female, "feminine")):
            genders.setdefault(token.lower(), gender)
            genders.setdefault(_fold_token(token), gender)
    return genders

# Token -> gendered counterpart in display form (bidirectional, accent-folded)
GIVEN_NAME_GENDER_VARIANTS: Dict[str, str] = _build_gender_variant_lookup(GENDER_VARIANT_NAME_PAIRS)
# Fast membership set for scanning the source name pool
GENDER_VARIANT_TOKENS: Set[str] = set(GIVEN_NAME_GENDER_VARIANTS)
# Token -> 'masculine' / 'feminine', used by the generated narrative
GENDER_VARIANT_TOKEN_GENDERS: Dict[str, str] = _build_token_genders(GENDER_VARIANT_NAME_PAIRS)

def apply_transliteration_variant(name: str) -> Tuple[str, str, str, str]:
    """
    Replaces a name token with a known cross-standard transliteration variant
    (e.g. Muhammad → Mohammed, Zhang → Chang, Aleksei → Alexey).
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    # Try each token against the transliteration map
    candidates = [(i, t) for i, t in enumerate(tokens) if t.lower() in TRANSLITERATION_VARIANTS]

    if candidates:
        idx, orig = random.choice(candidates)
        variants = TRANSLITERATION_VARIANTS[orig.lower()]
        new_tok = random.choice(variants)
        hit_tokens = tokens[:]
        hit_tokens[idx] = new_tok
        hit_name = " ".join(hit_tokens)
        matching_text = " ".join(t for i, t in enumerate(tokens) if i != idx)
        thinking = (
            f"Reviewing the name pair: '{orig}' in the client record and '{new_tok}' in the candidate hit. "
            f"These are recognised transliteration variants of the same name under different international romanisation standards. "
            f"The remaining name tokens {matching_text!r} match exactly. "
            f"Given the well-documented variation in how this name is romanised across jurisdictions and databases, "
            f"these are the same individual recorded under different transliteration conventions. Recommendation: Escalate to analyst."
        )
    else:
        # Fallback: apply a minor typo as character-level noise
        return apply_minor_typo(name)

    disposition_code = "DISP_TRANSLITERATION_VARIANT"
    return hit_name, matching_text, disposition_code, thinking


def apply_diacritic_normalization(name: str) -> Tuple[str, str, str, str]:
    """
    Strips or restores diacritics in a name token, simulating ASCII-only
    legacy banking systems or different Unicode normalisation passes.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    # Check if any token in the name matches a diacritic pair
    tokens = name.split()
    for diacritic_form, ascii_form in DIACRITIC_PAIRS:
        for i, tok in enumerate(tokens):
            if tok == diacritic_form:
                hit_tokens = tokens[:]
                hit_tokens[i] = ascii_form
                hit_name = " ".join(hit_tokens)
                matching_text = " ".join(t for j, t in enumerate(tokens) if j != i)
                thinking = (
                    f"Examining the name token: client record contains '{diacritic_form}' while the candidate hit shows '{ascii_form}'. "
                    f"This is a classic diacritic normalisation difference. Many legacy banking systems and SWIFT messaging platforms "
                    f"strip accented characters to produce ASCII-only representations, converting '{diacritic_form}' to '{ascii_form}'. "
                    f"The remaining name tokens {matching_text!r} are identical. "
                    f"This is the same individual—the difference is a system encoding artefact, not a different person. Recommendation: Escalate to analyst."
                )
                disposition_code = "DISP_DIACRITIC_STRIPPED"
                return hit_name, matching_text, disposition_code, thinking
            elif tok == ascii_form:
                hit_tokens = tokens[:]
                hit_tokens[i] = diacritic_form
                hit_name = " ".join(hit_tokens)
                matching_text = " ".join(t for j, t in enumerate(tokens) if j != i)
                thinking = (
                    f"Examining the name token: client record contains '{ascii_form}' while the candidate hit shows the accented form '{diacritic_form}'. "
                    f"This is a diacritic restoration difference—the candidate database stores the full Unicode form while the client record uses ASCII. "
                    f"The remaining name tokens {matching_text!r} are identical. "
                    f"This is the same individual under different character encoding standards. Recommendation: Escalate to analyst."
                )
                disposition_code = "DISP_DIACRITIC_STRIPPED"
                return hit_name, matching_text, disposition_code, thinking

    # Fallback: simulate stripping by lowercasing non-ASCII chars generically
    import unicodedata
    hit_tokens = tokens[:]
    changed_idx = -1
    orig_tok = ""
    for i, tok in enumerate(hit_tokens):
        normalized = unicodedata.normalize("NFKD", tok)
        ascii_tok = normalized.encode("ascii", "ignore").decode("ascii")
        if ascii_tok != tok and ascii_tok:
            orig_tok = tok
            hit_tokens[i] = ascii_tok
            changed_idx = i
            break
    if changed_idx >= 0:
        hit_name = " ".join(hit_tokens)
        matching_text = " ".join(t for j, t in enumerate(tokens) if j != changed_idx)
        thinking = (
            f"Examining the name: client record contains '{orig_tok}' while the candidate hit shows the ASCII-normalised form '{hit_tokens[changed_idx]}'. "
            f"This is a diacritic stripping difference introduced by ASCII-only legacy systems. "
            f"The remaining tokens match identically. This is the same individual. Recommendation: Escalate to analyst."
        )
        disposition_code = "DISP_DIACRITIC_STRIPPED"
        return hit_name, matching_text, disposition_code, thinking

    # Double fallback: minor typo
    return apply_minor_typo(name)


def apply_compound_name_restructuring(name: str) -> Tuple[str, str, str, str]:
    """
    Splits a hyphenated compound token into two space-separated tokens, or
    merges adjacent tokens with a hyphen. Simulates formatting differences
    in how compound names (Jean-Paul, Al-Mansoor) are recorded across systems.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()

    # Check for exact compound pair match in tokens
    for hyphen_form, split_form in COMPOUND_NAME_PAIRS:
        for i, tok in enumerate(tokens):
            if tok == hyphen_form:
                # Split the hyphenated token
                split_parts = split_form.split()
                hit_tokens = tokens[:i] + split_parts + tokens[i+1:]
                hit_name = " ".join(hit_tokens)
                matching_text = " ".join(t for j, t in enumerate(tokens) if j != i)
                thinking = (
                    f"Examining the name structure: the client record shows '{hyphen_form}' as a hyphenated compound name, "
                    f"while the candidate hit records it as '{split_form}' with a space separator. "
                    f"Hyphenated compound names are frequently split or joined differently across database systems depending on "
                    f"the jurisdiction's data entry conventions. The underlying name identity is the same. "
                    f"The remaining tokens {matching_text!r} match identically. Recommendation: Escalate to analyst."
                )
                disposition_code = "DISP_COMPOUND_NAME_RESTRUCTURED"
                return hit_name, matching_text, disposition_code, thinking

    # Check for split form in consecutive token pairs
    for hyphen_form, split_form in COMPOUND_NAME_PAIRS:
        split_parts = split_form.split()
        if len(split_parts) == 2:
            for i in range(len(tokens) - 1):
                if tokens[i] == split_parts[0] and tokens[i+1] == split_parts[1]:
                    hit_tokens = tokens[:i] + [hyphen_form] + tokens[i+2:]
                    hit_name = " ".join(hit_tokens)
                    matching_text = " ".join(t for j, t in enumerate(tokens) if j not in (i, i+1))
                    thinking = (
                        f"Examining the name structure: the client record shows '{split_parts[0]} {split_parts[1]}' as two separate tokens, "
                        f"while the candidate hit records it as the hyphenated form '{hyphen_form}'. "
                        f"This is a compound name formatting difference caused by inconsistent data entry conventions across systems. "
                        f"The matching text {matching_text!r} is otherwise identical. Recommendation: Escalate to analyst."
                    )
                    disposition_code = "DISP_COMPOUND_NAME_RESTRUCTURED"
                    return hit_name, matching_text, disposition_code, thinking

    # Fallback: apply token transposition
    return apply_token_transposition(name)


def apply_middle_name_omission(name: str) -> Tuple[str, str, str, str]:
    """
    Removes the middle name token entirely (simulating onboarding forms that
    omit middle names) or adds one from the name pool.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    tokens = name.split()
    if len(tokens) >= 3:
        # Remove middle name
        mid = tokens[1]
        hit_tokens = [tokens[0]] + tokens[2:]
        hit_name = " ".join(hit_tokens)
        matching_text = f"{tokens[0]} {tokens[-1]}"
        thinking = (
            f"Reviewing the name pair: the client record contains '{name}' with the middle name '{mid}', "
            f"while the candidate hit shows '{hit_name}' with the middle name omitted entirely. "
            f"It is common for clients to omit middle names during onboarding due to form field constraints "
            f"or personal preference, while official sanctions lists and regulatory databases include the full name. "
            f"The first name and surname {matching_text!r} are identical. "
            f"The absence of a middle name is not a conflict—it is an absence of data. Recommendation: Escalate to analyst."
        )
    else:
        # Add middle initial (reverse omission — hit has extra middle initial)
        initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        hit_name = f"{tokens[0]} {initial}. {tokens[-1]}"
        matching_text = f"{tokens[0]} {tokens[-1]}"
        thinking = (
            f"Reviewing the name pair: the client record shows '{name}' with no middle initial, "
            f"while the candidate hit shows '{hit_name}' with middle initial '{initial}.'. "
            f"Clients frequently omit their middle initial during onboarding. The first name and surname "
            f"'{matching_text}' match exactly with no conflicting tokens. Recommendation: Escalate to analyst."
        )
    disposition_code = "DISP_MIDDLE_NAME_OMITTED"
    return hit_name, matching_text, disposition_code, thinking


# ==============================================================================
# NEW CATEGORIES: CROSS-GENDER VARIANT (FALSE POSITIVE) & EXACT MATCH (TRUE POSITIVE)
# ==============================================================================

def is_gender_variant_pair(client_name: str, hit_name: str) -> bool:
    """
    True when two names share every token but one, and that single differing token is
    the curated gendered counterpart of the other — 'Daniel Martin' vs 'Daniela
    Martin' or 'Khalida Ali' vs 'Khalid Ali'.

    Deliberately position-agnostic: build_gender_variant_pair() only ever swaps the
    given name, but the detector also catches a gendered swap elsewhere in the name,
    so the DISP_MINOR_TYPO generator can refuse to emit such a pair.
    """
    client_tokens = client_name.split()
    hit_tokens = hit_name.split()
    if len(client_tokens) != len(hit_tokens):
        return False

    diffs = [i for i, (a, b) in enumerate(zip(client_tokens, hit_tokens)) if a.lower() != b.lower()]
    if len(diffs) != 1:
        return False

    idx = diffs[0]
    for a_tok, b_tok in ((client_tokens[idx], hit_tokens[idx]), (hit_tokens[idx], client_tokens[idx])):
        counterpart = GIVEN_NAME_GENDER_VARIANTS.get(_fold_token(a_tok))
        if counterpart and _fold_token(counterpart) == _fold_token(b_tok):
            return True
    return False


def build_gender_variant_pair(client_name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Builds a near-identical CROSS-GENDER FALSE POSITIVE pair: the client's GIVEN NAME
    (token 0) is replaced by its gendered counterpart while every other token (middle
    names, surname) stays byte-identical, e.g.
    'Daniel Martin' -> 'Daniela Martin' and 'Daniela Martin' -> 'Daniel Martin'.

    Only token 0 is ever swapped, so every emitted pair matches the
    DISP_GIVEN_NAME_GENDER_VARIANT label exactly. The direction of the swap follows
    the gender of the client's given name (masculine client -> feminine hit and vice
    versa); the caller controls the mix by which clients it sources
    (build_reserved_gender_variant_clients balances it ~50/50).

    Returns (hit_name, matching_text, disposition_code, thinking), or None when the
    given name does not participate in the curated map — or when the swap would repeat
    a token that is already present (e.g. 'Carla Carl' -> 'Carl Carl'). Such a pair is
    still label-correct, but it reads as an artefact rather than a real record, so the
    caller skips the candidate.
    """
    tokens = client_name.split()
    if not tokens:
        return None

    new_token = GIVEN_NAME_GENDER_VARIANTS.get(_fold_token(tokens[0]))
    if new_token is None:
        return None

    new_token_folded = _fold_token(new_token)
    if any(_fold_token(tok) == new_token_folded for tok in tokens[1:]):
        return None

    orig_token = tokens[0]
    hit_tokens = tokens[:]
    hit_tokens[0] = new_token
    hit_name = " ".join(hit_tokens)
    matching_text = " ".join(tokens[1:]) or hit_name

    orig_gender = GENDER_VARIANT_TOKEN_GENDERS.get(_fold_token(orig_token), "gendered")
    new_gender = GENDER_VARIANT_TOKEN_GENDERS.get(_fold_token(new_token), "gendered")

    thinking = (
        f"Reviewing the pair token by token: the client record reads '{client_name}' and the candidate hit reads '{hit_name}'. "
        f"Every other token {matching_text!r} is identical — the surname included — so the alert is driven by a near-identical string match. "
        f"The single difference is the given name: the client record carries the {orig_gender} form '{orig_token}', "
        f"while the candidate hit carries its {new_gender} counterpart '{new_token}'. "
        f"That is a different personal name rather than a typo, an abbreviation or a transliteration of one name. "
        f"A shared surname combined with an almost identical first name is the classic near-identical false positive pattern, "
        f"and the mismatch in the recorded gender rules out the same natural person. Recommendation: Disqualify."
    )
    disposition_code = "DISP_GIVEN_NAME_GENDER_VARIANT"
    return hit_name, matching_text, disposition_code, thinking


def apply_exact_name_match(name: str) -> Tuple[str, str, str, str]:
    """
    Rule-based EXACT MATCH TRUE POSITIVE: the candidate hit is the client name
    character for character, with zero token-level variation of any kind.

    Generated deterministically without an LLM call, because there is nothing
    ambiguous to reason about — the reasoning is a fixed template. This is the ONLY
    generator allowed to return the client name unchanged, and it is deliberately
    never registered in the random Pipeline A generator list.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    thinking = (
        f"Reviewing the alert: the client record and the candidate hit are identical character for character — '{name}'. "
        f"Every token matches exactly, with no typographical slip, no middle-name abbreviation or expansion, no token reordering, "
        f"no transliteration difference and no diacritic variation to reconcile. "
        f"An exact full-name match against a screening list record is the strongest possible name-level indicator, and there is no "
        f"name-level discrepancy available that could explain the alert away. "
        f"Name identity alone is never sufficient for adverse action, so an analyst must still confirm the secondary identifiers "
        f"(date of birth, nationality, address) before the alert is closed. Recommendation: Escalate to analyst."
    )
    disposition_code = "DISP_EXACT_NAME_MATCH"
    return name, name, disposition_code, thinking


# ==============================================================================
# PIPELINE A: DISPATCHER
# ==============================================================================

def generate_pipeline_a_sample(
    client_name: str,
    target_decision: str,
    double_mutation: bool = False
) -> Tuple[str, str, str, str]:
    """
    Dispatches deterministic mutations and synthesizes human cognitive
    reasoning for Pipeline A. If double_mutation is True, combines two mutations
    for collision fallback resolution.
    Returns (hit_name, matching_text, disposition_code, thinking).
    """
    if target_decision == DECISION_TP:
        generators = [
            apply_minor_typo,
            apply_token_transposition,
            apply_middle_truncation_expansion,
            apply_transliteration_variant,
            apply_diacritic_normalization,
            apply_compound_name_restructuring,
            apply_middle_name_omission,
        ]
        gen = random.choice(generators)
        hit_name, matching_text, disposition_code, thinking = gen(client_name)

        if double_mutation:
            hit_name2, matching_text2, _, thinking2 = apply_minor_typo(hit_name)
            hit_name = hit_name2
            matching_text = matching_text2
            disposition_code = "DISP_MINOR_TYPO"
            thinking = (
                f"Detailed multi-step evaluation: comparing client '{client_name}' against candidate hit '{hit_name}'. "
                f"{thinking} Additionally, a secondary clerical variation is observed: {thinking2} "
                f"Overall assessment confirms high likelihood of the same person despite multiple minor formatting variations."
            )
    else:  # DECISION_FP
        generators = [
            apply_distinct_surname_swap,
            apply_middle_name_conflict,
            apply_phonetic_shift,
        ]
        gen = random.choice(generators)
        hit_name, matching_text, disposition_code, thinking = gen(client_name)

        if double_mutation:
            # Use surname conflict as compounding FP signal
            hit_name2, matching_text2, disposition_code2, thinking2 = apply_distinct_surname_swap(hit_name)
            hit_name = hit_name2
            matching_text = matching_text2
            disposition_code = disposition_code2
            thinking = (
                f"Multi-factor discrepancy review: evaluating client '{client_name}' against '{hit_name}'. "
                f"{thinking} Furthermore, a compounding conflict is identified: {thinking2} "
                f"Decisive cumulative discrepancies confirm these are separate individuals."
            )

    # Invariant: a name-level mutation must never return the client name verbatim.
    # An identical pair is the exclusive territory of apply_exact_name_match(); a
    # stray duplicate here would contradict both DISP_MINOR_TYPO and
    # DISP_EXACT_NAME_MATCH. The fallback stays inside the requested decision's
    # vocabulary, so an emitted disposition code can never disagree with the
    # record's decision.
    if hit_name.strip().lower() == client_name.strip().lower():
        if target_decision == DECISION_TP:
            return apply_minor_typo(client_name)
        return apply_distinct_surname_swap(client_name)

    return hit_name, matching_text, disposition_code, thinking


# ==============================================================================
# PIPELINE B: OLLAMA ASYNC STRUCTURED GENERATION
# ==============================================================================

def clean_ollama_json_response(
    content: str,
    client_name: str = "",
    hit_name: str = "",
    target_decision: str = "",
    fallback_disposition_code: str = "",
    enforce_disposition_code: Optional[str] = None,
    enforce_decision: Optional[str] = None,
    override_counter: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Robustly parses raw text from LLM, stripping markdown code blocks,
    recovering JSON substrings, and normalizing disposition_code against
    the fixed controlled vocabulary (DISPOSITION_CODES).

    enforce_disposition_code / enforce_decision:
        When supplied (used by the reserved categories, e.g. the cross-gender
        variant), the category's code and decision WIN over whatever the model
        produced, so a reserved record can never carry a contradictory label.
        Every correction is tallied so the caller can discard the model's — by then
        untrustworthy — narrative in favour of the programmatic one.
    override_counter:
        Optional tally incremented with 'disposition_code' / 'decision' /
        'plausibility' whenever a model-supplied value had to be corrected. It
        exists for the pipeline metrics only and never affects control flow.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Could not parse JSON from model output: {text[:100]}")

    # Normalize thinking
    if "thinking" not in data or not data["thinking"]:
        for alt in ["reasoning", "rationale", "explanation", "analysis", "thought_process"]:
            if alt in data and data[alt]:
                data["thinking"] = str(data[alt])
                break
    if not data.get("thinking"):
        # 'thinking' is required by ScreeningAlert; failing here routes the sample through
        # the retry loop in generate_single_ollama_sample instead of dying at validation.
        raise ValueError("Model output contained no reasoning text ('thinking').")

    # Normalize decision
    dec = str(data.get("decision", "")).strip().lower()
    if "escalat" in dec or "true" in dec:
        data["decision"] = DECISION_TP
    elif "disqualif" in dec or "false" in dec or "reject" in dec:
        data["decision"] = DECISION_FP
    else:
        data["decision"] = target_decision or DECISION_TP

    # Normalize disposition_code — validate against the controlled vocabulary.
    # The model's code is normalised first (case, spaces/hyphens, an optional missing
    # 'DISP_' prefix), an exact vocabulary hit always wins, and loose containment is only
    # attempted for strings long enough to be meaningful. Without the length guard an
    # empty or truncated code would match the FIRST vocabulary entry ('DISP_MINOR_TYPO')
    # for every alert.
    raw_code = str(data.get("disposition_code") or data.get("key_reason") or "").strip()
    raw_code = re.sub(r"[\s\-]+", "_", raw_code).strip("_.:").upper()
    if raw_code and not raw_code.startswith("DISP_"):
        raw_code = f"DISP_{raw_code}"

    matched = raw_code if raw_code in DISPOSITION_CODES else None
    if matched is None and len(raw_code) >= 8:
        matched = next((k for k in DISPOSITION_CODES if k in raw_code or raw_code in k), None)
    if matched is None:
        if fallback_disposition_code and fallback_disposition_code in DISPOSITION_CODES:
            # Use the programmatically generated fallback code
            matched = fallback_disposition_code
        else:
            # Last resort: infer from decision
            matched = "DISP_MINOR_TYPO" if data["decision"] == DECISION_TP else "DISP_SURNAME_CONFLICT"
    data["disposition_code"] = matched

    # Normalize matching_text
    if "matching_text" not in data or not data["matching_text"]:
        for alt in ["matching_tokens", "matched_text", "match", "overlap"]:
            if alt in data and data[alt]:
                data["matching_text"] = str(data[alt])
                break
        # Fallback: compute overlapping tokens between client_name and hit_name
        if "matching_text" not in data or not data["matching_text"]:
            client_toks = set(re.findall(r"\b\w+\b", client_name.lower()))
            common: List[str] = []
            seen_toks: Set[str] = set()
            for tok in re.findall(r"\b\w+\b", hit_name):
                low = tok.lower()
                if low in client_toks and low not in seen_toks:
                    seen_toks.add(low)
                    common.append(tok)
            data["matching_text"] = " ".join(common) if common else hit_name

    # --- Category enforcement & plausibility (always runs LAST) ------------------
    def _tally(key: str) -> None:
        if override_counter is not None:
            override_counter[key] = override_counter.get(key, 0) + 1

    # An exact-match code is only plausible when the two names really are identical.
    # For any other pair the programmatically derived code wins.
    if data["disposition_code"] == "DISP_EXACT_NAME_MATCH" and (
        client_name.strip().lower() != hit_name.strip().lower()
    ):
        data["disposition_code"] = (
            fallback_disposition_code if fallback_disposition_code in DISPOSITION_CODES
            else ("DISP_MINOR_TYPO" if data["decision"] == DECISION_TP else "DISP_SURNAME_CONFLICT")
        )
        _tally("plausibility")

    # A disposition code and a decision are two views of the same judgement: a code that
    # belongs to the FP vocabulary can never ship with 'escalate to analyst', and a TP code
    # can never ship with 'disqualify'. The code family therefore wins over the model's
    # decision, which keeps the CSV internally consistent for fine-tuning.
    code_family_decision = (
        DECISION_FP if data["disposition_code"] in DISPOSITION_FP_CODES
        else DECISION_TP if data["disposition_code"] in DISPOSITION_TP_CODES
        else None
    )
    if code_family_decision and data["decision"] != code_family_decision:
        data["decision"] = code_family_decision
        _tally("decision")

    # Reserved categories (e.g. the cross-gender variant) must always end up with the
    # category's own code and decision, whatever label the model produced.
    if enforce_disposition_code and data["disposition_code"] != enforce_disposition_code:
        data["disposition_code"] = enforce_disposition_code
        _tally("disposition_code")

    if enforce_decision and data["decision"] != enforce_decision:
        data["decision"] = enforce_decision
        _tally("decision")

    return data

async def generate_single_ollama_sample(
    client: ollama.AsyncClient,
    model_name: str,
    client_name: str,
    hit_name: str,
    target_decision: str,
    fallback_disposition_code: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
    enforce_disposition_code: Optional[str] = None,
    enforce_decision: Optional[str] = None,
    override_counter: Optional[Dict[str, int]] = None
) -> Optional[ScreeningAlert]:
    """
    Submits a programmatically generated candidate pair (client_name, hit_name) to Ollama.
    The model acts as a human compliance screening analyst, producing natural cognitive
    thinking (no math formulas), a fixed-vocabulary disposition_code, and the screening decision.

    Identical name pairs are rejected up front (raise ValueError): they are the exclusive
    territory of the rule-based apply_exact_name_match() generator and must never cost an
    LLM call or carry an LLM narrative.
    The prompt context is derived from the pair itself, so reserved categories get the
    reasoning they need:
      - one token is the gendered counterpart   -> NEAR-IDENTICAL CROSS-GENDER context
      - anything else                           -> the generic True/False Positive context
    enforce_disposition_code / enforce_decision pin the final label of reserved
    categories, and override_counter tallies every correction for the pipeline metrics.

    A sample is only accepted when the model's own decision needed no correction AND agrees
    with the alert's requested resolution. A narrative whose conclusion contradicts the
    emitted label would be unusable training data, so such samples are discarded (tallied,
    never retried) and the caller emits the matching programmatic record instead. Purely
    cosmetic code substitutions (e.g. an informal code normalised onto the vocabulary) keep
    the model's narrative, because it still describes the same name comparison.
    """
    if client_name.strip().lower() == hit_name.strip().lower():
        raise ValueError(
            f"Identical name pair '{client_name}' reached the Ollama path. Exact matches are "
            "rule-based (apply_exact_name_match) and must never be sent to the model."
        )
    if is_gender_variant_pair(client_name, hit_name):
        context = (
            "Screening Context: This is a NEAR-IDENTICAL CROSS-GENDER alert. The two names differ by a single token, "
            "and that token is the gendered counterpart of the other one (for example 'Daniel' vs 'Daniela', "
            "'Khalid' vs 'Khalida'); every other token, the surname included, is identical. "
            "Reason about whether this can be the same natural person given that the recorded gender does not match, "
            "and conclude with 'disqualify'."
        )
    elif target_decision == DECISION_TP:
        context = (
            "Screening Context: This is a True Positive alert where the candidate hit is likely the same person "
            "with minor name recording differences (e.g. slight typo, token order swap, middle name abbreviation, "
            "transliteration variant, diacritic stripping, or compound name restructuring). "
            "Explain your reasoning step-by-step like a human compliance analyst crisply and conclude with 'escalate to analyst'."
        )
    else:
        context = (
            "Screening Context: This is a False Positive alert where the candidate hit is a distinct individual "
            "despite coincidental name overlap (e.g. simple non-matching pairs like xxxxxx yyyyyyy vs zzzzz aaaaaaa "
            "irreconcilably different surname, conflicting middle name, "
            "or phonetically divergent but legally distinct surname). "
            "Explain your reasoning step-by-step like a human compliance analyst crisply and conclude with 'disqualify'."
        )

    disposition_vocab = "\n".join(f"  {k} — {v}" for k, v in DISPOSITION_CODES.items())

    user_prompt = (
        f"Client Name: \"{client_name}\"\n"
        f"Candidate Hit Name: \"{hit_name}\"\n"
        f"{context}\n\n"
        f"Compare the two names token-by-token as a human reviewer. "
        f"Do NOT use mathematical formulas, algorithms, or similarity scores. "
        f"Output raw JSON with exactly these fields:\n"
        f"- \"matching_text\": overlapping name words or tokens shared between the two names\n"
        f"- \"disposition_code\": EXACTLY one of the following codes (copy it exactly):\n"
        f"{disposition_vocab}\n"
        f"- \"thinking\": human cognitive review of the name comparison\n"
        f"- \"decision\": 'escalate to analyst' or 'disqualify'"
    )

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=ScreeningAlert.model_json_schema(),
                    options={"temperature": 0.7}
                )
                raw_content = response["message"]["content"]
                # Corrections are tallied per attempt into a local dict and merged into the
                # shared counter afterwards, so the caller can tell whether THIS sample's
                # label had to be corrected.
                local_overrides: Dict[str, int] = {}
                parsed_data = clean_ollama_json_response(
                    raw_content,
                    client_name=client_name,
                    hit_name=hit_name,
                    target_decision=target_decision,
                    fallback_disposition_code=fallback_disposition_code,
                    enforce_disposition_code=enforce_disposition_code,
                    enforce_decision=enforce_decision,
                    override_counter=local_overrides
                )
                if override_counter is not None:
                    for key, value in local_overrides.items():
                        override_counter[key] = override_counter.get(key, 0) + value

                alert = ScreeningAlert.model_validate(parsed_data)

                # The model's conclusion must be the resolution this alert was generated for:
                # otherwise the emitted label would drift away from the requested TP/FP
                # balance and the narrative would argue for the other verdict.
                if target_decision and alert.decision != target_decision:
                    if override_counter is not None:
                        override_counter["decision_mismatch"] = override_counter.get("decision_mismatch", 0) + 1
                    return None

                if {"decision", "decision_mismatch"} & set(local_overrides):
                    # The decision had to be corrected (or disagreed with the request), so the
                    # narrative describes a different judgement than the emitted label: discard
                    # the sample and let the caller emit the programmatic record.
                    return None
                return alert
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "concurrent" in err_str or "rate" in err_str
                if attempt == max_retries - 1:
                    return None
                backoff = (3.0 * (attempt + 1)) if is_rate_limit else (1.0 * (attempt + 1))
                await asyncio.sleep(backoff)
    return None

# ==============================================================================
# RESERVED CATEGORY SOURCING
# ==============================================================================

def build_reserved_exact_match_names(source_names: List[str], count: int) -> List[str]:
    """
    Picks `count` distinct source names that will be emitted as rule-based exact-match
    records (hit_name == client_name). Sampling without replacement stops the same
    person from appearing twice on a list, and distinct names keep every generated
    (client_name, hit_name) pair globally unique.

    Raises ValueError when the requested count exceeds the number of distinct names.
    """
    if count <= 0:
        return []
    pool = list(dict.fromkeys(source_names))
    if count > len(pool):
        raise ValueError(
            f"Cannot generate {count} exact-match records from only {len(pool)} distinct source names."
        )
    return random.sample(pool, count)


def _gendered_token_gender(name: str) -> Optional[str]:
    """
    Returns the gender class ('masculine' / 'feminine') of the client's given name
    (token 0) when it participates in the curated gender map, or None otherwise.

    Deliberately restricted to token 0 so that sourcing agrees with
    build_gender_variant_pair(), which only ever swaps the given name — a name whose
    gendered token sits in a middle or surname position can never become a
    DISP_GIVEN_NAME_GENDER_VARIANT pair and must not be sourced as one.
    """
    tokens = name.split()
    if not tokens:
        return None
    return GENDER_VARIANT_TOKEN_GENDERS.get(_fold_token(tokens[0]))


def build_reserved_gender_variant_clients(source_names: List[str], count: int) -> List[str]:
    """
    Returns client names that will be turned into near-identical CROSS-GENDER alerts by
    build_gender_variant_pair().

    Two-tier sourcing preserves both realism and an exact count:
      1. real source names whose given name (token 0) is in the curated map — these
         read like genuine onboarding and screening-list records;
      2. if that pool is exhausted, synthetic names built from a curated gendered given
         name plus a surname drawn from the real source pool.

    Clients are interleaved masculine/feminine, so the emitted pairs are a controlled
    ~50/50 mix of 'male client -> female hit' and 'female client -> male hit'. The
    caller over-provisions when pair collisions are possible.
    """
    if count <= 0:
        return []

    masculine: List[str] = []
    feminine: List[str] = []
    for name in dict.fromkeys(source_names):
        gender = _gendered_token_gender(name)
        if gender == "masculine":
            masculine.append(name)
        elif gender == "feminine":
            feminine.append(name)
    random.shuffle(masculine)
    random.shuffle(feminine)

    surnames = sorted({n.split()[-1] for n in source_names})

    picks: List[str] = []
    used: Set[str] = set()

    def _synthesise(gender_wanted: str) -> Optional[str]:
        """Builds one synthetic client carrying a gendered given name of the wanted gender."""
        if not surnames:
            return None
        for _ in range(50):
            male, female = random.choice(GENDER_VARIANT_NAME_PAIRS)
            given = male if gender_wanted == "masculine" else female
            surname = random.choice(surnames)
            candidate = f"{given} {surname}"
            if _fold_token(surname) == _fold_token(given) or candidate in used:
                continue
            used.add(candidate)
            return candidate
        return None

    # Alternate the two gender slots so the mix stays ~50/50. When one side's real pool is
    # exhausted — the sanctions pool is heavily male-skewed — that side is topped up
    # synthetically instead of letting the other side dominate the quota.
    while len(picks) < count:
        progressed = False
        for pool, gender in ((masculine, "masculine"), (feminine, "feminine")):
            if len(picks) >= count:
                break
            if pool:
                name = pool.pop()
                used.add(name)
                picks.append(name)
                progressed = True
            else:
                synthetic = _synthesise(gender)
                if synthetic:
                    picks.append(synthetic)
                    progressed = True
        if not progressed:
            break

    if len(picks) < count:
        raise ValueError(
            f"Could not source {count} gender-variant client names "
            f"({len(GENDER_VARIANT_NAME_PAIRS)} curated pairs x {len(surnames)} surnames available)."
        )
    return picks


# ==============================================================================
# PAIR UNIQUENESS (SHARED BY BOTH PIPELINES)
# ==============================================================================

def build_unique_pipeline_a_pair(
    source_names: List[str],
    target_decision: str,
    seen_pairs: Set[Tuple[str, str]],
    metrics: Dict[str, Any],
    k_max: int = 3,
    max_attempts: int = 40
) -> Tuple[str, str, str, str, str]:
    """
    Builds ONE globally unique (client_name, hit_name) pair for the requested decision and
    registers it in `seen_pairs` before returning.

    Both pipelines need the same guarantee, so they share this builder:
      - the client name is redrawn on every attempt, because holding one name fixed can
        exhaust its mutation space and leave the caller with no pair at all;
      - single mutations are tried first and `double_mutation` kicks in after `k_max`
        attempts, widening the search space when the pool is reused heavily;
      - every collision is tallied in metrics['collision_retries'];
      - the returned pair is ALREADY registered, so no caller can emit a duplicate and
        the requested record counts are always met.

    Raises RuntimeError when `max_attempts` consecutive draws all collide. Failing loudly
    is deliberate: silently dropping the pair would shorten the dataset, and silently
    emitting it would duplicate a row — both corrupt the fine-tuning set.
    Returns (client_name, hit_name, matching_text, disposition_code, thinking).
    """
    if not source_names:
        raise ValueError("Cannot generate a screening alert from an empty source name pool.")

    for attempt in range(max_attempts):
        client_name = random.choice(source_names)
        hit_name, matching_text, disposition_code, thinking = generate_pipeline_a_sample(
            client_name, target_decision, double_mutation=(attempt > k_max)
        )
        pair_key = (client_name.strip().lower(), hit_name.strip().lower())
        if pair_key in seen_pairs:
            metrics["collision_retries"] += 1
            continue
        seen_pairs.add(pair_key)
        return client_name, hit_name, matching_text, disposition_code, thinking

    raise RuntimeError(
        f"Could not build a unique {target_decision} pair after {max_attempts} attempts from a "
        f"pool of {len(source_names)} names ({len(seen_pairs)} pairs already used). Enlarge the "
        f"source name pool (or keep --augment_diversity enabled) and re-run."
    )


# ==============================================================================
# ORCHESTRATION & COLLISION HANDLING
# ==============================================================================

async def run_hybrid_generation(
    source_names: List[str],
    num_tp: int,
    num_fp: int,
    ollama_model: str,
    batch_size: int,
    alpha: float = 0.90,
    k_max: int = 3,
    temp_dir: str = "temp",
    num_exact_matches: int = 0,
    num_gender_variants: int = 0,
    ollama_enabled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coordinates Pipeline A and Pipeline B generation, enforces global pair uniqueness,
    feeds generated candidate pairs to Ollama for human-like cognitive evaluation,
    and maintains strict data integrity with fixed-vocabulary disposition codes.

    Reserved categories are carved OUT of the decision budgets, so the requested total
    (num_tp + num_fp) is still produced exactly:
      - num_exact_matches  : rule-based TRUE POSITIVES (hit_name == client_name),
                             fully deterministic, zero LLM calls
      - num_gender_variants: FALSE POSITIVES built as a near-identical cross-gender pair
                             (e.g. 'Daniel Martin' / 'Daniela Martin'); LLM-narrated when
                             ollama_enabled, otherwise emitted with their rule-based
                             narrative
    Requesting more reserved records than the matching budget raises ValueError; the
    counts are never silently clamped.

    alpha is the PROGRAMMATIC share of the remaining budgets (the CLI passes
    fuzzy_pct / 100), and k_max is how many single-mutation draws build_unique_pipeline_a_pair
    attempts per pair before it escalates to a double mutation.
    """
    start_time = time.time()

    # Reserved categories are carved OUT of the decision budgets, so the grand total
    # stays exactly num_tp + num_fp. Mis-configuration fails loudly here as well, so a
    # direct API caller can never receive a silently shortchanged dataset.
    if num_exact_matches < 0 or num_gender_variants < 0:
        raise ValueError("Reserved category counts cannot be negative.")
    if num_exact_matches > num_tp:
        raise ValueError(
            f"Requested {num_exact_matches} exact-match TP alerts but only {num_tp} TPs are budgeted."
        )
    if num_gender_variants > num_fp:
        raise ValueError(
            f"Requested {num_gender_variants} cross-gender FP alerts but only {num_fp} FPs are budgeted."
        )
    remaining_tp = num_tp - num_exact_matches
    remaining_fp = num_fp - num_gender_variants

    # Calculate allocations over the remaining (non-reserved) budget
    n_tp_a = math.floor(alpha * remaining_tp)
    n_fp_a = math.floor(alpha * remaining_fp)
    n_tp_b = remaining_tp - n_tp_a
    n_fp_b = remaining_fp - n_fp_a

    records: List[Dict[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    override_counter: Dict[str, int] = {}

    metrics = {
        "num_tp_target": num_tp,
        "num_fp_target": num_fp,
        "num_exact_matches": num_exact_matches,
        "num_gender_variants": num_gender_variants,
        "exact_match_count": 0,
        "gender_variant_count": 0,
        "n_tp_a": n_tp_a,
        "n_fp_a": n_fp_a,
        "n_tp_b": n_tp_b,
        "n_fp_b": n_fp_b,
        "pipeline_a_count": 0,
        "pipeline_b_count": 0,
        "rule_based_fallbacks": 0,
        "collision_retries": 0,
        "ollama_failures": 0,
        "reserved_ollama_fallbacks": 0,
    }

    # PIPELINE A: Programmatic Generation with Human Cognitive Reasoning
    # --------------------------------------------------------------------------
    if n_tp_a + n_fp_a + num_exact_matches > 0:
        print(
            f"\n[Pipeline A] Generating {n_tp_a} TPs, {n_fp_a} FPs and "
            f"{num_exact_matches} exact-match TPs with human cognitive reasoning..."
        )
        tasks_a = [("TP", DECISION_TP)] * n_tp_a + [("FP", DECISION_FP)] * n_fp_a
        random.shuffle(tasks_a)

        temp_csv_path = get_temp_csv_path(temp_dir)
        batch_records = []
        batch_size_write = 250
        is_first_batch = True

        pbar_a = tqdm(total=len(tasks_a) + num_exact_matches, desc="Pipeline A (Human-Emulating Engine)", unit="records")

        # Reserved EXACT MATCH true positives: a fixed template, no LLM call, and the
        # only records in the dataset whose hit_name is identical to the client_name.
        for client_name in build_reserved_exact_match_names(source_names, num_exact_matches):
            hit_name, matching_text, disposition_code, thinking = apply_exact_name_match(client_name)
            seen_pairs.add((client_name.strip().lower(), hit_name.strip().lower()))
            record = {
                "client_name": client_name,
                "hit_name": hit_name,
                "matching_text": matching_text,
                "disposition_code": disposition_code,
                "thinking": thinking,
                "decision": DECISION_TP,
                "created_by": "rule_based"
            }
            batch_records.append(record)
            records.append(record)
            metrics["pipeline_a_count"] += 1
            metrics["exact_match_count"] += 1

            # Write batch to temp CSV every 250 records
            if len(batch_records) >= batch_size_write:
                write_records_batch_to_csv(batch_records, temp_csv_path, is_first_batch)
                is_first_batch = False
                batch_records = []

            pbar_a.update(1)

        for _, decision in tasks_a:
            # The shared builder redraws the client name on every attempt, escalates to a
            # double mutation after k_max collisions and registers the pair, so this loop
            # can neither emit a duplicate nor come up short of the requested count.
            client_name, hit_name, matching_text, disposition_code, thinking = build_unique_pipeline_a_pair(
                source_names, decision, seen_pairs, metrics, k_max=k_max
            )

            record = {
                "client_name": client_name,
                "hit_name": hit_name,
                "matching_text": matching_text,
                "disposition_code": disposition_code,
                "thinking": thinking,
                "decision": decision,
                "created_by": "rule_based"
            }
            batch_records.append(record)
            records.append(record)
            metrics["pipeline_a_count"] += 1
            
            # Write batch to temp CSV every 250 records
            if len(batch_records) >= batch_size_write:
                write_records_batch_to_csv(batch_records, temp_csv_path, is_first_batch)
                is_first_batch = False
                batch_records = []
            
            pbar_a.update(1)
        
        # Write remaining Pipeline A records
        if batch_records:
            write_records_batch_to_csv(batch_records, temp_csv_path, is_first_batch)
        pbar_a.close()

    # --------------------------------------------------------------------------
    # PIPELINE B: Feeding Generated Pairs to Ollama for Human Review
    # --------------------------------------------------------------------------
    total_b_samples = n_tp_b + n_fp_b + num_gender_variants
    if total_b_samples > 0:
        if ollama_enabled:
            print(
                f"\n[Pipeline B] Submitting {n_tp_b} TPs, {n_fp_b} FPs and "
                f"{num_gender_variants} cross-gender FPs to Ollama ('{ollama_model}') for human cognitive review..."
            )
            ollama_client = ollama.AsyncClient()
            eff_concurrency = min(batch_size, 2) if "cloud" in ollama_model.lower() else batch_size
            semaphore = asyncio.Semaphore(eff_concurrency)
        else:
            # Offline mode (--ollama_percentage 0): no client is created and no model call
            # is made — every queued pair below is emitted with its programmatic narrative.
            print(
                f"\n[Pipeline B] Offline mode: emitting {total_b_samples} queued pairs "
                f"({num_gender_variants} cross-gender FPs) with rule-based narratives (no LLM calls)."
            )
            ollama_client = None
            semaphore = None

        # Pre-generate the candidate pairs for Ollama review. Each item is
        # (client_name, hit_name, matching_text, disposition_code, thinking, target_decision,
        #  enforce_disposition_code, enforce_decision). The enforce_* entries are None for
        # generic alerts and pin the label for reserved categories.
        fuzzy_items_b: List[Tuple[str, str, str, str, str, str, Optional[str], Optional[str]]] = []

        for _ in range(n_tp_b):
            c_name, h_name, m_text, d_code, th = build_unique_pipeline_a_pair(
                source_names, DECISION_TP, seen_pairs, metrics, k_max=k_max
            )
            fuzzy_items_b.append((c_name, h_name, m_text, d_code, th, DECISION_TP, None, None))

        for _ in range(n_fp_b):
            c_name, h_name, m_text, d_code, th = build_unique_pipeline_a_pair(
                source_names, DECISION_FP, seen_pairs, metrics, k_max=k_max
            )
            fuzzy_items_b.append((c_name, h_name, m_text, d_code, th, DECISION_FP, None, None))

        # Reserved CROSS-GENDER false positives: the pair is built deterministically and
        # Ollama narrates why two near-identical names are nevertheless two different
        # people. The candidate list is over-provisioned so that a pair collision can
        # never shorten the requested quota.
        if num_gender_variants > 0:
            over_provisioned = num_gender_variants + max(5, num_gender_variants // 5)
            for c_name in build_reserved_gender_variant_clients(source_names, over_provisioned):
                if metrics["gender_variant_count"] >= num_gender_variants:
                    break
                built = build_gender_variant_pair(c_name)
                if built is None:
                    continue
                h_name, m_text, d_code, th = built
                key = (c_name.strip().lower(), h_name.strip().lower())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                fuzzy_items_b.append((
                    c_name, h_name, m_text, d_code, th, DECISION_FP,
                    "DISP_GIVEN_NAME_GENDER_VARIANT", DECISION_FP
                ))
                metrics["gender_variant_count"] += 1

        random.shuffle(fuzzy_items_b)
        temp_csv_path = get_temp_csv_path(temp_dir)
        batch_records_b = []
        batch_size_write = 250
        is_first_batch_b = not Path(temp_csv_path).exists()  # Check if file already has Pipeline A data

        pbar_b = tqdm(
            total=len(fuzzy_items_b),
            desc="Pipeline B (Ollama Human Review)" if ollama_enabled else "Pipeline B (Offline rule-based)",
            unit="records"
        )

        async def process_ollama_fuzzy_item(
            c_name: str, h_name: str, fallback_match: str, fallback_d_code: str, fallback_th: str, tgt_dec: str,
            enforce_code: Optional[str] = None, enforce_dec: Optional[str] = None
        ):
            nonlocal metrics, batch_records_b, is_first_batch_b
            alert: Optional[ScreeningAlert] = None
            if ollama_enabled:
                try:
                    alert = await generate_single_ollama_sample(
                        ollama_client, ollama_model, c_name, h_name, tgt_dec, fallback_d_code, semaphore,
                        enforce_disposition_code=enforce_code, enforce_decision=enforce_dec,
                        override_counter=override_counter
                    )
                except Exception as exc:
                    # A contract violation (e.g. an identical pair reaching the LLM path) or
                    # any unexpected client error must degrade THIS pair to its rule-based
                    # record — never abort the whole gather() run.
                    print(f"\n[Pipeline B] LLM review skipped for '{c_name}' / '{h_name}': {exc}")
                    alert = None
            if alert and alert.thinking and alert.decision:
                record = {
                    "client_name": c_name,
                    "hit_name": h_name,
                    "matching_text": alert.matching_text or fallback_match,
                    "disposition_code": alert.disposition_code or fallback_d_code,
                    "thinking": alert.thinking,
                    "decision": alert.decision,
                    "created_by": "ollama"
                }
                metrics["pipeline_b_count"] += 1
            else:
                if ollama_enabled:
                    metrics["ollama_failures"] += 1
                    if enforce_code is not None:
                        # A reserved category the model could not narrate keeps its category
                        # label and falls back to the programmatic narrative.
                        metrics["reserved_ollama_fallbacks"] += 1
                metrics["rule_based_fallbacks"] += 1
                record = {
                    "client_name": c_name,
                    "hit_name": h_name,
                    "matching_text": fallback_match,
                    "disposition_code": fallback_d_code,
                    "thinking": fallback_th,
                    "decision": tgt_dec,
                    "created_by": "rule_based"
                }
            
            batch_records_b.append(record)
            records.append(record)
            
            # Write batch to temp CSV every 250 records
            if len(batch_records_b) >= batch_size_write:
                write_records_batch_to_csv(batch_records_b, temp_csv_path, is_first_batch_b)
                is_first_batch_b = False
                batch_records_b = []
            
            pbar_b.update(1)

        await asyncio.gather(*(process_ollama_fuzzy_item(*item) for item in fuzzy_items_b))
        
        # Write remaining Pipeline B records
        if batch_records_b:
            write_records_batch_to_csv(batch_records_b, temp_csv_path, is_first_batch_b)
        
        pbar_b.close()

    elapsed = time.time() - start_time
    metrics["total_elapsed_seconds"] = elapsed
    metrics["records_per_second"] = len(records) / elapsed if elapsed > 0 else 0.0
    # Enforcement tally: 'plausibility' counts any alert whose exact-match code had to be
    # rejected, while 'disposition_code' / 'decision' count reserved-category labels that
    # had to be pinned over what the model returned.
    metrics["enforcement_overrides"] = dict(override_counter)

    df_out = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    # Derive disposition_label deterministically from the controlled vocabulary dict
    df_out["disposition_label"] = df_out["disposition_code"].map(
        lambda c: DISPOSITION_CODES.get(c, "Unknown disposition code")
    )
    # Re-order columns strictly according to specification (keeps an empty frame usable too)
    df_out = df_out[OUTPUT_COLUMNS]
    return df_out, metrics

# ==============================================================================
# PRE-FLIGHT VERIFICATION
# ==============================================================================

async def verify_ollama_model(model_tag: str) -> str:
    """
    Verifies that Ollama is accessible and checks for the existence of the
    requested model tag, matching case-insensitively or normalizing hyphens/spaces.
    Returns the exact normalized model name.
    """
    client = ollama.AsyncClient()
    try:
        model_list = await client.list()
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama daemon (127.0.0.1:11434). Please ensure Ollama is running.\nError: {e}"
        )

    available_models = []
    for m in model_list.get("models", []):
        name = m.get("model") or m.get("name")
        if name:
            available_models.append(name)

    # Normalize tag for comparison (e.g. 'gemma4:31b Cloud' vs 'gemma4:31b-cloud')
    norm_target = model_tag.strip().lower().replace(" ", "-")

    for avail in available_models:
        if avail.strip().lower().replace(" ", "-") == norm_target:
            return avail
        if ":" in avail and ":" in norm_target:
            if avail.split(":")[0] == norm_target.split(":")[0]:
                return avail

    for avail in available_models:
        if norm_target in avail.lower():
            return avail

    raise ValueError(
        f"Requested model '{model_tag}' not found in Ollama.\n"
        f"Available models: {available_models}\n"
        f"Please run `ollama pull {model_tag}` or specify an available model with --ollama_model."
    )

# ==============================================================================
# TEMP FOLDER & INCREMENTAL CSV MANAGEMENT
# ==============================================================================

def setup_temp_folder(temp_dir: str = "temp") -> str:
    """
    Create temp folder and initialize .gitignore file.
    Returns the temp directory path.
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    
    gitignore_path = temp_path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("# Temporary files for incremental CSV writing\n*.csv\n")
        print(f"[Setup] Created {gitignore_path}")
    
    return temp_dir

def get_temp_csv_path(temp_dir: str = "temp") -> str:
    """Get path to incremental CSV in temp folder."""
    return str(Path(temp_dir) / "synthetic_alerts_incremental.csv")

def write_records_batch_to_csv(
    records: List[Dict[str, str]], 
    csv_path: str, 
    is_first_batch: bool = False
) -> None:
    """
    Append batch of records to CSV file with proper formatting.
    
    Args:
        records: List of alert records to write
        csv_path: Path to CSV file
        is_first_batch: If True, write headers; if False, append without headers
    """
    if not records:
        return
    
    df_batch = pd.DataFrame(records)
    df_batch["disposition_label"] = df_batch["disposition_code"].map(
        lambda c: DISPOSITION_CODES.get(c, "Unknown disposition code")
    )
    # Ensure column order (shared constant, so the frame, the incremental file and the
    # exported dataset can never drift apart)
    df_batch = df_batch[OUTPUT_COLUMNS]
    
    mode = 'w' if is_first_batch else 'a'
    header = is_first_batch
    df_batch.to_csv(csv_path, mode=mode, header=header, index=False)

def cleanup_temp_csv(temp_dir: str = "temp", quiet: bool = False) -> bool:
    """
    Remove only the incremental CSV, keep .gitignore for future runs.

    Called with quiet=True at the start of a run, so a stale file left behind by an
    earlier (possibly crashed) run can never be appended to and then merged into this
    run's output. Returns True when a file was actually removed.
    """
    temp_csv = Path(temp_dir) / "synthetic_alerts_incremental.csv"
    if temp_csv.exists():
        temp_csv.unlink()
        if not quiet:
            print(f"[Cleanup] Removed temporary CSV: {temp_csv}")
        return True
    return False

# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================

def resolve_counts_and_percentages(args) -> Tuple[int, int, float, float, float, float, int, int]:
    """
    Computes (num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct,
    num_exact_matches, num_gender_variants) from command-line arguments, supporting
    both explicit counts and percentages.

    The two reserved-category quotas are resolved as a percentage of the requested
    total (or as explicit counts) and validated against the TP/FP budgets up front, so
    a mis-configuration fails loudly here instead of silently shortening the dataset.
    """
    # 1. Resolve Generation Source Split (% Ollama vs % Fuzzy)
    if args.ollama_percentage is not None and args.fuzzy_percentage is not None:
        total_source = args.ollama_percentage + args.fuzzy_percentage
        if abs(total_source - 100.0) > 0.01:
            raise ValueError(f"Ollama % ({args.ollama_percentage}) and Fuzzy % ({args.fuzzy_percentage}) must sum to 100%.")
        ollama_pct = float(args.ollama_percentage)
        fuzzy_pct = float(args.fuzzy_percentage)
    elif args.ollama_percentage is not None:
        ollama_pct = float(args.ollama_percentage)
        fuzzy_pct = 100.0 - ollama_pct
    elif args.fuzzy_percentage is not None:
        fuzzy_pct = float(args.fuzzy_percentage)
        ollama_pct = 100.0 - fuzzy_pct
    else:
        # Default: 10% Ollama, 90% Fuzzy
        fuzzy_pct = 90.0
        ollama_pct = 10.0

    if not (0.0 <= ollama_pct <= 100.0 and 0.0 <= fuzzy_pct <= 100.0):
        raise ValueError(f"Percentages must be between 0 and 100. Got Ollama: {ollama_pct}%, Fuzzy: {fuzzy_pct}%.")

    # 2. Resolve Decision Balance (TP % vs FP % and total counts).
    # Explicit counts are resolved FIRST: --tp_percentage / --fp_percentage carry non-None
    # defaults (40 / 60), so testing the percentages first made -tp / -fp dead code and
    # silently ignored the counts a caller asked for.
    explicit_tp = args.num_true_positives
    explicit_fp = args.num_false_positives

    if explicit_tp is not None and explicit_tp < 0:
        raise ValueError(f"--num_true_positives must be >= 0. Got {explicit_tp}.")
    if explicit_fp is not None and explicit_fp < 0:
        raise ValueError(f"--num_false_positives must be >= 0. Got {explicit_fp}.")

    if explicit_tp is not None and explicit_fp is not None:
        if args.total_samples is not None and args.total_samples != explicit_tp + explicit_fp:
            print(
                f"[Warning] Ignoring --total_samples {args.total_samples}: the explicit counts "
                f"-tp {explicit_tp} + -fp {explicit_fp} define the dataset size ({explicit_tp + explicit_fp})."
            )
        num_tp = explicit_tp
        num_fp = explicit_fp
    else:
        total_budget = args.total_samples if args.total_samples is not None else 100
        if total_budget < 0:
            raise ValueError(f"--total_samples must be >= 0. Got {total_budget}.")

        if explicit_tp is not None:
            # A single explicit count wins over the percentages and the other side fills the
            # --total_samples budget (100 when not supplied) — never silently clamped.
            if explicit_tp > total_budget:
                raise ValueError(
                    f"--num_true_positives {explicit_tp} exceeds --total_samples {total_budget}."
                )
            num_tp = explicit_tp
            num_fp = total_budget - num_tp
            print(f"[Warning] -tp given explicitly, so percentages are ignored: TP {num_tp} / FP {num_fp}.")
        elif explicit_fp is not None:
            if explicit_fp > total_budget:
                raise ValueError(
                    f"--num_false_positives {explicit_fp} exceeds --total_samples {total_budget}."
                )
            num_fp = explicit_fp
            num_tp = total_budget - num_fp
            print(f"[Warning] -fp given explicitly, so percentages are ignored: TP {num_tp} / FP {num_fp}.")
        else:
            if args.tp_percentage is not None and args.fp_percentage is not None:
                if abs(args.tp_percentage + args.fp_percentage - 100.0) > 0.01:
                    raise ValueError(f"TP % ({args.tp_percentage}) and FP % ({args.fp_percentage}) must sum to 100%.")
                tp_pct = float(args.tp_percentage)
                fp_pct = float(args.fp_percentage)
            elif args.tp_percentage is not None:
                tp_pct = float(args.tp_percentage)
                fp_pct = 100.0 - tp_pct
            elif args.fp_percentage is not None:
                fp_pct = float(args.fp_percentage)
                tp_pct = 100.0 - fp_pct
            else:
                # Documented default split
                tp_pct, fp_pct = 40.0, 60.0

            if not (0.0 <= tp_pct <= 100.0 and 0.0 <= fp_pct <= 100.0):
                raise ValueError(f"TP/FP percentages must be between 0 and 100. Got TP: {tp_pct}%, FP: {fp_pct}%.")

            num_tp = round((tp_pct / 100.0) * total_budget)
            num_fp = total_budget - num_tp

    # Percentages reported to the caller are always derived from the resolved counts, so the
    # printed balance can never disagree with the data that is actually generated.
    total = num_tp + num_fp
    tp_pct = (num_tp / total * 100.0) if total > 0 else 0.0
    fp_pct = (num_fp / total * 100.0) if total > 0 else 0.0

    # 3. Resolve reserved category quotas (cross-gender FP and exact-match TP)
    total = num_tp + num_fp
    if total <= 0:
        raise ValueError(
            "Nothing to generate: the resolved TP/FP counts are both 0. Raise --total_samples, "
            "--tp_percentage or --fp_percentage."
        )
    if args.num_exact_matches is not None:
        num_exact_matches = int(args.num_exact_matches)
    else:
        num_exact_matches = round((float(args.exact_match_percentage) / 100.0) * total)

    if args.num_gender_variants is not None:
        num_gender_variants = int(args.num_gender_variants)
    else:
        num_gender_variants = round((float(args.gender_variant_percentage) / 100.0) * total)

    if num_exact_matches < 0 or num_gender_variants < 0:
        raise ValueError(
            f"Reserved category counts cannot be negative. Got exact-match TP: {num_exact_matches}, "
            f"cross-gender FP: {num_gender_variants}."
        )
    if num_exact_matches > num_tp:
        raise ValueError(
            f"Requested {num_exact_matches} exact-match TP alerts but only {num_tp} TPs are budgeted. "
            f"Lower --exact_match_pct/--num_exact_matches, or raise the TP share (--tp_percentage / -n)."
        )
    if num_gender_variants > num_fp:
        raise ValueError(
            f"Requested {num_gender_variants} cross-gender FP alerts but only {num_fp} FPs are budgeted. "
            f"Lower --gender_variant_pct/--num_gender_variants, or raise the FP share (--fp_percentage / -n)."
        )

    return num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct, num_exact_matches, num_gender_variants

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic individual name-screening alert dataset for SLM fine-tuning."
    )
    parser.add_argument(
        "--input_csv_path",
        type=str,
        default="sdn.csv",
        help="Path to input CSV containing source individual names (column header: name or HitName)."
    )
    parser.add_argument(
        "--total_samples", "-n",
        type=int,
        default=None,
        help="Total number of synthetic screening alerts to generate (default: 100 if counts not specified)."
    )
    parser.add_argument(
        "--tp_percentage", "--tp_pct",
        type=float,
        default=40,
        help="Target percentage for True Positive (TP) alerts (0.0 - 100.0, default: 40.0)."
    )
    parser.add_argument(
        "--fp_percentage", "--fp_pct",
        type=float,
        default=60,
        help="Target percentage for False Positive (FP) alerts (0.0 - 100.0, default: 60.0)."
    )
    parser.add_argument(
        "--num_true_positives", "-tp",
        type=int,
        default=None,
        help="Explicit integer count for True Positive (TP) alerts."
    )
    parser.add_argument(
        "--num_false_positives", "-fp",
        type=int,
        default=None,
        help="Explicit integer count for False Positive (FP) alerts."
    )
    parser.add_argument(
        "--gender_variant_percentage", "--gender_variant_pct",
        type=float,
        default=2.0,
        help="RESERVED: percentage of near-identical cross-gender False Positive alerts (e.g. 'Daniel Martin' vs "
             "'Daniela Martin'), carved out of the FP budget and narrated by Ollama (default: 2.0)."
    )
    parser.add_argument(
        "--num_gender_variants",
        type=int,
        default=None,
        help="RESERVED: explicit integer count of cross-gender FP alerts (overrides --gender_variant_percentage)."
    )
    parser.add_argument(
        "--exact_match_percentage", "--exact_match_pct",
        type=float,
        default=2.0,
        help="RESERVED: percentage of exact-match True Positive alerts where the hit name is identical to the client "
             "name, generated by rule with zero LLM calls, carved out of the TP budget (default: 2.0)."
    )
    parser.add_argument(
        "--num_exact_matches",
        type=int,
        default=None,
        help="RESERVED: explicit integer count of exact-match TP alerts (overrides --exact_match_percentage)."
    )
    parser.add_argument(
        "--ollama_percentage", "--ollama_pct",
        type=float,
        default=70,
        help="Target percentage of alerts generated via Ollama LLM (0.0 - 100.0, default: 70.0)."
    )
    parser.add_argument(
        "--fuzzy_percentage", "--fuzzy_pct",
        type=float,
        default=30,
        help="Target percentage of alerts generated via RapidFuzz logic (0.0 - 100.0, default: 30.0)."
    )
    parser.add_argument(
        "--augment_diversity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment source names with diverse global multi-ethnic name pools (East Asian, South Asian, African, MENA, Slavic, European, Hispanic). Default: True."
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="gemma4:31b Cloud",
        help="Ollama model tag to use for Pipeline B (default: 'gemma4:31b Cloud')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of concurrent requests for Ollama batched execution (default: 20)."
    )
    parser.add_argument(
        "--output_csv_path", "-o",
        type=str,
        default="synthetic_screening_alerts.csv",
        help="Path to save final merged CSV dataset (default: 'synthetic_screening_alerts.csv')."
    )
    parser.add_argument(
        "--name_column",
        type=str,
        default=None,
        help="Specific column name in input CSV containing names (optional, auto-detected by default)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling and mutations."
    )
    return parser.parse_args()

def estimate_generation_time(num_records: int, fuzzy_pct: float, ollama_pct: float, batch_size: int) -> Tuple[float, str]:
    """
    Estimates the total generation time based on pipeline configuration.
    
    Returns:
        (estimated_seconds, human_readable_string)
    
    Assumptions:
        - Pipeline A (RapidFuzz): ~20,000 records/sec
        - Pipeline B (Ollama): ~10-15 records/sec per concurrent batch
    """
    # Split records by pipeline
    num_a = num_records * (fuzzy_pct / 100.0)
    num_b = num_records * (ollama_pct / 100.0)
    
    # Throughput estimates (records per second)
    pipeline_a_throughput = 20000.0  # RapidFuzz is very fast
    pipeline_b_throughput = max(5.0, min(15.0, batch_size * 0.8))  # Ollama: ~0.8 rec/sec per concurrent request
    
    # Calculate time per pipeline
    time_a = num_a / pipeline_a_throughput if pipeline_a_throughput > 0 else 0
    time_b = num_b / pipeline_b_throughput if pipeline_b_throughput > 0 else 0
    
    total_seconds = time_a + time_b
    
    # Format human-readable output
    if total_seconds < 1:
        time_str = f"{total_seconds:.1f}s"
    elif total_seconds < 60:
        time_str = f"{total_seconds:.1f}s"
    elif total_seconds < 3600:
        minutes = total_seconds / 60.0
        time_str = f"{minutes:.1f}m"
    else:
        hours = total_seconds / 3600.0
        minutes = (total_seconds % 3600) / 60.0
        time_str = f"{hours:.1f}h {minutes:.0f}m"
    
    return total_seconds, time_str

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 0. Setup temp folder with .gitignore, and clear any incremental CSV left behind by an
    # earlier (possibly crashed) run: otherwise that stale data would be appended to and then
    # merged into this run's output, inflating the dataset with rows from a different run.
    temp_dir = setup_temp_folder()
    if cleanup_temp_csv(temp_dir, quiet=True):
        print("[Setup] Cleared a stale incremental CSV from a previous run")

    # Resolve counts and percentages
    (num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct,
     num_exact_matches, num_gender_variants) = resolve_counts_and_percentages(args)
    total_target = num_tp + num_fp
    alpha = fuzzy_pct / 100.0

    # Reserved categories are carved out of the decision budgets, so Pipeline A emits the
    # exact matches plus its alpha share of the remaining budgets, while everything else
    # (Pipeline B plus every reserved cross-gender pair) is LLM-narrated.
    pipeline_a_generic = (
        math.floor(alpha * (num_tp - num_exact_matches)) + math.floor(alpha * (num_fp - num_gender_variants))
    )
    llm_records = total_target - num_exact_matches - pipeline_a_generic
    effective_ollama_pct = (llm_records / total_target * 100.0) if total_target > 0 else ollama_pct

    print("=" * 75)
    print("      CONFIGURABLE SYNTHETIC SCREENING ALERT DATA GENERATOR")
    print("=" * 75)
    print(f"Input CSV:               {args.input_csv_path}")
    print(f"Output CSV:              {args.output_csv_path}")
    print(f"Total Target Size:       {total_target} alerts")
    print(f"Decision Balance:        TP: {num_tp} ({tp_pct:.1f}%) | FP: {num_fp} ({fp_pct:.1f}%)")
    print(f"Generation Split:        Fuzzy Match: {fuzzy_pct:.1f}% | Ollama: {ollama_pct:.1f}%")
    print(f"Reserved Categories:     {num_gender_variants} cross-gender FP (Ollama) | {num_exact_matches} exact-match TP (rule-based)")
    print(f"Reserved Budget:         carved out of the TP/FP targets above - total size unchanged")
    print(f"Multi-Ethnic Diversity:  {'ENABLED (Global Pools Included)' if args.augment_diversity else 'DISABLED'}")
    print(f"Ollama Concurrency:      Batch Size {args.batch_size}")
    print(f"Temp Folder:             {temp_dir}/")
    
    # Estimate time to completion using the effective LLM share (the reserved exact
    # matches are rule-based and add no generation cost)
    est_seconds, est_time_str = estimate_generation_time(
        total_target, 100.0 - effective_ollama_pct, effective_ollama_pct, args.batch_size
    )
    print(f"LLM Load:                {llm_records} LLM records ({effective_ollama_pct:.1f}%) | {total_target - llm_records} rule-based")
    print(f"Estimated Time:          ~{est_time_str}")
    print("=" * 75)

    # 1. Pre-flight verification
    # Records that reach Ollama: Pipeline B's share of the remaining budgets plus every
    # reserved cross-gender pair, which is always LLM-narrated even at --ollama_pct 0.
    resolved_model = args.ollama_model
    total_b_expected = llm_records

    if total_b_expected > 0:
        print("\n[Pre-flight] Verifying Ollama model availability...")
        try:
            resolved_model = asyncio.run(verify_ollama_model(args.ollama_model))
            print(f"[Pre-flight] Verified model '{resolved_model}' successfully.")
        except Exception as e:
            print(f"\n[Pre-flight Warning] {e}")
            if num_gender_variants > 0:
                # Reserved cross-gender pairs are always LLM-narrated, so an
                # unreachable model cannot silently degrade them to templates.
                # Re-run with --num_gender_variants 0 for a fully offline run.
                print(
                    "Reserved cross-gender pairs require a reachable Ollama model. "
                    "To proceed without Ollama, pass --num_gender_variants 0 as well."
                )
                sys.exit(1)
            print("To proceed without Ollama, pass --ollama_percentage 0 --fuzzy_percentage 100.")
            sys.exit(1)
    else:
        print("\n[Pre-flight] Pipeline B allocation is 0; skipping Ollama verification.")

    # 2. Load and sanitize input dataset
    print(f"\n[Loading Data] Reading source names from '{args.input_csv_path}'...")
    source_names = load_source_names(args.input_csv_path, args.name_column, augment_diversity=args.augment_diversity)
    print(f"[Loading Data] Pool contains {len(source_names)} individual names across global ethnic origins.")

    # 3. Execute Hybrid Generation
    df_results, metrics = asyncio.run(
        run_hybrid_generation(
            source_names=source_names,
            num_tp=num_tp,
            num_fp=num_fp,
            ollama_model=resolved_model,
            batch_size=args.batch_size,
            alpha=alpha,
            k_max=3,
            temp_dir=temp_dir,
            num_exact_matches=num_exact_matches,
            num_gender_variants=num_gender_variants,
            # Offline runs (no LLM records at all) must not create a client or attempt a
            # single call — this flag is what the Pipeline B block branches on.
            ollama_enabled=total_b_expected > 0
        )
    )

    # 4. Merge temp CSV with in-memory results
    temp_csv_path = get_temp_csv_path(temp_dir)
    if Path(temp_csv_path).exists():
        try:
            df_temp = pd.read_csv(temp_csv_path)
            # Combine temp CSV with final results, avoiding duplicates
            df_results = pd.concat([df_temp, df_results], ignore_index=True)
            df_results = df_results.drop_duplicates(subset=["client_name", "hit_name"], keep="first")
            print(f"[Merge] Loaded {len(df_temp)} records from temp CSV")
        except Exception as e:
            print(f"[Warning] Failed to merge temp CSV: {e}")

    # 5. Global Uniqueness and Integrity Verification
    total_records = len(df_results)
    denom = total_records if total_records > 0 else 1  # guards every percentage below
    unique_pairs = df_results[["client_name", "hit_name"]].drop_duplicates()
    is_strictly_unique = len(unique_pairs) == total_records

    # 6. Export to CSV
    df_results.to_csv(args.output_csv_path, index=False)
    print(f"\n[Export] Saved {len(df_results)} records to '{args.output_csv_path}'.")

    # 7. Cleanup temp CSV (keep .gitignore)
    cleanup_temp_csv(temp_dir)

    # 8. Print Execution Metrics Summary
    if total_records == 0:
        print("\n[Warning] No records were generated; skipping the metrics summary.")
        return

    print("\n" + "=" * 75)
    print("                       PIPELINE EXECUTION METRICS                        ")
    print("=" * 75)
    print(f"Total Alerts Generated:             {total_records}")
    print(f"Total Execution Time:               {metrics['total_elapsed_seconds']:.2f} seconds")
    print(f"Throughput:                         {metrics['records_per_second']:.2f} records/sec")
    print("-" * 75)
    p_a = metrics['pipeline_a_count']
    p_b = metrics['pipeline_b_count']
    p_fb = metrics['rule_based_fallbacks']
    print(f"Pipeline A Output (RapidFuzz):      {p_a} ({p_a/denom*100.0:.1f}%)")
    print(f"Pipeline B Output (Ollama):         {p_b} ({p_b/denom*100.0:.1f}%)")
    print(f"Rule-Based Fallbacks (Pipeline B):  {p_fb} ({p_fb/denom*100.0:.1f}%)")
    print(f"Pipeline B Collision Retries:       {metrics['collision_retries']}")
    print(f"Ollama Failures/Mismatches:         {metrics['ollama_failures']}")
    print("-" * 75)
    tp_count = (df_results['decision'] == DECISION_TP).sum()
    fp_count = (df_results['decision'] == DECISION_FP).sum()
    print(f"True Positives ('{DECISION_TP}'):    {tp_count} ({tp_count/denom*100.0:.1f}%)")
    print(f"False Positives ('{DECISION_FP}'):          {fp_count} ({fp_count/denom*100.0:.1f}%)")
    print(f"Strict Global Uniqueness:           {'PASSED (Zero Duplicates)' if is_strictly_unique else 'FAILED'}")
    print("-" * 75)

    # Integrity of the two reserved categories: an identical pair must always be the
    # exact-match true positive, and a near-identical cross-gender pair must always be
    # the false positive — never a same-person typo label.
    exact_rows = df_results[df_results["client_name"] == df_results["hit_name"]]
    gender_rows = df_results[df_results["disposition_code"] == "DISP_GIVEN_NAME_GENDER_VARIANT"]
    contradictory_rows = [
        r.client_name for r in df_results.itertuples()
        if r.disposition_code == "DISP_MINOR_TYPO" and is_gender_variant_pair(r.client_name, r.hit_name)
    ]
    category_problems: List[str] = []
    if len(exact_rows) != num_exact_matches:
        category_problems.append(f"exact-match rows {len(exact_rows)} != requested {num_exact_matches}")
    if len(exact_rows) and (
        (exact_rows["disposition_code"] != "DISP_EXACT_NAME_MATCH").any()
        or (exact_rows["decision"] != DECISION_TP).any()
    ):
        category_problems.append("an identical name pair is not labelled exact-match/TP")
    if len(gender_rows) != num_gender_variants:
        category_problems.append(f"cross-gender rows {len(gender_rows)} != requested {num_gender_variants}")
    if len(gender_rows) and (
        (gender_rows["decision"] != DECISION_FP).any()
        or not all(is_gender_variant_pair(r.client_name, r.hit_name) for r in gender_rows.itertuples())
    ):
        category_problems.append("a cross-gender row is not a genuine gendered pair labelled FP")
    if contradictory_rows:
        category_problems.append(f"{len(contradictory_rows)} DISP_MINOR_TYPO rows are gender-variant pairs")

    print(f"Category Integrity:                 {'PASSED' if not category_problems else 'FAILED'}")
    for problem in category_problems:
        print(f"  ! {problem}")
    print(
        f"Reserved Emitted (req/out):         cross-gender FP {num_gender_variants}/{len(gender_rows)} | "
        f"exact-match TP {num_exact_matches}/{len(exact_rows)}"
    )
    print(f"Reserved Ollama Fallbacks:          {metrics['reserved_ollama_fallbacks']}")
    overrides = metrics.get("enforcement_overrides", {})
    print(f"Label Enforcement Corrections:      {'none' if not overrides else ', '.join(f'{k}={v}' for k, v in sorted(overrides.items()))}")
    print("-" * 75)
    print("Disposition Code Distribution:")
    disp_counts = df_results['disposition_code'].value_counts().sort_index()
    for code, count in disp_counts.items():
        pct = (count / denom * 100.0)
        label = DISPOSITION_CODES.get(code, "Unknown")
        print(f"  {code:40s} {count:5d} ({pct:5.1f}%) — {label}")
    print("=" * 75)

if __name__ == "__main__":
    main()
