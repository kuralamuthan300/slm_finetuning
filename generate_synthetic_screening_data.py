#!/usr/bin/env python3
"""
generate_synthetic_screening_data.py

High-performance synthetic screening alert data generator for Small Language Model (SLM) training.
Simulates individual name-matching alerts from an AML/Sanctions client screening engine using a
90% Programmatic (RapidFuzz) / 10% Ollama structured generation architecture.

Output CSV Schema:
    client_name, hit_name, matching_text, thinking, decision
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
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
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
    # False Positive codes — distinct individual
    "DISP_SURNAME_CONFLICT":            "Surnames are entirely different names",
    "DISP_MIDDLE_NAME_CONFLICT":        "Middle names are irreconcilably different",
    "DISP_PHONETIC_SURNAME_DIVERGENCE": "Surnames phonetically shifted but distinct",
}

DISPOSITION_TP_CODES = [k for k in DISPOSITION_CODES if k not in (
    "DISP_SURNAME_CONFLICT", "DISP_MIDDLE_NAME_CONFLICT", "DISP_PHONETIC_SURNAME_DIVERGENCE"
)]
DISPOSITION_FP_CODES = [
    "DISP_SURNAME_CONFLICT", "DISP_MIDDLE_NAME_CONFLICT", "DISP_PHONETIC_SURNAME_DIVERGENCE"
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
       DISP_SURNAME_CONFLICT         — Surnames are entirely different names
       DISP_MIDDLE_NAME_CONFLICT     — Middle names are irreconcilably different
       DISP_PHONETIC_SURNAME_DIVERGENCE — Surnames phonetically shifted but distinct
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

def apply_minor_typo(name: str) -> Tuple[str, str, str]:
    """
    Applies Levenshtein distance <= 2 typo mutation (character transposition,
    substitution, or single omission).
    Returns (hit_name, matching_text, thinking).
    """
    tokens = name.split()
    chosen_token_idx = random.randint(0, len(tokens) - 1)
    orig_token = tokens[chosen_token_idx]

    if len(orig_token) > 3:
        token_chars = list(orig_token)
        mutation_kind = random.choice(["transpose", "substitute", "omit"])
        idx = random.randint(1, len(token_chars) - 2)

        if mutation_kind == "transpose":
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

        tokens[chosen_token_idx] = "".join(token_chars)
    else:
        # Fallback substitution
        tokens[chosen_token_idx] = orig_token + "n"
        desc = "single typographical character append"

    hit_name = " ".join(tokens)

    # Overlapping text: unaffected tokens
    unaffected = [t for i, t in enumerate(name.split()) if i != chosen_token_idx]
    matching_text = " ".join(unaffected) if unaffected else name[:3]

    thinking = (
        f"Comparing the name strings: the primary identifying words '{matching_text}' match identically. "
        f"The candidate hit has a minor clerical difference in word '{orig_token}' (written as '{tokens[chosen_token_idx]}'), "
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
            shifted = random.choice(ALTERNATIVE_SURNAMES)
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

    return hit_name, matching_text, disposition_code, thinking


# ==============================================================================
# PIPELINE B: OLLAMA ASYNC STRUCTURED GENERATION
# ==============================================================================

def clean_ollama_json_response(
    content: str, client_name: str = "", hit_name: str = "", target_decision: str = "", fallback_disposition_code: str = ""
) -> Dict[str, Any]:
    """
    Robustly parses raw text from LLM, stripping markdown code blocks,
    recovering JSON substrings, and normalizing disposition_code against
    the fixed 11-code controlled vocabulary.
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

    # Normalize decision
    dec = str(data.get("decision", "")).strip().lower()
    if "escalat" in dec or "true" in dec:
        data["decision"] = DECISION_TP
    elif "disqualif" in dec or "false" in dec or "reject" in dec:
        data["decision"] = DECISION_FP
    else:
        data["decision"] = target_decision or DECISION_TP

    # Normalize disposition_code — validate against the controlled vocabulary
    raw_code = str(data.get("disposition_code") or data.get("key_reason") or "").strip()
    # Try exact match first
    if raw_code in DISPOSITION_CODES:
        data["disposition_code"] = raw_code
    else:
        # Try case-insensitive and prefix match
        raw_upper = raw_code.upper()
        matched = next((k for k in DISPOSITION_CODES if k == raw_upper or raw_upper in k), None)
        if matched:
            data["disposition_code"] = matched
        elif fallback_disposition_code and fallback_disposition_code in DISPOSITION_CODES:
            # Use the programmatically generated fallback code
            data["disposition_code"] = fallback_disposition_code
        else:
            # Last resort: infer from decision
            data["disposition_code"] = (
                "DISP_MINOR_TYPO" if data["decision"] == DECISION_TP else "DISP_SURNAME_CONFLICT"
            )

    # Normalize matching_text
    if "matching_text" not in data or not data["matching_text"]:
        for alt in ["matching_tokens", "matched_text", "match", "overlap"]:
            if alt in data and data[alt]:
                data["matching_text"] = str(data[alt])
                break
        # Fallback: compute overlapping tokens between client_name and hit_name
        if "matching_text" not in data or not data["matching_text"]:
            client_toks = set(re.findall(r"\b\w+\b", client_name.lower()))
            common = [t for t in re.findall(r"\b\w+\b", hit_name) if t.lower() in client_toks]
            data["matching_text"] = " ".join(common) if common else hit_name

    return data

async def generate_single_ollama_sample(
    client: ollama.AsyncClient,
    model_name: str,
    client_name: str,
    hit_name: str,
    target_decision: str,
    fallback_disposition_code: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4
) -> Optional[ScreeningAlert]:
    """
    Submits a fuzzy-match generated candidate pair (client_name, hit_name) to Ollama.
    The model acts as a human compliance screening analyst, producing natural cognitive
    thinking (no math formulas), a fixed-vocabulary disposition_code, and the screening decision.
    """
    if target_decision == DECISION_TP:
        context = (
            "Screening Context: This is a True Positive alert where the candidate hit is likely the same person "
            "with minor name recording differences (e.g. slight typo, token order swap, middle name abbreviation, "
            "transliteration variant, diacritic stripping, or compound name restructuring). "
            "Explain your reasoning step-by-step like a human compliance analyst and conclude with 'escalate to analyst'."
        )
    else:
        context = (
            "Screening Context: This is a False Positive alert where the candidate hit is a distinct individual "
            "despite coincidental name overlap (e.g. simple non-matching pairs like xxxxxx yyyyyyy vs zzzzz aaaaaaa "
            "irreconcilably different surname, conflicting middle name, "
            "or phonetically divergent but legally distinct surname). "
            "Explain your reasoning step-by-step like a human compliance analyst and conclude with 'disqualify'."
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
                parsed_data = clean_ollama_json_response(
                    raw_content,
                    client_name=client_name,
                    hit_name=hit_name,
                    target_decision=target_decision,
                    fallback_disposition_code=fallback_disposition_code
                )
                alert = ScreeningAlert.model_validate(parsed_data)
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
# ORCHESTRATION & COLLISION HANDLING
# ==============================================================================

async def run_hybrid_generation(
    source_names: List[str],
    num_tp: int,
    num_fp: int,
    ollama_model: str,
    batch_size: int,
    alpha: float = 0.90,
    k_max: int = 3
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coordinates Pipeline A and Pipeline B generation, enforces global pair uniqueness,
    feeds fuzzy-generated candidate pairs to Ollama for human-like cognitive evaluation,
    and maintains strict data integrity with fixed-vocabulary disposition codes.
    """
    start_time = time.time()

    # Calculate allocations
    n_tp_a = math.floor(alpha * num_tp)
    n_fp_a = math.floor(alpha * num_fp)
    n_tp_b = num_tp - n_tp_a
    n_fp_b = num_fp - n_fp_a

    records: List[Dict[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    metrics = {
        "num_tp_target": num_tp,
        "num_fp_target": num_fp,
        "n_tp_a": n_tp_a,
        "n_fp_a": n_fp_a,
        "n_tp_b": n_tp_b,
        "n_fp_b": n_fp_b,
        "pipeline_a_count": 0,
        "pipeline_b_count": 0,
        "collision_retries": 0,
        "fallbacks_to_double_mutation": 0,
    }

    # --------------------------------------------------------------------------
    # PIPELINE A: Programmatic Generation with Human Cognitive Reasoning
    # --------------------------------------------------------------------------
    if n_tp_a + n_fp_a > 0:
        print(f"\n[Pipeline A] Generating {n_tp_a} TPs and {n_fp_a} FPs with human cognitive reasoning...")
        tasks_a = [("TP", DECISION_TP)] * n_tp_a + [("FP", DECISION_FP)] * n_fp_a
        random.shuffle(tasks_a)

        pbar_a = tqdm(total=len(tasks_a), desc="Pipeline A (Human-Emulating Engine)", unit="records")
        for _, decision in tasks_a:
            client_name = random.choice(source_names)
            attempts = 0
            hit_name, matching_text, disposition_code, thinking = "", "", "", ""

            while attempts < 10:
                hit_name, matching_text, disposition_code, thinking = generate_pipeline_a_sample(
                    client_name, decision, double_mutation=(attempts > 3)
                )
                pair_key = (client_name.strip().lower(), hit_name.strip().lower())
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    break
                attempts += 1

            records.append({
                "client_name": client_name,
                "hit_name": hit_name,
                "matching_text": matching_text,
                "disposition_code": disposition_code,
                "thinking": thinking,
                "decision": decision,
                "created_by": "rule_based"
            })
            metrics["pipeline_a_count"] += 1
            pbar_a.update(1)
        pbar_a.close()

    # --------------------------------------------------------------------------
    # PIPELINE B: Feeding Fuzzy-Generated Pairs to Ollama for Human Review
    # --------------------------------------------------------------------------
    total_b_samples = n_tp_b + n_fp_b
    if total_b_samples > 0:
        print(f"\n[Pipeline B] Feeding {n_tp_b} TPs and {n_fp_b} FPs fuzzy pairs to Ollama ('{ollama_model}') for human cognitive review...")
        ollama_client = ollama.AsyncClient()
        eff_concurrency = min(batch_size, 2) if "cloud" in ollama_model.lower() else batch_size
        semaphore = asyncio.Semaphore(eff_concurrency)

        # Pre-generate candidate fuzzy pairs for Ollama review
        # (client_name, hit_name, matching_text, disposition_code, thinking, target_decision)
        fuzzy_items_b: List[Tuple[str, str, str, str, str, str]] = []

        for _ in range(n_tp_b):
            c_name = random.choice(source_names)
            attempts = 0
            while attempts < 10:
                h_name, m_text, d_code, th = generate_pipeline_a_sample(c_name, DECISION_TP, double_mutation=(attempts > 3))
                key = (c_name.strip().lower(), h_name.strip().lower())
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    fuzzy_items_b.append((c_name, h_name, m_text, d_code, th, DECISION_TP))
                    break
                attempts += 1

        for _ in range(n_fp_b):
            c_name = random.choice(source_names)
            attempts = 0
            while attempts < 10:
                h_name, m_text, d_code, th = generate_pipeline_a_sample(c_name, DECISION_FP, double_mutation=(attempts > 3))
                key = (c_name.strip().lower(), h_name.strip().lower())
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    fuzzy_items_b.append((c_name, h_name, m_text, d_code, th, DECISION_FP))
                    break
                attempts += 1

        random.shuffle(fuzzy_items_b)
        pbar_b = tqdm(total=len(fuzzy_items_b), desc="Pipeline B (Ollama Human Review)", unit="records")

        async def process_ollama_fuzzy_item(
            c_name: str, h_name: str, fallback_match: str, fallback_d_code: str, fallback_th: str, tgt_dec: str
        ):
            nonlocal metrics
            alert = await generate_single_ollama_sample(
                ollama_client, ollama_model, c_name, h_name, tgt_dec, fallback_d_code, semaphore
            )
            if alert and alert.thinking and alert.decision:
                records.append({
                    "client_name": c_name,
                    "hit_name": h_name,
                    "matching_text": alert.matching_text or fallback_match,
                    "disposition_code": alert.disposition_code or fallback_d_code,
                    "thinking": alert.thinking,
                    "decision": alert.decision,
                    "created_by": "ollama"
                })
                metrics["pipeline_b_count"] += 1
            else:
                metrics["fallbacks_to_double_mutation"] += 1
                records.append({
                    "client_name": c_name,
                    "hit_name": h_name,
                    "matching_text": fallback_match,
                    "disposition_code": fallback_d_code,
                    "thinking": fallback_th,
                    "decision": tgt_dec,
                    "created_by": "rule_based"
                })
                metrics["pipeline_a_count"] += 1
            pbar_b.update(1)

        await asyncio.gather(*(process_ollama_fuzzy_item(*item) for item in fuzzy_items_b))
        pbar_b.close()

    elapsed = time.time() - start_time
    metrics["total_elapsed_seconds"] = elapsed
    metrics["records_per_second"] = len(records) / elapsed if elapsed > 0 else 0.0

    df_out = pd.DataFrame(records)
    # Derive disposition_label deterministically from the controlled vocabulary dict
    df_out["disposition_label"] = df_out["disposition_code"].map(
        lambda c: DISPOSITION_CODES.get(c, "Unknown disposition code")
    )
    # Re-order columns strictly according to specification
    df_out = df_out[["client_name", "hit_name", "matching_text", "disposition_code", "disposition_label", "thinking", "decision", "created_by"]]
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
# MAIN CLI ENTRYPOINT
# ==============================================================================

def resolve_counts_and_percentages(args) -> Tuple[int, int, float, float, float, float]:
    """
    Computes (num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct)
    from command-line arguments, supporting both explicit counts and percentages.
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

    # 2. Resolve Decision Balance (TP % vs FP % and total counts)
    explicit_tp = args.num_true_positives
    explicit_fp = args.num_false_positives

    if explicit_tp is not None and explicit_fp is not None:
        num_tp = explicit_tp
        num_fp = explicit_fp
        total = num_tp + num_fp
        tp_pct = (num_tp / total * 100.0) if total > 0 else 50.0
        fp_pct = 100.0 - tp_pct
    else:
        total = args.total_samples if args.total_samples is not None else 100

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
        elif explicit_tp is not None:
            num_tp = explicit_tp
            num_fp = max(0, total - num_tp)
            tp_pct = (num_tp / total * 100.0) if total > 0 else 50.0
            fp_pct = 100.0 - tp_pct
        elif explicit_fp is not None:
            num_fp = explicit_fp
            num_tp = max(0, total - num_fp)
            fp_pct = (num_fp / total * 100.0) if total > 0 else 50.0
            tp_pct = 100.0 - fp_pct
        else:
            tp_pct = 50.0
            fp_pct = 50.0

        if not (0.0 <= tp_pct <= 100.0 and 0.0 <= fp_pct <= 100.0):
            raise ValueError(f"TP/FP percentages must be between 0 and 100. Got TP: {tp_pct}%, FP: {fp_pct}%.")

        num_tp = round((tp_pct / 100.0) * total)
        num_fp = total - num_tp

    return num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct

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
        help="Number of concurrent requests for Ollama batched execution (default: 15)."
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

    # Resolve counts and percentages
    num_tp, num_fp, tp_pct, fp_pct, ollama_pct, fuzzy_pct = resolve_counts_and_percentages(args)
    total_target = num_tp + num_fp
    alpha = fuzzy_pct / 100.0

    print("=" * 75)
    print("      CONFIGURABLE SYNTHETIC SCREENING ALERT DATA GENERATOR")
    print("=" * 75)
    print(f"Input CSV:               {args.input_csv_path}")
    print(f"Output CSV:              {args.output_csv_path}")
    print(f"Total Target Size:       {total_target} alerts")
    print(f"Decision Balance:        TP: {num_tp} ({tp_pct:.1f}%) | FP: {num_fp} ({fp_pct:.1f}%)")
    print(f"Generation Split:        Fuzzy Match: {fuzzy_pct:.1f}% | Ollama: {ollama_pct:.1f}%")
    print(f"Multi-Ethnic Diversity:  {'ENABLED (Global Pools Included)' if args.augment_diversity else 'DISABLED'}")
    print(f"Ollama Concurrency:      Batch Size {args.batch_size}")
    
    # Estimate time to completion
    est_seconds, est_time_str = estimate_generation_time(total_target, fuzzy_pct, ollama_pct, args.batch_size)
    print(f"Estimated Time:          ~{est_time_str}")
    print("=" * 75)

    # 1. Pre-flight verification
    resolved_model = args.ollama_model
    total_b_expected = total_target - (math.floor(alpha * num_tp) + math.floor(alpha * num_fp))

    if total_b_expected > 0 and ollama_pct > 0.0:
        print("\n[Pre-flight] Verifying Ollama model availability...")
        try:
            resolved_model = asyncio.run(verify_ollama_model(args.ollama_model))
            print(f"[Pre-flight] Verified model '{resolved_model}' successfully.")
        except Exception as e:
            print(f"\n[Pre-flight Warning] {e}")
            print("To proceed using 100% Fuzzy generation without Ollama, pass --ollama_percentage 0.")
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
            k_max=3
        )
    )

    # 4. Global Uniqueness and Integrity Verification
    total_records = len(df_results)
    unique_pairs = df_results[["client_name", "hit_name"]].drop_duplicates()
    is_strictly_unique = len(unique_pairs) == total_records

    # 5. Export to CSV
    df_results.to_csv(args.output_csv_path, index=False)
    print(f"\n[Export] Saved {len(df_results)} records to '{args.output_csv_path}'.")

    # 6. Print Execution Metrics Summary
    print("\n" + "=" * 75)
    print("                       PIPELINE EXECUTION METRICS                        ")
    print("=" * 75)
    print(f"Total Alerts Generated:             {total_records}")
    print(f"Total Execution Time:               {metrics['total_elapsed_seconds']:.2f} seconds")
    print(f"Throughput:                         {metrics['records_per_second']:.2f} records/sec")
    print("-" * 75)
    p_a = metrics['pipeline_a_count']
    p_b = metrics['pipeline_b_count']
    print(f"Pipeline A Output (RapidFuzz):      {p_a} ({p_a/total_records*100.0:.1f}%)")
    print(f"Pipeline B Output (Ollama):         {p_b} ({p_b/total_records*100.0:.1f}%)")
    print(f"Pipeline B Collision Retries:       {metrics['collision_retries']}")
    print(f"Fallback Double Mutations:          {metrics['fallbacks_to_double_mutation']}")
    print("-" * 75)
    tp_count = (df_results['decision'] == DECISION_TP).sum()
    fp_count = (df_results['decision'] == DECISION_FP).sum()
    print(f"True Positives ('{DECISION_TP}'):    {tp_count} ({tp_count/total_records*100.0:.1f}%)")
    print(f"False Positives ('{DECISION_FP}'):          {fp_count} ({fp_count/total_records*100.0:.1f}%)")
    print(f"Strict Global Uniqueness:           {'PASSED (Zero Duplicates)' if is_strictly_unique else 'FAILED'}")
    print("-" * 75)
    print("Disposition Code Distribution:")
    disp_counts = df_results['disposition_code'].value_counts().sort_index()
    for code, count in disp_counts.items():
        pct = (count / total_records * 100.0)
        label = DISPOSITION_CODES.get(code, "Unknown")
        print(f"  {code:40s} {count:5d} ({pct:5.1f}%) — {label}")
    print("=" * 75)

if __name__ == "__main__":
    main()
