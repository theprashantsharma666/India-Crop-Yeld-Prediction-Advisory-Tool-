"""
preprocessor.py
---------------
This file handles everything BEFORE the model sees any data.
Think of it as the "data preparation kitchen" — raw ingredients
come in, cleaned and structured data goes out.

What this file does:
  1. Loads the two raw CSV datasets from Kaggle
  2. Adds 9 missing Indian crops using ICAR agronomic data
  3. Defines soil and climate defaults for 20 Indian states
  4. Defines soil type profiles (black, red, sandy, loamy, etc.)
  5. Builds the 7-feature input dict the model needs to predict

This file does NOT train any model or make any predictions.
"""

import os
import json
import numpy as np
import pandas as pd


# ==============================================================
# SECTION 1 — DATASET LOADING
# ==============================================================
# These two functions just load the raw CSV files from wherever
# Kaggle downloaded them on your machine or Colab.

def load_datasets(crop_rec_path: str, crop_yld_path: str):
    """
    Loads both raw CSVs into pandas DataFrames.

    crop_rec_path : folder path returned by kagglehub for the
                    crop recommendation dataset
    crop_yld_path : folder path returned by kagglehub for the
                    yield prediction dataset

    Returns (df_rec, df_yld) — two DataFrames ready to use.
    """
    # Crop recommendation dataset — 2200 rows, 22 crops
    # Columns: N, P, K, temperature, humidity, ph, rainfall, label
    df_rec = pd.read_csv(f"{crop_rec_path}/Crop_recommendation.csv")

    # Yield dataset — 28242 rows covering 101 countries and 10 crops
    # Columns: Area, Item, Year, hg/ha_yield, rainfall, pesticides, avg_temp
    df_yld = pd.read_csv(f"{crop_yld_path}/yield_df.csv")

    print(f"Recommendation dataset loaded: {df_rec.shape}")
    print(f"Yield dataset loaded         : {df_yld.shape}")

    return df_rec, df_yld


# ==============================================================
# SECTION 2 — SYNTHETIC INDIAN CROP GENERATION
# ==============================================================
# The original crop recommendation dataset is missing 9 major
# Indian crops: wheat, sugarcane, potato, onion, tomato, mustard,
# barley, sorghum, and turmeric.
#
# These are the backbone of Indian agriculture — a tool without
# them is useless for most Indian farmers.
#
# Solution: We use ICAR (Indian Council of Agricultural Research)
# published agronomic guidelines to define realistic soil and
# climate conditions for each crop, then generate 100 synthetic
# data rows per crop using a normal distribution.
#
# This technique is called "dataset augmentation".

# Each crop profile defines (mean, standard_deviation) for all
# 7 features. The values come from ICAR crop production guides.
ICAR_PROFILES = {
    # Rabi crop — sown Oct-Nov in cool dry weather
    # Needs moderate N, high P, low K, neutral pH
    "wheat": {
        "N": (120, 10), "P": (60, 8),  "K": (40, 6),
        "temperature": (21, 3), "humidity": (65, 8),
        "ph": (7.0, 0.4), "rainfall": (200, 30),
    },
    # Tropical perennial — needs very high water and N
    "sugarcane": {
        "N": (175, 15), "P": (45, 8),  "K": (80, 10),
        "temperature": (28, 3), "humidity": (75, 8),
        "ph": (6.5, 0.4), "rainfall": (1200, 100),
    },
    # Rabi crop — cool climate, high P and K for tuber development
    "potato": {
        "N": (140, 12), "P": (70, 8),  "K": (100, 10),
        "temperature": (18, 3), "humidity": (70, 8),
        "ph": (5.8, 0.4), "rainfall": (500, 60),
    },
    # Grown both Kharif and Rabi — moderate water, well-drained soil
    "onion": {
        "N": (100, 10), "P": (50, 8),  "K": (75, 8),
        "temperature": (20, 3), "humidity": (60, 8),
        "ph": (6.0, 0.4), "rainfall": (650, 70),
    },
    # Warm season crop — high P and K, disease-prone in humidity
    "tomato": {
        "N": (120, 10), "P": (80, 8),  "K": (100, 10),
        "temperature": (24, 3), "humidity": (68, 8),
        "ph": (6.2, 0.4), "rainfall": (700, 70),
    },
    # Rabi oilseed — low water, cool dry climate, sulphur important
    "mustard": {
        "N": (80, 8),  "P": (40, 6),  "K": (30, 5),
        "temperature": (18, 3), "humidity": (55, 8),
        "ph": (6.8, 0.4), "rainfall": (250, 30),
    },
    # Rabi cereal — coldest tolerance, lowest water of all cereals
    "barley": {
        "N": (90, 8),  "P": (40, 6),  "K": (30, 5),
        "temperature": (15, 3), "humidity": (55, 8),
        "ph": (6.8, 0.4), "rainfall": (180, 25),
    },
    # Kharif crop — drought tolerant, thrives in hot dry conditions
    "sorghum": {
        "N": (100, 10), "P": (50, 7), "K": (40, 6),
        "temperature": (30, 3), "humidity": (55, 8),
        "ph": (6.5, 0.4), "rainfall": (400, 50),
    },
    # Tropical spice — high rainfall, high humidity, high K for rhizomes
    "turmeric": {
        "N": (120, 10), "P": (60, 8), "K": (120, 10),
        "temperature": (27, 3), "humidity": (80, 8),
        "ph": (5.8, 0.4), "rainfall": (1500, 120),
    },
}

# How many synthetic rows to generate per crop.
# 100 rows matches the original dataset (100 rows per crop).
ROWS_PER_CROP = 100


def generate_synthetic_rows(profiles: dict = ICAR_PROFILES,
                             rows: int = ROWS_PER_CROP,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic training rows for each crop.

    How it works:
      For each crop, we sample each feature from a normal distribution
      centered on the ICAR mean with the ICAR standard deviation.
      This creates natural variation, just like real field measurements vary.

    seed=42 makes the generation reproducible — same result every run.
    np.clip keeps humidity and pH in their valid physical ranges.
    max(0, ...) prevents negative values for N, P, K, rainfall.

    Returns a DataFrame with the same columns as df_rec.
    """
    np.random.seed(seed)  # fix the random seed for reproducibility
    new_rows = []

    for crop, profile in profiles.items():
        for _ in range(rows):
            row = {
                # max(0,...) prevents negative nutrient values
                "N":           max(0, np.random.normal(*profile["N"])),
                "P":           max(0, np.random.normal(*profile["P"])),
                "K":           max(0, np.random.normal(*profile["K"])),
                "temperature": max(0, np.random.normal(*profile["temperature"])),
                # humidity must stay between 20% and 100%
                "humidity":    np.clip(np.random.normal(*profile["humidity"]), 20, 100),
                # pH must stay between 3.5 (very acidic) and 9.5 (very alkaline)
                "ph":          np.clip(np.random.normal(*profile["ph"]), 3.5, 9.5),
                # rainfall cannot be negative
                "rainfall":    max(0, np.random.normal(*profile["rainfall"])),
                "label":       crop,
            }
            new_rows.append(row)

    return pd.DataFrame(new_rows)


def build_combined_dataset(df_rec: pd.DataFrame,
                            save_path: str = None) -> pd.DataFrame:
    """
    Merges the original 22-crop dataset with our synthetic Indian crops.

    Result: 31 crops × 100 rows = 3100 rows total.
    This combined dataset is what gets fed into model training.

    If save_path is provided, the CSV is saved there so you don't
    have to regenerate it every session.
    """
    df_new = generate_synthetic_rows()

    # Stack original and new rows — ignore_index resets row numbers
    df_combined = pd.concat([df_rec, df_new], ignore_index=True)

    print(f"Original : {len(df_rec)} rows, {df_rec['label'].nunique()} crops")
    print(f"Added    : {len(df_new)} rows — {sorted(df_new['label'].unique())}")
    print(f"Combined : {len(df_combined)} rows, {df_combined['label'].nunique()} crops")

    if save_path:
        df_combined.to_csv(save_path, index=False)
        print(f"Saved combined dataset → {save_path}")

    return df_combined


# ==============================================================
# SECTION 3 — TRAINING FEATURE LIMITS
# ==============================================================
# These are the min and max values we saw in the training data.
# Any input from a farmer that falls outside these ranges gets
# clipped back to the boundary.
#
# Why? The model only "knows" conditions it was trained on.
# If we feed it N=500 when training data only went to N=211,
# predictions become unreliable — the model is guessing in
# territory it has never seen.

LIMITS = {
    "N":           (0,    211),    # kg/ha nitrogen
    "P":           (5,    145),    # kg/ha phosphorus
    "K":           (5,    205),    # kg/ha potassium
    "temperature": (7.5,  43.7),   # degrees Celsius
    "humidity":    (14.3, 100.0),  # percentage
    "ph":          (3.5,  9.9),    # soil pH scale
    "rainfall":    (20.2, 1690.8), # mm per year
}


# ==============================================================
# SECTION 4 — STATE SOIL DEFAULTS (Alternative A)
# ==============================================================
# Most Indian farmers don't have a Soil Health Card and don't
# know their NPK values. But soil type varies predictably by
# region — Haryana's alluvial plains always have similar chemistry.
#
# IMPORTANT: These values are calibrated to MATCH the training
# crop profiles, not raw real-world soil measurements. This is
# because the model learned from the training data distribution,
# not from actual field measurements.
#
# Example: Haryana's real N is ~40 kg/ha, but wheat training
# data used N~120. So Haryana defaults are set to 122 so the
# model correctly predicts wheat.

STATE_SOIL_DEFAULTS = {

    # ── North India — wheat / barley / mustard belt ──────────
    # These match the wheat training profile (N=120, P=60, K=40)
    "Haryana":          {"N": 122, "P": 58, "K": 40,  "ph": 7.0},
    "Punjab":           {"N": 118, "P": 62, "K": 38,  "ph": 6.9},
    "Uttar Pradesh":    {"N": 125, "P": 55, "K": 42,  "ph": 7.1},
    "Rajasthan":        {"N": 100, "P": 38, "K": 35,  "ph": 7.5},
    "Uttarakhand":      {"N": 108, "P": 55, "K": 48,  "ph": 6.5},
    "Himachal Pradesh": {"N": 105, "P": 58, "K": 50,  "ph": 6.2},

    # ── Central India — cotton / soybean belt ─────────────────
    # These match the cotton training profile (N=118, P=46, K=20)
    "Maharashtra":      {"N": 118, "P": 46, "K": 20,  "ph": 6.9},
    "Madhya Pradesh":   {"N": 115, "P": 44, "K": 20,  "ph": 7.0},
    "Gujarat":          {"N": 116, "P": 45, "K": 21,  "ph": 7.1},

    # ── East India — rice / jute belt ─────────────────────────
    # These match the rice training profile (N=80, P=47, K=40)
    "West Bengal":      {"N": 80,  "P": 47, "K": 40,  "ph": 6.4},
    "Bihar":            {"N": 82,  "P": 48, "K": 40,  "ph": 6.8},
    "Odisha":           {"N": 79,  "P": 46, "K": 39,  "ph": 6.3},
    "Jharkhand":        {"N": 78,  "P": 45, "K": 38,  "ph": 6.1},
    "Assam":            {"N": 79,  "P": 47, "K": 40,  "ph": 5.9},
    "Chhattisgarh":     {"N": 80,  "P": 46, "K": 39,  "ph": 6.4},

    # ── South India — coconut / banana / coffee belt ──────────
    # Kerala matches coconut (N=22, P=17, K=30, humidity=94)
    # South matches banana (N=100, P=82, K=50, humidity=80)
    "Kerala":           {"N": 22,  "P": 17, "K": 30,  "ph": 6.0},
    "Tamil Nadu":       {"N": 100, "P": 82, "K": 50,  "ph": 6.0},
    "Karnataka":        {"N": 101, "P": 80, "K": 50,  "ph": 6.1},
    "Andhra Pradesh":   {"N": 100, "P": 78, "K": 49,  "ph": 6.2},
    "Telangana":        {"N": 99,  "P": 76, "K": 48,  "ph": 6.3},
}


# ==============================================================
# SECTION 5 — STATE CLIMATE DEFAULTS
# ==============================================================
# Temperature, humidity, and rainfall are auto-filled from state.
# Farmers cannot measure these — we use known regional climate data.
#
# Again, these are calibrated to the training data, not raw IMD
# (India Meteorological Department) annual averages, because the
# model was trained on crop-specific conditions not calendar averages.

STATE_CLIMATE_DEFAULTS = {

    # ── North India — cool dry (wheat needs 200mm, temp 21°C) ─
    "Haryana":          {"temperature": 21, "humidity": 64,  "rainfall": 200},
    "Punjab":           {"temperature": 20, "humidity": 65,  "rainfall": 195},
    "Uttar Pradesh":    {"temperature": 22, "humidity": 66,  "rainfall": 210},
    "Rajasthan":        {"temperature": 19, "humidity": 55,  "rainfall": 180},
    "Uttarakhand":      {"temperature": 17, "humidity": 65,  "rainfall": 200},
    "Himachal Pradesh": {"temperature": 14, "humidity": 64,  "rainfall": 195},

    # ── Central India — warm dry (cotton needs only 80mm) ─────
    # Note: actual Maharashtra gets 900mm rain, but cotton grows
    # in the driest months. Training data reflects growing season.
    "Maharashtra":      {"temperature": 24, "humidity": 79,  "rainfall": 82},
    "Madhya Pradesh":   {"temperature": 24, "humidity": 78,  "rainfall": 80},
    "Gujarat":          {"temperature": 25, "humidity": 77,  "rainfall": 81},

    # ── East India — warm humid (rice needs 82% humidity) ─────
    "West Bengal":      {"temperature": 24, "humidity": 82,  "rainfall": 238},
    "Bihar":            {"temperature": 25, "humidity": 81,  "rainfall": 235},
    "Odisha":           {"temperature": 26, "humidity": 82,  "rainfall": 236},
    "Jharkhand":        {"temperature": 25, "humidity": 80,  "rainfall": 234},
    "Assam":            {"temperature": 24, "humidity": 82,  "rainfall": 236},
    "Chhattisgarh":     {"temperature": 25, "humidity": 81,  "rainfall": 235},

    # ── South India — hot humid (coconut needs 94% humidity) ──
    "Kerala":           {"temperature": 27, "humidity": 94,  "rainfall": 176},
    "Tamil Nadu":       {"temperature": 27, "humidity": 80,  "rainfall": 105},
    "Karnataka":        {"temperature": 26, "humidity": 80,  "rainfall": 104},
    "Andhra Pradesh":   {"temperature": 28, "humidity": 80,  "rainfall": 105},
    "Telangana":        {"temperature": 28, "humidity": 79,  "rainfall": 104},
}


# ==============================================================
# SECTION 6 — SOIL TYPE PROFILES (Alternative B)
# ==============================================================
# If a farmer doesn't know their NPK values but CAN identify what
# their soil looks and feels like, this gives a reasonable estimate.
#
# Every farmer knows if their soil is black and sticky or light
# and sandy — this converts that physical knowledge into numbers.
# The descriptions are written in plain farmer-friendly language.

SOIL_TYPE_PROFILES = {
    # Maharashtra, MP — high K, moderate N, slightly alkaline
    "Black (dark, sticky when wet)": {
        "N": 200, "P": 15, "K": 200, "ph": 7.5,
        "description": "Black/Regur soil — Maharashtra, MP",
        "crops_hint":  "Cotton, soybean, wheat",
    },
    # Karnataka, AP, Odisha — low fertility across all nutrients
    "Red (reddish, grainy, crumbles easily)": {
        "N": 148, "P": 10, "K": 138, "ph": 6.5,
        "description": "Red laterite — Karnataka, AP, Odisha",
        "crops_hint":  "Groundnut, millets, pulses",
    },
    # Rajasthan, Gujarat — lowest fertility, but barley and mustard thrive
    "Sandy (light, flows through fingers, dries fast)": {
        "N": 130, "P": 8,  "K": 118, "ph": 7.0,
        "description": "Sandy/Desert soil — Rajasthan, Gujarat",
        "crops_hint":  "Barley, mustard, drought-tolerant crops",
    },
    # Punjab, Haryana, UP — the most fertile soil, good for almost anything
    "Loamy (dark brown, holds shape, feels smooth)": {
        "N": 205, "P": 20, "K": 198, "ph": 7.0,
        "description": "Alluvial loamy — Punjab, Haryana, UP",
        "crops_hint":  "Wheat, rice, sugarcane, most crops",
    },
    # River deltas, Bengal — retains water, good for rice and jute
    "Clay (heavy, sticks to hands, waterlogged after rain)": {
        "N": 185, "P": 16, "K": 178, "ph": 6.8,
        "description": "Clay/Heavy soil — river deltas",
        "crops_hint":  "Rice, jute, sugarcane",
    },
    # NE India, Himachal, Kerala — rich in organic matter, acidic
    "Hilly/Forest (dark, rich, lots of leaves and organic matter)": {
        "N": 200, "P": 18, "K": 158, "ph": 5.8,
        "description": "Forest/Hill soil — NE India, Himachal, Kerala",
        "crops_hint":  "Tea, coffee, turmeric, spices",
    },
}


# ==============================================================
# SECTION 7 — INPUT ASSEMBLY (THE 4 PATHS)
# ==============================================================
# This is the heart of preprocessor.py.
#
# The model needs exactly 7 numbers to make a prediction:
#   N, P, K, temperature, humidity, ph, rainfall
#
# But farmers come with different levels of knowledge.
# We support 4 paths to get those 7 numbers:
#
#   Path 1 (manual)    — farmer has Soil Health Card → enters NPK directly
#   Path 2 (state)     — farmer selects their state → defaults auto-filled
#   Path 3 (soil_type) — farmer picks soil colour → mapped to NPK ranges
#   Path 4 (proxy)     — farmer answers 4 questions → NPK estimated
#
# In all cases, climate (temperature, humidity, rainfall) is always
# auto-filled from state — farmers cannot measure this.

def get_soil_from_state(state: str) -> dict:
    """
    Returns soil NPK + pH defaults for a given state.
    Falls back to national average if state not found.
    """
    return STATE_SOIL_DEFAULTS.get(
        state,
        # National average fallback — used if state is not in our list
        {"N": 185, "P": 14, "K": 165, "ph": 7.0}
    )


def get_soil_from_type(soil_type: str) -> dict:
    """
    Converts a soil type description (what the farmer sees)
    into NPK + pH numbers (what the model needs).
    Only extracts N, P, K, ph — drops description and crops_hint.
    """
    profile = SOIL_TYPE_PROFILES.get(soil_type)
    if profile:
        return {k: profile[k] for k in ["N", "P", "K", "ph"]}
    # Fallback to national average if soil type string is unrecognised
    return {"N": 185, "P": 14, "K": 165, "ph": 7.0}


def estimate_soil_from_questions(answers: dict) -> dict:
    """
    Alternative C — estimates NPK from 4 simple yes/no questions.

    Every answer adjusts NPK up or down from a national baseline.
    The logic encodes real agricultural relationships:
      - Legumes leave nitrogen in soil
      - Cereals deplete nitrogen
      - Waterlogged clay soils have higher potassium
      - Near-river alluvial soils are more fertile

    answers dict format:
      'q1': 'yes'/'no'      — Is your field near a river or canal?
      'q2': 'legume'/'cereal'/'fallow' — What did you grow last season?
      'q3': 'yes'/'no'      — Did you use organic manure (FYM/compost)?
      'q4': 'yes'/'no'      — Do you have irrigation access?
    """
    # Start from national average baseline
    N, P, K, ph = 185, 14, 165, 7.0

    # Q1: Near river/canal → alluvial soil, higher K and N, slightly acidic
    if answers.get("q1") == "yes":
        K += 25; N += 15; ph -= 0.3
    else:
        K -= 15; N -= 10  # no water body = less fertile soil

    # Q2: Last crop type
    # Legumes (dal, soybean) fix atmospheric nitrogen → boost N
    # Cereals (wheat, rice) are heavy N consumers → deplete N
    # Fallow → soil partially recovered
    q2 = answers.get("q2")
    if q2 == "legume":
        N += 30; P += 5   # legumes are nature's fertilizer
    elif q2 == "cereal":
        N -= 25            # cereals leave soil N-depleted
    elif q2 == "fallow":
        N += 12            # rested soil recovers some nitrogen

    # Q3: Used organic manure → boosts all three nutrients
    if answers.get("q3") == "yes":
        N += 20; P += 5; K += 15
    else:
        N -= 20; P -= 5; K -= 15  # no manure = below average fertility

    # Q4: Irrigation available → better moisture management = better nutrient uptake
    if answers.get("q4") == "yes":
        N += 15; P += 4; K += 12; ph -= 0.3

    # Clip final values to stay within valid training ranges
    # This prevents an extreme combination of answers from creating
    # an impossible input value
    N  = max(0,   min(211, N))
    P  = max(5,   min(145, P))
    K  = max(5,   min(205, K))
    ph = max(3.5, min(9.9, ph))

    return {"N": round(N), "P": round(P),
            "K": round(K), "ph": round(ph, 1)}


def get_climate_from_state(state: str) -> dict:
    """
    Returns temperature, humidity, and rainfall for a state.
    These values are always auto-filled — no farmer input needed.
    """
    return STATE_CLIMATE_DEFAULTS.get(
        state,
        # National average fallback
        {"temperature": 26, "humidity": 67, "rainfall": 900}
    )


def get_full_input(path: str,
                   state: str = None,
                   soil_type: str = None,
                   proxy_answers: dict = None,
                   manual_values: dict = None) -> dict:
    """
    THE MAIN FUNCTION of this file.

    Takes whatever the farmer provides and returns the complete
    7-feature dictionary the model needs to make a prediction.

    Parameters:
      path          : which input path the farmer is using
                      ('manual', 'state', 'soil_type', or 'proxy')
      state         : Indian state name (used for climate in all paths)
      soil_type     : soil visual description (only for 'soil_type' path)
      proxy_answers : {'q1': 'yes', 'q2': 'legume', ...} (only for 'proxy')
      manual_values : {'N': 90, 'P': 42, 'K': 43, 'ph': 6.5} (for 'manual')

    Returns:
      dict with keys: N, P, K, temperature, humidity, ph, rainfall
      All values are guaranteed to be within LIMITS (safety-clipped).
    """

    # ── Step 1: Get soil NPK + pH ────────────────────────────
    # This depends on which path the farmer chose.
    if path == "manual" and manual_values:
        # Farmer has a Soil Health Card — trust their values directly
        soil = {k: manual_values[k] for k in ["N", "P", "K", "ph"]}

    elif path == "state" and state:
        # No soil card — use regional defaults for their state
        soil = get_soil_from_state(state)

    elif path == "soil_type" and soil_type:
        # Farmer can identify soil visually — map to NPK ranges
        soil = get_soil_from_type(soil_type)

    elif path == "proxy" and proxy_answers:
        # Farmer answers 4 simple questions — estimate NPK
        soil = estimate_soil_from_questions(proxy_answers)

    else:
        # Last resort fallback — use national average
        soil = {"N": 185, "P": 14, "K": 165, "ph": 7.0}

    # ── Step 2: Get climate from state ───────────────────────
    # Always auto-filled. Farmer doesn't need to measure this.
    # If state is None (e.g. developer testing), use national average.
    climate = (get_climate_from_state(state) if state
               else {"temperature": 26, "humidity": 67, "rainfall": 900})

    # ── Step 3: Combine soil + climate into one dict ─────────
    full = {
        "N":           soil["N"],
        "P":           soil["P"],
        "K":           soil["K"],
        "temperature": climate["temperature"],
        "humidity":    climate["humidity"],
        "ph":          soil["ph"],
        "rainfall":    climate["rainfall"],
    }

    # ── Step 4: Safety clip ───────────────────────────────────
    # No matter which path was used, clip all values to training
    # data bounds. This is a final safety net that prevents any
    # out-of-distribution input from reaching the model.
    for key, (lo, hi) in LIMITS.items():
        full[key] = max(lo, min(hi, full[key]))

    return full


# ==============================================================
# SECTION 8 — SAVE / LOAD JSON HELPERS
# ==============================================================
# The Streamlit app loads state and soil data from JSON files
# rather than importing this Python file directly.
# This allows updating state defaults without touching Python code.

def save_defaults_to_json(output_dir: str = "output") -> None:
    """
    Saves all three lookup dictionaries as JSON files.
    Called once at the end of Stage 1 in main.py.

    These JSON files are then loaded by the Streamlit app
    using json.load() — no Python import needed.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "state_soil_defaults.json":    STATE_SOIL_DEFAULTS,
        "state_climate_defaults.json": STATE_CLIMATE_DEFAULTS,
        "soil_type_profiles.json":     SOIL_TYPE_PROFILES,
    }

    for fname, data in files.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)  # indent=2 makes it human-readable
        print(f"Saved → {path}")