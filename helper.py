"""
helper.py
---------
Everything that runs AFTER the model makes a prediction.
Think of it as the "output enrichment layer" — a crop name comes in,
a full farmer-ready advisory goes out.

What this file does:
  1. Maps model crop names to FAO yield dataset names
  2. Stores ICAR yield estimates (quintal/acre) for all 31 crops
  3. Stores MSP (government minimum support prices) for 2024-25
  4. Calculates expected yield range and income range
  5. Stores full farming advisory for all 31 crops
  6. Generates contextual disease warnings based on climate
  7. Provides nearest government agriculture contact per state

This file does NOT load datasets, train models, or handle user input.
"""

import pandas as pd
from typing import List


# ==============================================================
# SECTION 1 — CROP NAME MAPPING
# ==============================================================
# Our model uses lowercase names like 'rice', 'wheat', 'banana'.
# The FAO yield dataset uses different names like 'Rice, paddy',
# 'Wheat', 'Bananas'.
#
# This mapping bridges that gap. None means the crop is not in
# the yield dataset at all — we fall back to ICAR estimates.

CROP_NAME_MAPPING = {
    "rice":         "Rice, paddy",
    "maize":        "Maize",
    "chickpea":     "Chick peas, dry",
    "kidneybeans":  "Beans, dry",
    "pigeonpeas":   "Pigeon peas, dry",
    "mothbeans":    None,                 # not in FAO dataset
    "mungbean":     "Mung beans, dry",
    "blackgram":    None,                 # not in FAO dataset
    "lentil":       "Lentils, dry",
    "pomegranate":  None,                 # not in FAO dataset
    "banana":       "Bananas",
    "mango":        "Mangoes, guavas and mangosteens",
    "grapes":       "Grapes",
    "watermelon":   "Watermelons",
    "muskmelon":    "Cantaloupes and other melons",
    "apple":        "Apples",
    "orange":       "Oranges",
    "papaya":       "Papayas",
    "coconut":      "Coconuts",
    "cotton":       "Seed cotton, unginned",
    "jute":         "Jute",
    "coffee":       "Coffee, green",
    "wheat":        "Wheat",
    "sugarcane":    "Sugar cane",
    "potato":       "Potatoes",
    "onion":        "Onions and shallots, dry (excluding dehydrated)",
    "tomato":       "Tomatoes",
    "mustard":      "Rapeseed or canola, dry",
    "barley":       "Barley",
    "sorghum":      "Sorghum",
    "turmeric":     None,                 # not in FAO dataset
}


# ==============================================================
# SECTION 2 — ICAR YIELD ESTIMATES (quintal/acre)
# ==============================================================
# Source: ICAR (Indian Council of Agricultural Research) crop
# production guides and annual reports, 2022-24.
#
# Each crop has three values:
#   low    — 25th percentile (bad year / poor management)
#   median — 50th percentile (average year)
#   high   — 75th percentile (good year / good management)
#
# Unit is QUINTAL PER ACRE — the standard unit Indian farmers use.
# (1 quintal = 100 kg. 1 acre = 0.405 hectares.)
#
# We use ICAR data instead of the FAO yield dataset because:
# 1. ICAR is India-specific (FAO mixes 100+ countries)
# 2. ICAR data is more recent (FAO dataset ends at 2013)
# 3. ICAR reports are in quintal/acre (farmer-friendly unit)

ICAR_YIELD_INDIA = {
    # Cereals — staple food crops
    "wheat":        {"low": 16.0, "median": 19.0, "high": 22.0},
    "rice":         {"low": 15.0, "median": 18.0, "high": 22.0},
    "maize":        {"low": 10.0, "median": 13.0, "high": 16.0},
    "sorghum":      {"low":  8.0, "median": 10.0, "high": 13.0},
    "barley":       {"low": 12.0, "median": 15.0, "high": 18.0},

    # Vegetables and tubers
    "potato":       {"low": 80.0, "median":100.0, "high":120.0},
    "onion":        {"low": 40.0, "median": 55.0, "high": 70.0},
    "tomato":       {"low": 80.0, "median":100.0, "high":130.0},

    # Cash crops — high income but need specific conditions
    "sugarcane":    {"low":280.0, "median":320.0, "high":380.0},  # very high quintal due to weight of cane
    "cotton":       {"low":  2.0, "median":  2.5, "high":  3.5},  # low because only lint counted
    "jute":         {"low": 12.0, "median": 15.0, "high": 18.0},

    # Oilseeds
    "mustard":      {"low":  5.0, "median":  6.5, "high":  8.0},

    # Pulses — low yield but high MSP price per quintal
    "chickpea":     {"low":  4.0, "median":  5.5, "high":  7.0},
    "lentil":       {"low":  3.5, "median":  5.0, "high":  6.5},
    "pigeonpeas":   {"low":  3.0, "median":  4.0, "high":  5.5},
    "mungbean":     {"low":  2.5, "median":  3.5, "high":  4.5},
    "blackgram":    {"low":  2.0, "median":  3.0, "high":  4.0},
    "kidneybeans":  {"low":  3.0, "median":  4.0, "high":  5.5},
    "mothbeans":    {"low":  1.5, "median":  2.5, "high":  3.5},

    # Fruits — high income per acre
    "banana":       {"low": 80.0, "median":100.0, "high":130.0},
    "coconut":      {"low": 40.0, "median": 55.0, "high": 70.0},
    "mango":        {"low": 25.0, "median": 35.0, "high": 50.0},
    "grapes":       {"low": 60.0, "median": 80.0, "high":110.0},
    "apple":        {"low": 60.0, "median": 80.0, "high":100.0},
    "orange":       {"low": 50.0, "median": 65.0, "high": 85.0},
    "papaya":       {"low": 80.0, "median":100.0, "high":130.0},
    "watermelon":   {"low": 60.0, "median": 80.0, "high":110.0},
    "muskmelon":    {"low": 40.0, "median": 55.0, "high": 75.0},
    "pomegranate":  {"low": 30.0, "median": 40.0, "high": 55.0},

    # Beverages and spices
    "coffee":       {"low":  3.0, "median":  4.0, "high":  5.5},
    "turmeric":     {"low": 25.0, "median": 30.0, "high": 40.0},
}


def get_yield_estimate(crop: str) -> dict | None:
    """
    Returns yield range (low/median/high) in quintal/acre for a crop.
    Returns None if the crop is not in our ICAR data.

    Example output:
      {'crop': 'wheat', 'low_q_acre': 16.0, 'median_q_acre': 19.0,
       'high_q_acre': 22.0, 'source': 'ICAR estimate'}
    """
    if crop in ICAR_YIELD_INDIA:
        est = ICAR_YIELD_INDIA[crop]
        return {
            "crop":          crop,
            "low_q_acre":    est["low"],
            "median_q_acre": est["median"],
            "high_q_acre":   est["high"],
            "source":        "ICAR estimate",
        }
    return None  # crop not in our data


# ==============================================================
# SECTION 3 — MSP / MARKET PRICES (INR per quintal)
# ==============================================================
# MSP = Minimum Support Price — the government's guaranteed
# minimum price. Farmers are protected from selling below this.
#
# Source: CACP (Commission for Agricultural Costs and Prices)
# announcement for 2024-25 Kharif and Rabi seasons.
#
# For crops WITHOUT official MSP (fruits, vegetables), we use
# approximate mandi (market) prices. These change seasonally —
# the values here are conservative annual averages.
#
# Note: Sugarcane uses FRP (Fair and Remunerative Price) which
# is announced separately from MSP but serves the same purpose.

MSP_2024 = {
    # Government MSP crops — guaranteed price
    "wheat":        2275,   # major Rabi cereal — MSP raised from 2015
    "rice":         2300,   # major Kharif cereal
    "maize":        2090,   # Kharif cereal
    "barley":       1735,   # Rabi cereal — lower MSP than wheat
    "sorghum":      3371,   # Kharif cereal (jowar)
    "chickpea":     5440,   # Rabi pulse (chana)
    "lentil":       6425,   # Rabi pulse (masoor)
    "pigeonpeas":   7550,   # Kharif pulse (arhar/tur)
    "mungbean":     8682,   # Kharif pulse (moong) — highest pulse MSP
    "blackgram":    7400,   # Kharif pulse (urad)
    "mustard":      5650,   # Rabi oilseed (rapeseed)
    "sugarcane":     340,   # FRP per quintal — paid by sugar mills
    "cotton":       7121,   # Kharif cash crop (medium staple)
    "jute":         5335,   # Kharif fibre crop
    "mothbeans":    8558,   # Kharif pulse (moth)
    "kidneybeans":  6000,   # approximate — no official MSP

    # Non-MSP crops — approximate mandi price (annual average)
    # These prices are more volatile than MSP crops
    "potato":       1200,
    "onion":         800,   # highly volatile — can spike to 3000+
    "tomato":       1000,   # very volatile — seasonal price swings
    "banana":       1500,
    "coconut":      2200,   # per quintal (approx ₹22 per coconut)
    "mango":        3000,
    "grapes":       4000,
    "apple":        5000,
    "orange":       2500,
    "papaya":       1200,
    "watermelon":    600,   # low price but high volume per acre
    "muskmelon":     800,
    "pomegranate":  6000,
    "coffee":      12000,   # highest price per quintal in our list
    "turmeric":     7000,
}


def get_income_estimate(crop: str, acres: float = 1.0) -> dict:
    """
    Calculates estimated income range for a crop over given acres.

    Formula: yield_low × MSP = income_low
             yield_high × MSP = income_high

    Scales linearly with field size — 2 acres gives double the income.

    Returns dict with yield_low, yield_high, price_per_q,
    income_low, income_high.
    If yield or price data is missing, all values are None.
    """
    yield_data = get_yield_estimate(crop)
    price      = MSP_2024.get(crop)

    # If either yield or price is missing, return empty result
    if yield_data is None or price is None:
        return {
            "crop":        crop,
            "acres":       acres,
            "yield_low":   None,
            "yield_high":  None,
            "income_low":  None,
            "income_high": None,
            "price_per_q": price,
            "note":        "Yield or price data not available",
        }

    # Scale yield by field size
    yield_low  = round(yield_data["low_q_acre"]  * acres, 1)
    yield_high = round(yield_data["high_q_acre"] * acres, 1)

    return {
        "crop":         crop,
        "acres":        acres,
        "yield_low":    yield_low,
        "yield_high":   yield_high,
        "price_per_q":  price,
        # round() to nearest rupee — no decimals on income display
        "income_low":   round(yield_low  * price),
        "income_high":  round(yield_high * price),
        "yield_source": yield_data["source"],
    }


# ==============================================================
# SECTION 4 — CROP ADVISORY
# ==============================================================
# Static farming advice for each of the 31 crops.
# This is the content a farmer actually reads and acts on.
#
# Each crop has 6 fields:
#   season       — when to sow
#   harvest      — when to harvest
#   water        — irrigation schedule and critical stages
#   fertilizer   — NPK application timing and tips
#   disease_risk — main diseases and pests to watch for
#   tip          — the single most important practical advice
#
# Written in plain language — no jargon. Field-tested advice
# from ICAR extension publications and Krishi Vigyan Kendra guides.

CROP_ADVISORY = {
    # ── North India cereals ──────────────────────────────────
    "wheat": {
        "season":       "Rabi (October – November sowing)",
        "harvest":      "March – April",
        "water":        "Moderate — 4 to 6 irrigations. Critical stages: crown root (20-25 days), tillering, jointing, flowering, grain filling",
        "fertilizer":   "Apply full P and K at sowing. Split N: half at sowing, quarter at first irrigation, quarter at second irrigation",
        "disease_risk": "Yellow rust if humidity above 80% in Feb-March. Watch for loose smut at heading stage",
        "tip":          "Avoid late sowing after November 25 — each week of delay reduces yield by 1-2 quintal/acre",
    },
    "rice": {
        "season":       "Kharif (June – July transplanting)",
        "harvest":      "October – November",
        "water":        "High — keep 5cm standing water during vegetative stage. Drain 10 days before harvest",
        "fertilizer":   "Apply FYM 2 weeks before transplanting. Split urea in 3 doses: basal, tillering, panicle initiation",
        "disease_risk": "Blast disease in cool humid weather. Brown planthopper in dense crop. Stem borer July-August",
        "tip":          "Use 25-day-old seedlings for transplanting — older seedlings give lower yield",
    },
    "maize": {
        "season":       "Kharif (June-July) or Rabi (Oct-Nov)",
        "harvest":      "90-100 days after sowing",
        "water":        "Moderate — critical at knee-high stage, tasseling, and grain filling. Avoid waterlogging",
        "fertilizer":   "High N demand. Apply 25% N at sowing, 50% at knee-high, 25% at tasseling",
        "disease_risk": "Turcicum leaf blight in humid conditions. Fall armyworm — check whorls weekly from July",
        "tip":          "Maintain 60cm row spacing. Closer spacing increases lodging risk",
    },
    "barley": {
        "season":       "Rabi (October – November sowing)",
        "harvest":      "March – April",
        "water":        "Very low — 2 irrigations only. Crown root initiation and booting stage",
        "fertilizer":   "Lower N than wheat. Apply P at sowing. K rarely needed in most Indian soils",
        "disease_risk": "Loose smut and stripe rust in cool humid weather. Generally more resistant than wheat",
        "tip":          "Barley tolerates saline and alkaline soils better than any other cereal — ideal for Rajasthan problem soils",
    },
    "sorghum": {
        "season":       "Kharif (June – July sowing)",
        "harvest":      "September – October",
        "water":        "Very low — rain-fed mostly. 1-2 irrigations if dry spell at flowering",
        "fertilizer":   "Moderate NPK at sowing. Sorghum is efficient at extracting nutrients from poor soils",
        "disease_risk": "Shoot fly attacks seedlings in first 2 weeks. Grain mould in wet harvesting conditions",
        "tip":          "Ratoon sorghum from stubble is common for fodder in second cutting",
    },

    # ── Cash crops ───────────────────────────────────────────
    "cotton": {
        "season":       "Kharif (May – June sowing)",
        "harvest":      "October – January (multiple pickings)",
        "water":        "Low to moderate — 5 to 6 irrigations. Very sensitive to waterlogging. Critical at boll development",
        "fertilizer":   "High K need. Apply NPK at sowing. Foliar boron spray at flowering improves boll setting",
        "disease_risk": "Pink bollworm from August. Whitefly spreads leaf curl virus. Monitor with pheromone traps",
        "tip":          "Bt cotton requires fewer pesticide sprays but still needs monitoring for sucking pests",
    },
    "sugarcane": {
        "season":       "Planting: Feb-March (spring) or Oct-Nov (autumn)",
        "harvest":      "12 months after planting",
        "water":        "Very high — 10 to 12 irrigations. Critical during germination and grand growth period July-September",
        "fertilizer":   "Highest N user of all crops. Split in 3-4 doses. Apply zinc sulphate if soil pH above 7.5",
        "disease_risk": "Red rot — use disease-free setts. Smut causes whip-shaped growth. Pyrilla pest August-September",
        "tip":          "Ratoon crop from stubble is viable for 2 years and reduces cost significantly",
    },
    "jute": {
        "season":       "Kharif (March – May sowing)",
        "harvest":      "July – August",
        "water":        "High — rain-fed in West Bengal. Needs 150-200cm rainfall well distributed",
        "fertilizer":   "Moderate NPK. N critical for stem growth. Apply urea in 2 splits",
        "disease_risk": "Stem rot in waterlogged conditions. Semilooper caterpillar defoliates young plants",
        "tip":          "Retting for 20-30 days after harvest — water quality affects fibre quality significantly",
    },

    # ── Vegetables ───────────────────────────────────────────
    "potato": {
        "season":       "Rabi (October – November planting)",
        "harvest":      "February – March",
        "water":        "Moderate — 6 to 8 irrigations. Critical at tuber initiation. Stop 2 weeks before harvest",
        "fertilizer":   "Very high P and K. Apply full dose at planting. Avoid excess N — causes leafy growth",
        "disease_risk": "Late blight — spreads rapidly in cool humid weather above 85% humidity. Early blight in older leaves",
        "tip":          "Use certified seed tubers. Treat cut surface with fungicide before planting",
    },
    "onion": {
        "season":       "Kharif (June-July) or Rabi (Oct-Nov)",
        "harvest":      "Kharif: Oct-Nov  |  Rabi: March-April",
        "water":        "Moderate — 10 to 12 irrigations. Stop when tops fall over (sign of maturity)",
        "fertilizer":   "Apply sulphur — critical for pungency and storage. Split N in 3 doses",
        "disease_risk": "Purple blotch and stemphylium blight in humid conditions. Thrips are major pest",
        "tip":          "Cure bulbs in shade for 7-10 days after harvest to prevent rotting",
    },
    "tomato": {
        "season":       "June-July (Kharif) or September-October (Rabi)",
        "harvest":      "60-70 days after transplanting",
        "water":        "Moderate and regular — drip preferred. Irregular watering causes blossom end rot",
        "fertilizer":   "High P for root development. High K for fruit quality. Calcium spray prevents blossom end rot",
        "disease_risk": "Early blight, late blight, leaf curl virus by whitefly. Fruit borer is major pest",
        "tip":          "Stake plants early. Remove lower leaves touching soil to prevent soil-borne disease",
    },

    # ── Oilseeds ─────────────────────────────────────────────
    "mustard": {
        "season":       "Rabi (October – November sowing)",
        "harvest":      "February – March",
        "water":        "Low — 2 to 3 irrigations only. Excess water causes lodging",
        "fertilizer":   "Apply sulphur — essential for oil content. Moderate N. P at sowing",
        "disease_risk": "White rust and Alternaria blight. Aphids in February-March",
        "tip":          "Can be intercropped with wheat at 9:1 ratio for extra income with no yield loss",
    },

    # ── Pulses ───────────────────────────────────────────────
    "chickpea": {
        "season":       "Rabi (October – November sowing)",
        "harvest":      "February – March",
        "water":        "Very low — mostly rain-fed. 1 irrigation at pod filling only if very dry",
        "fertilizer":   "Legume — fixes own N. Apply only starter 20 kg N/ha. High P needed",
        "disease_risk": "Fusarium wilt — use resistant varieties. Helicoverpa pod borer is major pest",
        "tip":          "Adds 40-50 kg N/ha to soil — always follow with a cereal crop to benefit from residual N",
    },
    "lentil": {
        "season":       "Rabi (October – November sowing)",
        "harvest":      "March – April",
        "water":        "Very low — 1 to 2 irrigations. Highly drought tolerant once established",
        "fertilizer":   "Minimal N — fixes own. Apply P and zinc at sowing for good nodulation",
        "disease_risk": "Rust and wilt are main diseases. Aphids in cool dry weather",
        "tip":          "Ideal for marginal lands and low rainfall areas. Excellent residual benefit for next crop",
    },
    "pigeonpeas": {
        "season":       "Kharif (June – July sowing)",
        "harvest":      "December – January",
        "water":        "Low — mostly rain-fed. Drought tolerant due to deep root system",
        "fertilizer":   "Minimal N. Apply P and K at sowing. Responds well to zinc",
        "disease_risk": "Fusarium wilt — use wilt-resistant varieties. Maruca pod borer from October",
        "tip":          "Can be intercropped with sorghum or maize in 1:2 ratio for higher combined income",
    },
    "mungbean": {
        "season":       "Kharif (June-July) or Zaid (March-April)",
        "harvest":      "60-70 days after sowing",
        "water":        "Low — 3 to 4 irrigations. Very sensitive to waterlogging",
        "fertilizer":   "Minimal N. P at sowing. Short duration — no split applications needed",
        "disease_risk": "Yellow mosaic virus by whitefly — use resistant varieties. Cercospora leaf spot in humid weather",
        "tip":          "Excellent short-duration crop for gap filling between wheat harvest and kharif season",
    },
    "blackgram": {
        "season":       "Kharif (June-July) or Rabi (September-October)",
        "harvest":      "70-80 days after sowing",
        "water":        "Low — 3 to 4 irrigations. Sensitive to waterlogging at all stages",
        "fertilizer":   "Minimal N. P and K at sowing. Rhizobium seed treatment improves nodulation",
        "disease_risk": "Yellow mosaic virus most damaging. Powdery mildew in cool dry weather",
        "tip":          "Grows well in residual moisture after kharif crops — use as a short-duration catch crop",
    },
    "mothbeans": {
        "season":       "Kharif (June – July sowing)",
        "harvest":      "September – October",
        "water":        "Extremely low — survives on 200-300mm rainfall. Most drought tolerant pulse",
        "fertilizer":   "Minimal — fixes own N. Light P at sowing",
        "disease_risk": "Generally hardy. Cercospora leaf spot in wet conditions. Pod borers near maturity",
        "tip":          "Best pulse for arid/semi-arid zones — Rajasthan and Gujarat. Also used for fodder",
    },
    "kidneybeans": {
        "season":       "Kharif (June-July) in plains, summer crop in hills",
        "harvest":      "90-120 days after sowing",
        "water":        "Moderate — 4 to 5 irrigations. Critical at flowering and pod filling",
        "fertilizer":   "Moderate P and K. Starter N only. Rhizobium inoculation at sowing",
        "disease_risk": "Angular leaf spot and bean common mosaic virus. Bean weevil damages stored grain",
        "tip":          "Important cash crop in hill states — Uttarakhand, Himachal Pradesh, J&K",
    },

    # ── Fruits ───────────────────────────────────────────────
    "banana": {
        "season":       "Can be planted year-round. Avoid extreme summer planting",
        "harvest":      "11-14 months after planting",
        "water":        "Very high — drip irrigation recommended. Cannot tolerate drought or waterlogging",
        "fertilizer":   "Very high NPK demand. Apply in 4-6 splits. K is critical for bunch weight",
        "disease_risk": "Panama wilt (Fusarium) — devastating, soil-borne, no chemical cure. Sigatoka in humid conditions",
        "tip":          "Use tissue culture plants — virus-free, uniform, 20-30% higher yield than suckers",
    },
    "mango": {
        "season":       "Flowering: January-February. Fruit development: March-June",
        "harvest":      "May – July depending on variety",
        "water":        "Low once established. Dry period before flowering is essential for good flowering",
        "fertilizer":   "Apply FYM and fertilizers after harvest in August-September. K improves colour and sweetness",
        "disease_risk": "Powdery mildew at flowering — spray sulphur. Mango hopper is key pest at flowering",
        "tip":          "Never irrigate October-January — dry stress triggers flowering. Most important management practice",
    },
    "grapes": {
        "season":       "Pruning: October (North India) or June (South India)",
        "harvest":      "February – April",
        "water":        "Moderate — drip essential. Deficit irrigation before harvest improves sugar content",
        "fertilizer":   "High K for berry quality. Apply in splits. Zinc and boron sprays at berry set",
        "disease_risk": "Downy and powdery mildew most serious. Anthracnose in rainy season",
        "tip":          "Training system — bower or telephone — determines yield and quality significantly",
    },
    "watermelon": {
        "season":       "Zaid (February – March sowing)",
        "harvest":      "May – June",
        "water":        "Moderate — reduce at fruit ripening for better sweetness",
        "fertilizer":   "High K for fruit quality. P at sowing. Side-dress N at vine formation",
        "disease_risk": "Fusarium wilt in poorly drained soils. Fruit fly is major pest",
        "tip":          "Place straw mat under fruit to prevent rotting and sunscald",
    },
    "muskmelon": {
        "season":       "Zaid (February – March sowing)",
        "harvest":      "May – June",
        "water":        "Moderate — reduce at fruit maturity for sweetness and aroma development",
        "fertilizer":   "Similar to watermelon. Boron spray improves fruit set and quality",
        "disease_risk": "Powdery mildew in dry weather. Aphids transmit mosaic virus",
        "tip":          "Fruit is ready when it detaches easily with gentle pressure",
    },
    "apple": {
        "season":       "Flowering: April-May. Harvest: August-October",
        "harvest":      "August – October depending on variety",
        "water":        "Moderate — critical at petal fall, fruit cell division, and fruit enlargement",
        "fertilizer":   "Apply FYM in autumn. Split N — avoid late applications which delay dormancy",
        "disease_risk": "Scab — spray captan or mancozeb at petal fall. Codling moth is key pest",
        "tip":          "Apple requires 1000-1500 chilling hours below 7°C — only above 1500m elevation in India",
    },
    "orange": {
        "season":       "Flowering: Feb-March. Harvest: November-January",
        "harvest":      "November – January",
        "water":        "Moderate — drip preferred. Stress before flowering improves fruit set",
        "fertilizer":   "Split NPK in 3 doses — Feb, June, Sept. Micronutrient spray every 6 months",
        "disease_risk": "Citrus greening — most serious, spread by psyllid, no cure. Canker in humid conditions",
        "tip":          "Rejuvenation pruning every 5-7 years restores productivity significantly",
    },
    "papaya": {
        "season":       "Year-round. Avoid cold months in North India",
        "harvest":      "9-12 months after planting",
        "water":        "Moderate — very sensitive to waterlogging. Even one day of standing water causes root rot",
        "fertilizer":   "High N for rapid growth. Monthly splits. K improves fruit quality",
        "disease_risk": "Papaya ring spot virus by aphids — most devastating. Powdery mildew on leaves and fruits",
        "tip":          "Plant at least 10% male plants. Hermaphrodite varieties are more reliable",
    },
    "coconut": {
        "season":       "Plant at beginning of rainy season — June-July",
        "harvest":      "Round the year after 6-7 years. 60-80 nuts per palm per year",
        "water":        "High — 150 litres per palm per day in summer. Drip with fertigation gives best results",
        "fertilizer":   "Apply NPK in 2 splits — June and December. Common salt (2 kg/palm) is essential",
        "disease_risk": "Root wilt in Kerala — most serious, no cure. Rhinoceros beetle bores into crown",
        "tip":          "Intercrop with banana, pineapple, or cocoa between palms in the first 5 years",
    },
    "pomegranate": {
        "season":       "Main crop: Mrig bahar (June-July). Hasth bahar: Sept-October",
        "harvest":      "5-7 months after flowering",
        "water":        "Low — drought tolerant. Drip preferred. Stress before flowering for bahar treatment",
        "fertilizer":   "Moderate NPK. Boron and zinc critical for fruit set and quality",
        "disease_risk": "Bacterial blight most serious. Fruit borer and thrips are key pests",
        "tip":          "Bahar treatment — withholding water — controls flowering time for market demand management",
    },

    # ── Beverages and spices ─────────────────────────────────
    "coffee": {
        "season":       "Flowering: Feb-March after dry period. Harvest: November-January",
        "harvest":      "November – January",
        "water":        "Moderate — needs a dry period before flowering. Irrigation during berry development",
        "fertilizer":   "Apply in 3 splits — April, August, November. Shade trees reduce fertilizer need",
        "disease_risk": "White stem borer most serious pest. Coffee leaf rust major disease — spray copper fungicide",
        "tip":          "Shade-grown coffee under silver oak gives better quality beans and climate resilience",
    },
    "turmeric": {
        "season":       "Kharif (May – June planting)",
        "harvest":      "January – March (8-9 months)",
        "water":        "High — 15-20 irrigations if rain insufficient",
        "fertilizer":   "High K for rhizome development. Heavy FYM before planting. Split N in 3 doses",
        "disease_risk": "Rhizome rot in waterlogged conditions — most serious. Shoot borer June-October",
        "tip":          "Cure rhizomes by boiling before drying — improves colour and prevents sprouting in storage",
    },
}


# ==============================================================
# SECTION 5 — DISEASE RISK WARNINGS
# ==============================================================
# These are CONTEXTUAL warnings — they only trigger when the
# farmer's actual climate conditions match known disease outbreak
# conditions for their recommended crop.
#
# Format: crop → list of (condition_function, warning_message)
#
# The condition function takes (humidity, rainfall, temperature)
# and returns True if disease risk is elevated.
#
# Example: wheat rust triggers when humidity > 78% AND temp < 20°C
# — these are the exact conditions where yellow rust spreads fast.
#
# NOTE FOR DEVELOPERS: These lambda functions CANNOT be saved
# to pickle or JSON files. They are evaluated in-memory only.
# If you ever need to serialize disease conditions, convert
# them to threshold dictionaries instead.

DISEASE_RISK_CONDITIONS = {
    "wheat":   [
        # Yellow rust needs cool + humid conditions in Feb-March
        (lambda h, r, t: h > 78 and t < 20,
         "High rust risk — cool humid conditions favour yellow rust. Monitor weekly from February"),
        # Karnal bunt spores spread in high humidity at heading
        (lambda h, r, t: h > 85,
         "Karnal bunt risk — spray propiconazole at flag leaf stage"),
    ],
    "rice":    [
        # Blast fungus thrives in high humidity + warm temperatures
        (lambda h, r, t: h > 85 and t > 25,
         "Blast disease risk — apply tricyclazole at tillering stage"),
        # Brown planthopper (BPH) outbreak conditions
        (lambda h, r, t: t > 30 and h > 80,
         "Brown planthopper conditions — monitor weekly, avoid excess N"),
    ],
    "potato":  [
        # Late blight (Phytophthora) needs cool + very humid conditions
        (lambda h, r, t: h > 85 and t < 20,
         "Late blight alert — ideal conditions. Apply mancozeb every 7 days preventively"),
    ],
    "cotton":  [
        # Whitefly (Bemisia tabaci) thrives in any high humidity
        (lambda h, r, t: h > 75,
         "Whitefly and leaf curl virus risk — install yellow sticky traps, avoid excess N"),
    ],
    "tomato":  [
        # Both early and late blight spread rapidly in high humidity
        (lambda h, r, t: h > 80,
         "Early and late blight conditions — spray copper oxychloride, improve air circulation"),
    ],
    "mustard": [
        # Aphids (Lipaphis erysimi) peak in cool + moderately humid conditions
        (lambda h, r, t: h > 70 and t < 18,
         "Aphid and white rust risk — spray imidacloprid at first sign of aphid colonies"),
    ],
    "banana":  [
        # Sigatoka leaf spot (Mycosphaerella musicola) in very humid conditions
        (lambda h, r, t: h > 85,
         "Sigatoka leaf spot conditions — remove infected leaves, spray copper fungicide"),
    ],
    "mango":   [
        # Anthracnose (Colletotrichum) in warm humid fruiting season
        (lambda h, r, t: h > 80 and t > 28,
         "Anthracnose risk at fruiting — spray carbendazim before fruit development"),
        # Powdery mildew (Oidium mangiferae) at flowering in cooler humid conditions
        (lambda h, r, t: h > 75 and t < 25,
         "Powdery mildew at flowering — spray wettable sulphur immediately"),
    ],
}


def get_disease_warning(crop: str,
                        humidity: float,
                        rainfall: float,
                        temperature: float) -> List[str]:
    """
    Checks if current climate conditions trigger any disease warnings
    for the recommended crop.

    Returns a list of warning strings (empty list if conditions are normal).
    Called by get_full_advisory() — not directly by the app.

    Parameters:
      crop        : recommended crop name
      humidity    : % humidity from state climate defaults
      rainfall    : mm/year rainfall from state climate defaults
      temperature : °C temperature from state climate defaults
    """
    warnings = []

    # Get the list of (condition_fn, message) for this crop
    # If crop has no disease conditions defined, returns empty list
    for condition_fn, message in DISEASE_RISK_CONDITIONS.get(crop, []):
        if condition_fn(humidity, rainfall, temperature):
            warnings.append(message)

    return warnings


# ==============================================================
# SECTION 6 — KVK / GOVERNMENT CONTACTS
# ==============================================================
# Krishi Vigyan Kendras (KVKs) are government farm science centres.
# There are 731 KVKs across India.
#
# We store the nearest agricultural university per state —
# these have the most reliable extension services and helplines.

KVK_CONTACTS = {
    "Haryana":          "Haryana Agricultural University, Hisar — 01662-289000",
    "Punjab":           "Punjab Agricultural University, Ludhiana — 0161-2401960",
    "Uttar Pradesh":    "CSAUA&T, Kanpur — 0512-2534157",
    "Rajasthan":        "SKRAU, Bikaner — 0151-2250616",
    "Madhya Pradesh":   "JNKVV, Jabalpur — 0761-2681706",
    "Maharashtra":      "PDKV, Akola — 0724-2258370",
    "Gujarat":          "AAU, Anand — 02692-261304",
    "West Bengal":      "BCKV, Mohanpur — 03473-222275",
    "Bihar":            "BAU, Sabour — 06421-222480",
    "Odisha":           "OUAT, Bhubaneswar — 0674-2397700",
    "Karnataka":        "UAS, Bengaluru — 080-23330153",
    "Andhra Pradesh":   "ANGRAU, Guntur — 0863-2293829",
    "Tamil Nadu":       "TNAU, Coimbatore — 0422-6611200",
    "Kerala":           "KAU, Thrissur — 0487-2370019",
    "Assam":            "AAU, Jorhat — 0376-2340012",
    "Jharkhand":        "BAU, Ranchi — 0651-2450086",
    "Chhattisgarh":     "IGKV, Raipur — 0771-2443524",
    "Telangana":        "PJTSAU, Hyderabad — 040-24015346",
    "Himachal Pradesh": "UHF, Solan — 01792-252167",
    "Uttarakhand":      "VCSG UUHF, Bharsar — 01370-245202",
}

# National toll-free helpline — fallback if state not in our list
NATIONAL_HELPLINE = "Kisan Call Centre: 1800-180-1551 (toll-free, 6AM-10PM)"


def get_kvk_contact(state: str) -> str:
    """Returns the nearest agriculture university contact for a state."""
    return KVK_CONTACTS.get(state, NATIONAL_HELPLINE)


# ==============================================================
# SECTION 7 — FULL ADVISORY ASSEMBLER
# ==============================================================
# This is the main function the Streamlit app calls.
# It combines static crop advice + contextual disease warnings
# + government contact into one clean dictionary.

def get_full_advisory(crop: str,
                      state: str = None,
                      humidity: float = 65,
                      rainfall: float = 500,
                      temperature: float = 25) -> dict:
    """
    Assembles the complete advisory for a recommended crop.

    Merges:
      1. Static crop advice from CROP_ADVISORY
      2. Contextual disease warnings (if climate triggers them)
      3. Nearest KVK government contact

    Default climate values (65% humidity, 500mm rain, 25°C) are
    used only if neither state nor explicit climate is provided.
    In practice, the app always passes state so actual climate
    defaults are used.

    Returns a dict with: season, harvest, water, fertilizer,
    disease_risk, tip, warnings (list), contact (string).
    """
    # Get the static advisory for this crop
    advisory = CROP_ADVISORY.get(crop, {})

    # Get contextual disease warnings based on current conditions
    warnings = get_disease_warning(crop, humidity, rainfall, temperature)

    # Get nearest government agriculture contact
    contact  = get_kvk_contact(state) if state else NATIONAL_HELPLINE

    return {
        "crop":         crop,
        "season":       advisory.get("season",       "Consult local KVK"),
        "harvest":      advisory.get("harvest",      "Consult local KVK"),
        "water":        advisory.get("water",         "Consult local KVK"),
        "fertilizer":   advisory.get("fertilizer",   "Consult local KVK"),
        "disease_risk": advisory.get("disease_risk", "Consult local KVK"),
        "tip":          advisory.get("tip",           "Consult local KVK"),
        "warnings":     warnings,   # empty list if no elevated risk
        "contact":      contact,
    }