"""
main.py
-------
Orchestrates the complete training pipeline.

Think of this file as the DIRECTOR — it doesn't do any heavy
lifting itself. It calls the right functions from the right
files in the right order.

Run this in Google Colab cell by cell, or as a full script.

Pipeline:
  Stage 1 — Load & build combined dataset
  Stage 2 — EDA (text + charts)
  Stage 3 — Train & compare models
  Stage 4 — Evaluate best model
  Stage 5 — Yield engine test
  Stage 6 — Advisory engine test
  Stage 7 — Save all artefacts to /content/

Imports from:
  preprocessor.py  — data loading, cleaning, input assembly
  helper.py        — yield, income, advisory, disease warnings
  eda.py           — statistical analysis functions
  visualization.py — all matplotlib/seaborn charts
"""

import os
import json
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import kagglehub

# Project modules — each file has one clear responsibility
import preprocessor as pre   # data preparation
import helper       as hlp   # output enrichment (yield, advisory)
import eda                    # text-based data analysis
import visualization as viz  # all charts


# ─────────────────────────────────────────────
# CONFIG
# All important settings in one place.
# Change these here — nowhere else.
# ─────────────────────────────────────────────

# OUTPUT_DIR is where all saved files go — model pkl, json data files, etc.
# os.path.dirname(__file__) gets the folder where this script lives.
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# The combined 31-crop CSV is saved here after Stage 1
DATASET_PATH = os.path.join(OUTPUT_DIR, "crop_recommendation_india.csv")

# Create the output folder if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 20% of data is held back for testing — never seen during training
TEST_SIZE    = 0.2

# Fixed seed — ensures the same train/test split every time you run
# Without this, results would vary between runs (non-reproducible)
RANDOM_STATE = 42

# ════════════════════════════════════════════════
# STAGE 1 — LOAD & BUILD COMBINED DATASET
# What happens: Downloads 2 Kaggle datasets, adds 9 missing
# Indian crops using ICAR profiles, saves combined CSV.
# ════════════════════════════════════════════════

def stage1_load_data():
    print("\n" + "=" * 55)
    print("STAGE 1 — Loading & building combined dataset")
    print("=" * 55)

    # kagglehub downloads datasets to a local cache folder.
    # If already downloaded, it uses the cached version (fast).
    crop_rec_path = kagglehub.dataset_download(
        "atharvaingle/crop-recommendation-dataset"
    )
    crop_yld_path = kagglehub.dataset_download(
        "patelris/crop-yield-prediction-dataset"
    )
    print(f"Rec path : {crop_rec_path}")
    print(f"Yld path : {crop_yld_path}")

    # Load the two raw CSV files into DataFrames
    # df_rec = 2200 rows, 22 crops, 8 columns
    # df_yld = 28242 rows, 101 countries, 10 crops
    df_rec, df_yld = pre.load_datasets(crop_rec_path, crop_yld_path)

    # Add 9 missing Indian crops (wheat, sugarcane, potato, etc.)
    # using ICAR agronomic profiles. This gives us 31 crops total.
    # save_path saves the combined CSV so we don't regenerate every session.
    df_combined = pre.build_combined_dataset(df_rec, save_path=DATASET_PATH)

    # Save state and soil defaults as JSON files.
    # These are loaded by the Streamlit app later.
    pre.save_defaults_to_json(OUTPUT_DIR)

    print(f"\nCombined dataset: {df_combined.shape[0]} rows, "
          f"{df_combined['label'].nunique()} crops")

    return df_combined, df_yld, crop_yld_path


# ════════════════════════════════════════════════
# STAGE 2 — EDA
# What happens: Runs text analysis + generates 9 charts.
# Purpose: Understand the data before touching the model.
# ════════════════════════════════════════════════

def stage2_eda(df_combined: pd.DataFrame, df_yld: pd.DataFrame):
    print("\n" + "=" * 55)
    print("STAGE 2 — Exploratory Data Analysis")
    print("=" * 55)

    # Text analysis — prints summaries to console
    eda.check_missing(df_combined, "Combined dataset")   # any nulls?
    eda.check_class_balance(df_combined)                  # 100 rows per crop?
    eda.value_range_summary(df_combined)                  # min/max per feature
    eda.per_crop_profile(df_combined)                     # mean values per crop
    eda.correlation_analysis(df_combined, threshold=0.3)  # feature correlations
    eda.crop_overlap_analysis(df_combined, df_yld)        # which crops have yield data?
    eda.india_yield_overview(df_yld)                      # top India crops by yield

    # The 9 newly added crops (wheat, potato, etc.) get highlighted
    # in orange in the crop distribution chart
    new_crops = list(pre.ICAR_PROFILES.keys())

    # ── EDA charts ───────────────────────────────
    # Each viz function returns a Figure — we call plt.show() to display it
    # This pattern works in both Colab and Streamlit (just swap plt.show for st.pyplot)

    fig = viz.plot_crop_distribution(df_combined, new_crops)
    plt.show()

    fig = viz.plot_rainfall_per_crop(df_combined)
    plt.show()

    fig = viz.plot_temperature_per_crop(df_combined)
    plt.show()

    fig = viz.plot_npk_per_crop(df_combined)
    plt.show()

    fig = viz.plot_ph_and_humidity(df_combined)
    plt.show()

    fig = viz.plot_correlation_heatmap(df_combined, threshold=0.1)
    plt.show()

    # ── Yield charts ─────────────────────────────
    # These charts use the yield dataset, not the recommendation dataset
    fig = viz.plot_global_top_yield(df_yld)
    plt.show()

    fig = viz.plot_india_yield_trend(df_yld)
    plt.show()

    # Shows yield in quintal/acre — the unit Indian farmers actually use
    fig = viz.plot_india_yield_quintal(df_yld)
    plt.show()

    print("\n✅ Stage 2 complete.")


# ════════════════════════════════════════════════
# STAGE 3 — TRAIN & COMPARE MODELS
# What happens: Trains 3 models, compares accuracy,
# picks the best one based on test accuracy.
# ════════════════════════════════════════════════

def stage3_train(df_combined: pd.DataFrame):
    print("\n" + "=" * 55)
    print("STAGE 3 — Model training & comparison")
    print("=" * 55)

    # Separate features (X) from target (y)
    X = df_combined.drop("label", axis=1)  # 7 numeric columns
    y = df_combined["label"]               # crop name strings

    # LabelEncoder converts crop names to integers for sklearn
    # e.g. 'apple'=0, 'banana'=1, ..., 'wheat'=30
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Label encoding:")
    for i, crop in enumerate(le.classes_):
        print(f"  {i:2} → {crop}")

    # stratify=y_encoded ensures each crop appears proportionally
    # in both train and test sets — not all wheat in train, none in test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"\nTrain: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    # ── Compare 3 models ─────────────────────────
    # We try 3 different algorithms and compare their performance.
    # Random Forest almost always wins on this type of dataset.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    results = {}
    print(f"\n{'Model':<25} {'Train':>8} {'Test':>8} {'Time':>7}")
    print("-" * 55)

    for name, model in candidates.items():
        t0        = time.time()
        model.fit(X_train, y_train)          # train on 80%
        elapsed   = round(time.time() - t0, 2)
        train_acc = accuracy_score(y_train, model.predict(X_train))  # score on train
        test_acc  = accuracy_score(y_test,  model.predict(X_test))   # score on unseen test
        results[name] = {
            "model": model, "train_acc": train_acc,
            "test_acc": test_acc, "time": elapsed,
        }
        print(f"{name:<25} {train_acc:>8.4f} {test_acc:>8.4f} {elapsed:>6}s")

    # ── Plot comparison ───────────────────────────
    fig = viz.plot_model_comparison(results)
    plt.show()

    # ── Pick best ────────────────────────────────
    # Select the model with the highest test accuracy.
    # We use test accuracy (not train) because train accuracy
    # can be 100% even for overfitting models.
    best_name  = max(results, key=lambda n: results[n]["test_acc"])
    best_model = results[best_name]["model"]
    print(f"\nBest model : {best_name}")
    print(f"Test acc   : {results[best_name]['test_acc']:.4f}")

    # Return everything Stage 4, 5, 7 will need
    return best_model, best_name, le, X, X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════
# STAGE 4 — EVALUATE BEST MODEL
# What happens: Deep evaluation of the winning model —
# full per-class report, confusion matrix, feature importance.
# ════════════════════════════════════════════════

def stage4_evaluate(best_model, best_name: str,
                    le: LabelEncoder,
                    X_test: pd.DataFrame,
                    y_test: np.ndarray,
                    X: pd.DataFrame):
    print("\n" + "=" * 55)
    print(f"STAGE 4 — Evaluating {best_name}")
    print("=" * 55)

    y_pred = best_model.predict(X_test)

    # Classification report shows precision, recall, F1 per crop.
    # This is more informative than a single accuracy number.
    # Any crop with F1 below 0.80 needs attention before deployment.
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix shows which crops get confused with each other.
    # Diagonal = correct. Off-diagonal = mistakes.
    cm  = confusion_matrix(y_test, y_pred)
    fig = viz.plot_confusion_matrix(cm, le.classes_.tolist(), best_name)
    plt.show()

    # Print the most common confusion pairs in text form
    confused = [
        (cm[i][j], le.classes_[i], le.classes_[j])
        for i in range(len(le.classes_))
        for j in range(len(le.classes_))
        if i != j and cm[i][j] > 0
    ]
    confused.sort(reverse=True)
    print("\n=== Most confused pairs ===")
    for count, actual, predicted in confused[:8]:
        print(f"  {actual:<15} → {predicted:<15}  ({count}×)")

    # Feature importance — which of the 7 features matters most?
    # Only tree-based models (RF, GBM) support feature_importances_
    # Logistic Regression does not have this attribute
    if hasattr(best_model, "feature_importances_"):
        fig = viz.plot_feature_importance(
            best_model.feature_importances_, X.columns.tolist()
        )
        plt.show()

    print("\n✅ Stage 4 complete.")


# ════════════════════════════════════════════════
# STAGE 5 — YIELD & INCOME ENGINE TEST
# What happens: Tests the prediction + yield + income
# pipeline end-to-end for 6 different Indian states.
# ════════════════════════════════════════════════

def stage5_yield_test(best_model, le: LabelEncoder,
                      feature_cols: list):
    print("\n" + "=" * 55)
    print("STAGE 5 — Yield & income engine test")
    print("=" * 55)

    # Test the complete pipeline for 6 different states with different field sizes
    test_states = [
        ("Haryana",     2.0),   # wheat belt
        ("Punjab",      4.0),   # wheat belt, larger farm
        ("Maharashtra", 3.0),   # cotton belt
        ("West Bengal", 1.5),   # rice belt
        ("Kerala",      1.0),   # coconut belt
        ("Tamil Nadu",  2.0),   # banana belt
    ]

    print(f"\n{'State':<20} {'Crop':<13} {'Conf':>6}  "
          f"{'Yield (q)':>12}  {'Income (₹)':>20}")
    print("-" * 80)

    for state, acres in test_states:
        # Step 1: Build the 7-feature input from state defaults
        inp    = pre.get_full_input("state", state=state)

        # Step 2: Convert to DataFrame with correct column order
        # (must match the column order the model was trained on)
        inp_df = pd.DataFrame([inp])[feature_cols]

        # Step 3: Model predicts crop + confidence
        pred_enc  = best_model.predict(inp_df)[0]
        pred_prob = best_model.predict_proba(inp_df)[0]
        crop      = le.inverse_transform([pred_enc])[0]
        conf      = round(pred_prob[pred_enc] * 100, 1)

        # Step 4: Yield and income estimate from helper.py
        inc = hlp.get_income_estimate(crop, acres=acres)
        if inc["yield_low"]:
            yield_str  = f"{inc['yield_low']}–{inc['yield_high']} q"
            income_str = f"₹{inc['income_low']:,}–₹{inc['income_high']:,}"
        else:
            yield_str  = "no data"
            income_str = "—"

        print(f"{state:<20} {crop.upper():<13} {conf:>5}%  "
              f"{yield_str:>12}  {income_str:>20}")

    print("\n✅ Stage 5 complete.")


# ════════════════════════════════════════════════
# STAGE 6 — ADVISORY ENGINE TEST
# What happens: Tests the farming advisory content
# for 4 state/crop combinations including disease warnings.
# ════════════════════════════════════════════════

def stage6_advisory_test():
    print("\n" + "=" * 55)
    print("STAGE 6 — Advisory engine test")
    print("=" * 55)

    # Test 4 different state/crop combinations that cover North, Central,
    # East, and South India — the main agricultural zones
    samples = [
        ("Haryana",     "wheat"),    # Rabi wheat — Punjab/Haryana heartland
        ("Maharashtra", "cotton"),   # Kharif cotton — Vidarbha region
        ("West Bengal", "rice"),     # Kharif rice — Bengal delta
        ("Kerala",      "coconut"),  # Perennial — Kerala coast
    ]

    for state, crop in samples:
        # get_full_advisory combines:
        #   1. Static crop advice (season, water, fertilizer, tip)
        #   2. Contextual disease warnings (only if climate triggers them)
        #   3. Nearest government KVK contact
        adv = hlp.get_full_advisory(crop, state=state)
        print(f"\n{'─'*50}")
        print(f"Crop: {crop.upper()}  |  State: {state}")
        print(f"Season   : {adv['season']}")
        print(f"Harvest  : {adv['harvest']}")
        # [:70] truncates long text for cleaner console output
        print(f"Water    : {adv['water'][:70]}...")
        print(f"Tip      : {adv['tip'][:70]}...")
        # Disease warnings are only shown if climate conditions trigger them
        for w in adv["warnings"]:
            print(f"⚠️  {w}")
        print(f"Contact  : {adv['contact']}")

    print("\n✅ Stage 6 complete.")


# ════════════════════════════════════════════════
# STAGE 7 — SAVE ALL ARTEFACTS
# What happens: Saves all model files + data files to
# OUTPUT_DIR so the Streamlit app can load them.
# ════════════════════════════════════════════════

def stage7_save(best_model, le: LabelEncoder,
                feature_cols: list):
    print("\n" + "=" * 55)
    print("STAGE 7 — Saving artefacts")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Pickle files ──────────────────────────────
    # Pickle is used for Python objects that can't be stored as text:
    #   - The trained model (RandomForestClassifier)
    #   - The label encoder (keeps the integer ↔ crop name mapping)
    #   - The feature column list (critical: column ORDER must match)
    pickle_files = {
        "crop_model.pkl":       best_model,
        "label_encoder.pkl":    le,
        "feature_columns.pkl":  feature_cols,
    }
    for fname, obj in pickle_files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved → {path}")

    # ── JSON files ────────────────────────────────
    # JSON is used for plain data dictionaries — human-readable,
    # editable without retraining, loadable by any language.
    # MSP prices and advisory content can be updated here without
    # touching the model at all.
    json_files = {
        "msp_prices.json":             hlp.MSP_2024,
        "crop_name_mapping.json":      hlp.CROP_NAME_MAPPING,
        "icar_yield.json":             hlp.ICAR_YIELD_INDIA,
        "crop_advisory.json":          hlp.CROP_ADVISORY,
        "kisan_kendra.json":           hlp.KVK_CONTACTS,
        "state_soil_defaults.json":    pre.STATE_SOIL_DEFAULTS,
        "state_climate_defaults.json": pre.STATE_CLIMATE_DEFAULTS,
        "soil_type_profiles.json":     pre.SOIL_TYPE_PROFILES,
    }
    for fname, data in json_files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            # ensure_ascii=False preserves ₹ and other Unicode characters
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {path}")

    # ── Verify reload ─────────────────────────────
    # Reload the saved files and make one test prediction.
    # If this gives WHEAT for Haryana, everything saved correctly.
    model_v = pickle.load(open(f"{OUTPUT_DIR}/crop_model.pkl",      "rb"))
    le_v    = pickle.load(open(f"{OUTPUT_DIR}/label_encoder.pkl",   "rb"))
    cols_v  = pickle.load(open(f"{OUTPUT_DIR}/feature_columns.pkl", "rb"))

    inp    = pre.get_full_input("state", state="Haryana")
    inp_df = pd.DataFrame([inp])[cols_v]
    pred   = le_v.inverse_transform(model_v.predict(inp_df))[0]
    print(f"\nVerification — Haryana prediction: {pred.upper()}")
    print("✅ All artefacts saved and verified.")

    return cols_v


# ════════════════════════════════════════════════
# FULL RUN
# This block only executes when you run main.py directly.
# It does NOT run when another file imports main.py.
# ════════════════════════════════════════════════

if __name__ == "__main__":
    # Stage 1 — download datasets, build 31-crop combined dataset
    df_combined, df_yld, crop_yld_path = stage1_load_data()

    # Stage 2 — explore and visualise the data
    stage2_eda(df_combined, df_yld)

    # Stage 3 — train 3 models, pick the best one
    best_model, best_name, le, X, X_train, X_test, y_train, y_test = \
        stage3_train(df_combined)

    # feature_cols is the ordered list of column names the model expects
    # Must be saved and used consistently in app.py too
    feature_cols = X.columns.tolist()

    # Stage 4 — classification report + confusion matrix + feature importance
    stage4_evaluate(best_model, best_name, le, X_test, y_test, X)

    # Stage 5 — test full prediction → yield → income pipeline
    stage5_yield_test(best_model, le, feature_cols)

    # Stage 6 — test farming advisory output for 4 state/crop combos
    stage6_advisory_test()

    # Stage 7 — save model pkl files + all JSON data files
    stage7_save(best_model, le, feature_cols)

    print("\n" + "=" * 55)
    print("ALL STAGES COMPLETE — Ready for Streamlit app")
    print("=" * 55)