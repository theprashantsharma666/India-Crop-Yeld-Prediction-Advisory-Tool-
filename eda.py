"""
eda.py
------
Exploratory Data Analysis — text/tabular outputs only.
No charts here (those are in visualization.py).

Covers:
  - Dataset shape and dtypes
  - Missing value checks
  - Class balance verification
  - Value range summaries
  - Crop overlap between the two datasets
  - Per-crop statistical profiles
  - Correlation analysis
  - Yield dataset overview (India filter)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. BASIC DATASET OVERVIEW
# ─────────────────────────────────────────────

def dataset_overview(df_rec: pd.DataFrame, df_yld: pd.DataFrame) -> None:
    """Prints shape, dtypes, and head for both datasets."""
    print("=" * 55)
    print("CROP RECOMMENDATION DATASET")
    print("=" * 55)
    print(f"Shape : {df_rec.shape[0]} rows × {df_rec.shape[1]} cols")
    print(f"Crops : {df_rec['label'].nunique()} unique")
    print(f"Cols  : {df_rec.columns.tolist()}")
    print("\nDtypes:")
    print(df_rec.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df_rec.head(3).to_string())

    print("\n" + "=" * 55)
    print("CROP YIELD DATASET")
    print("=" * 55)
    print(f"Shape     : {df_yld.shape[0]} rows × {df_yld.shape[1]} cols")
    print(f"Countries : {df_yld['Area'].nunique()}")
    print(f"Crops     : {df_yld['Item'].nunique()}")
    print(f"Year range: {df_yld['Year'].min()} – {df_yld['Year'].max()}")
    print("\nDtypes:")
    print(df_yld.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df_yld.head(3).to_string())


# ─────────────────────────────────────────────
# 2. MISSING VALUE CHECK
# ─────────────────────────────────────────────

def check_missing(df: pd.DataFrame, name: str = "Dataset") -> pd.Series:
    """
    Prints missing value counts per column.
    Returns Series of missing counts.
    """
    missing = df.isnull().sum()
    total   = len(df)
    print(f"\n=== Missing Values — {name} ===")
    if missing.sum() == 0:
        print("  ✅ No missing values found.")
    else:
        pct = (missing / total * 100).round(2)
        report = pd.DataFrame({"Missing": missing, "Pct %": pct})
        print(report[report["Missing"] > 0].to_string())
    return missing


# ─────────────────────────────────────────────
# 3. CLASS BALANCE CHECK
# ─────────────────────────────────────────────

def check_class_balance(df: pd.DataFrame,
                        label_col: str = "label") -> pd.Series:
    """
    Checks whether all classes have equal sample counts.
    Returns value_counts Series.
    """
    counts = df[label_col].value_counts().sort_index()
    unique_counts = counts.unique()

    print(f"\n=== Class Balance ===")
    print(f"Total crops : {len(counts)}")
    print(f"Rows/crop   : {unique_counts}")

    if len(unique_counts) == 1:
        print(f"✅ Perfectly balanced — {unique_counts[0]} rows per crop")
    else:
        print("⚠️  Imbalanced dataset — check counts below")
        print(counts.to_string())

    return counts


# ─────────────────────────────────────────────
# 4. VALUE RANGE SUMMARY
# ─────────────────────────────────────────────

def value_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints min/mean/max for all numeric columns.
    Returns describe() DataFrame.
    """
    numeric = df.select_dtypes(include=[np.number])
    desc    = numeric.describe().round(2)
    print("\n=== Value Ranges (numeric features) ===")
    print(desc.to_string())
    return desc


# ─────────────────────────────────────────────
# 5. CROP OVERLAP ANALYSIS
# ─────────────────────────────────────────────

def crop_overlap_analysis(df_rec: pd.DataFrame,
                          df_yld: pd.DataFrame) -> dict:
    """
    Finds which crops appear in both datasets vs only one.
    Returns dict with 'overlap', 'only_rec', 'only_yld' sets.
    """
    rec_crops = set(df_rec["label"].unique())
    yld_crops = set(df_yld["Item"].str.lower().unique())

    overlap   = rec_crops.intersection(yld_crops)
    only_rec  = rec_crops - yld_crops
    only_yld  = yld_crops - rec_crops

    print("\n=== Crop Overlap Analysis ===")
    print(f"Crops in BOTH datasets       : {len(overlap)}")
    print(f"  {sorted(overlap)}")
    print(f"\nCrops ONLY in recommendation : {len(only_rec)}")
    print(f"  {sorted(only_rec)}")
    print(f"\nNote: Crops only in recommendation cannot show yield estimates.")

    return {"overlap": overlap, "only_rec": only_rec, "only_yld": only_yld}


# ─────────────────────────────────────────────
# 6. PER-CROP PROFILE SUMMARY
# ─────────────────────────────────────────────

def per_crop_profile(df: pd.DataFrame,
                     features: list = None) -> pd.DataFrame:
    """
    Shows mean of each feature per crop.
    Useful for quickly comparing crop soil fingerprints.
    """
    if features is None:
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    profile = (
        df.groupby("label")[features]
        .mean()
        .round(1)
        .sort_values("rainfall", ascending=False)
    )

    print("\n=== Per-Crop Feature Profile (mean values) ===")
    print(profile.to_string())
    return profile


# ─────────────────────────────────────────────
# 7. CORRELATION ANALYSIS
# ─────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame,
                         threshold: float = 0.3) -> pd.DataFrame:
    """
    Computes feature correlation matrix.
    Prints pairs with |corr| >= threshold.
    Returns full correlation DataFrame.
    """
    numeric = df.select_dtypes(include=[np.number])
    corr    = numeric.corr().round(3)

    print(f"\n=== Feature Correlation (threshold |r| >= {threshold}) ===")
    pairs = (
        corr.unstack()
        .reset_index()
        .rename(columns={"level_0": "Feature A",
                          "level_1": "Feature B",
                          0: "r"})
    )
    pairs = pairs[
        (pairs["Feature A"] < pairs["Feature B"]) &
        (pairs["r"].abs() >= threshold)
    ].sort_values("r", ascending=False)

    if pairs.empty:
        print(f"  No feature pairs with |r| >= {threshold}")
    else:
        print(pairs.to_string(index=False))

    return corr


# ─────────────────────────────────────────────
# 8. INDIA YIELD OVERVIEW
# ─────────────────────────────────────────────

def india_yield_overview(df_yld: pd.DataFrame) -> pd.DataFrame:
    """
    Filters yield dataset to India and shows
    average yield per crop (sorted highest to lowest).
    """
    india = df_yld[df_yld["Area"] == "India"].copy()
    col   = "hg/ha_yield" if "hg/ha_yield" in india.columns else "Value"

    avg = (
        india.groupby("Item")[col]
        .mean()
        .sort_values(ascending=False)
        .round(0)
        .astype(int)
    )

    print(f"\n=== India Average Yield per Crop (hg/ha) — top 20 ===")
    print(avg.head(20).to_string())
    return avg


# ─────────────────────────────────────────────
# 9. COMBINED DATASET VALIDATION
#    (run after build_combined_dataset)
# ─────────────────────────────────────────────

def validate_combined(df_combined: pd.DataFrame,
                      df_original: pd.DataFrame,
                      new_crop_names: list) -> None:
    """
    Confirms the combined 31-crop dataset looks correct:
    - Row count
    - No missing values
    - Columns unchanged
    - New crops have realistic value ranges
    """
    print("\n=== Combined Dataset Validation ===")
    print(f"Original  : {len(df_original)} rows, "
          f"{df_original['label'].nunique()} crops")
    print(f"Combined  : {len(df_combined)} rows, "
          f"{df_combined['label'].nunique()} crops")
    print(f"All crops : {sorted(df_combined['label'].unique())}")

    missing = df_combined.isnull().sum().sum()
    print(f"Missing values: {'✅ None' if missing == 0 else f'⚠️  {missing}'}")

    col_match = df_combined.columns.tolist() == df_original.columns.tolist()
    print(f"Column order matches: {'✅' if col_match else '⚠️  MISMATCH'}")

    print("\nRows per crop (should be 100 each):")
    print(df_combined["label"].value_counts().sort_index().to_string())

    print("\nSample — wheat profile:")
    print(df_combined[df_combined["label"] == "wheat"].describe().round(2).to_string())