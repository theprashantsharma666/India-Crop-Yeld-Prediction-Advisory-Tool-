"""
visualization.py
----------------
All matplotlib / seaborn charts for the project.

Sections:
  A. EDA charts         — crop distribution, rainfall, temp, NPK, pH, humidity, heatmap
  B. Yield charts       — global top 20, India trend lines
  C. Model evaluation   — accuracy comparison, confusion matrix, feature importance
  D. Prediction output  — top-3 probability bar chart (used in Streamlit app)

Every function is self-contained: pass the data, get the figure.
Call plt.show() / st.pyplot(fig) at the call site.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np


# ── shared palette & style ─────────────────────
PALETTE_MAIN   = "viridis"
COLOR_N        = "#4C9BE8"
COLOR_P        = "#F4A261"
COLOR_K        = "#2A9D8F"
COLOR_TRAIN    = "#4C9BE8"
COLOR_TEST     = "#2A9D8F"
COLOR_ORIGINAL = "#4C9BE8"
COLOR_NEW      = "#F4A261"


# ════════════════════════════════════════════════
# A. EDA CHARTS
# ════════════════════════════════════════════════

def plot_crop_distribution(df: pd.DataFrame,
                           new_crops: list = None) -> plt.Figure:
    """
    Bar chart — number of samples per crop.
    Highlights newly added synthetic crops in orange if new_crops is provided.
    """
    counts = df["label"].value_counts().sort_index()
    colors = (
        [COLOR_NEW if c in new_crops else COLOR_ORIGINAL for c in counts.index]
        if new_crops
        else COLOR_ORIGINAL
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_title("Rows per crop in combined dataset\n"
                 "(orange = newly added Indian crops)", pad=12)
    ax.set_ylabel("Row count")
    if new_crops:
        from matplotlib.patches import Patch
        legend = [Patch(color=COLOR_ORIGINAL, label="Original 22 crops"),
                  Patch(color=COLOR_NEW,      label="Synthetic Indian crops")]
        ax.legend(handles=legend)
    plt.tight_layout()
    return fig


def plot_rainfall_per_crop(df: pd.DataFrame) -> plt.Figure:
    """Bar chart — average rainfall required per crop (sorted descending)."""
    order = df.groupby("label")["rainfall"].mean().sort_values(
        ascending=False
    ).index

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(data=df, x="label", y="rainfall",
                order=order, palette="Blues_r", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Average rainfall required per crop (mm)")
    ax.set_ylabel("Rainfall (mm)")
    plt.tight_layout()
    return fig


def plot_temperature_per_crop(df: pd.DataFrame) -> plt.Figure:
    """Boxplot — temperature range per crop (sorted by mean)."""
    order = df.groupby("label")["temperature"].mean().sort_values().index

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=df, x="label", y="temperature",
                order=order, palette="YlOrRd", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Temperature range per crop (°C)")
    ax.set_ylabel("Temperature (°C)")
    plt.tight_layout()
    return fig


def plot_npk_per_crop(df: pd.DataFrame) -> plt.Figure:
    """3-panel bar chart — N, P, K requirements per crop."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, nutrient, color in zip(
        axes,
        ["N", "P", "K"],
        [COLOR_N, COLOR_P, COLOR_K]
    ):
        order = df.groupby("label")[nutrient].mean().sort_values(
            ascending=False
        ).index
        sns.barplot(data=df, x="label", y=nutrient,
                    order=order, color=color, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"{nutrient} requirement per crop")
        ax.set_ylabel(f"{nutrient} (kg/ha)")

    plt.suptitle("Soil nutrient requirements per crop",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_ph_and_humidity(df: pd.DataFrame) -> plt.Figure:
    """Side-by-side: pH boxplot (with neutral line) + humidity barplot."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # pH
    order_ph = df.groupby("label")["ph"].mean().sort_values().index
    sns.boxplot(data=df, x="label", y="ph",
                order=order_ph, palette="coolwarm", ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                             rotation=45, ha="right")
    axes[0].set_title("Soil pH requirement per crop")
    axes[0].axhline(7, color="gray", linestyle="--",
                    linewidth=0.8, label="Neutral pH = 7")
    axes[0].legend()

    # Humidity
    order_hm = df.groupby("label")["humidity"].mean().sort_values(
        ascending=False
    ).index
    sns.barplot(data=df, x="label", y="humidity",
                order=order_hm, palette="BuGn_r", ax=axes[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                              rotation=45, ha="right")
    axes[1].set_title("Humidity requirement per crop (%)")
    axes[1].set_ylabel("Humidity (%)")

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                             threshold: float = 0.1) -> plt.Figure:
    """
    Correlation heatmap — weak correlations (|r| < threshold) are hidden.
    """
    corr = df.drop("label", axis=1).corr()
    mask = corr.abs() < threshold

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5,
                mask=mask, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Feature correlation (values |r| < {threshold} hidden)")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════
# B. YIELD CHARTS
# ════════════════════════════════════════════════

def plot_global_top_yield(df_yld: pd.DataFrame,
                          top_n: int = 20) -> plt.Figure:
    """
    Bar chart — top N crops by average global yield (hg/ha).
    """
    col = "hg/ha_yield" if "hg/ha_yield" in df_yld.columns else "Value"
    avg = (
        df_yld.groupby("Item")[col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(x=avg.index, y=avg.values, palette="mako", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"Top {top_n} crops by average global yield (hg/ha)")
    ax.set_ylabel("Yield (hg/ha)")
    plt.tight_layout()
    return fig


def plot_india_yield_trend(df_yld: pd.DataFrame) -> plt.Figure:
    """
    Line chart — yield trend in India over years for 5 major crops.
    """
    crops_to_plot = ["Rice, paddy", "Wheat", "Maize", "Potatoes", "Sugarcane"]
    col           = "hg/ha_yield" if "hg/ha_yield" in df_yld.columns else "Value"

    india    = df_yld[df_yld["Area"] == "India"]
    filtered = india[india["Item"].isin(crops_to_plot)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for crop in crops_to_plot:
        data = filtered[filtered["Item"] == crop].sort_values("Year")
        if len(data) > 0:
            ax.plot(data["Year"], data[col],
                    marker="o", markersize=3, label=crop)

    ax.set_title("Crop yield trend in India over years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (hg/ha)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_india_yield_quintal(df_yld: pd.DataFrame) -> plt.Figure:
    """
    6-panel line chart — yield trends for 6 key crops
    converted to quintal/acre for Indian farmer context.
    Conversion: hg/ha ÷ 2470 = quintal/acre
    """
    plot_pairs = [
        ("wheat",     "Wheat"),
        ("rice",      "Rice, paddy"),
        ("cotton",    "Seed cotton, unginned"),
        ("banana",    "Bananas"),
        ("maize",     "Maize"),
        ("potato",    "Potatoes"),
    ]
    col        = "hg/ha_yield" if "hg/ha_yield" in df_yld.columns else "Value"
    conversion = 1 / 2470

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes      = axes.flatten()

    for ax, (label, yld_name) in zip(axes, plot_pairs):
        india = df_yld[
            (df_yld["Item"] == yld_name) & (df_yld["Area"] == "India")
        ].sort_values("Year")

        if len(india) > 0:
            y = india[col] * conversion
            ax.plot(india["Year"], y, color=COLOR_N, linewidth=2)
            ax.fill_between(india["Year"], y, alpha=0.15, color=COLOR_N)
            ax.set_title(f"{label.capitalize()} yield in India")
            ax.set_ylabel("Quintal/acre")
            ax.set_xlabel("Year")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{label} — no India data")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    plt.suptitle("Crop yield trends in India (quintal/acre)", fontsize=14)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════
# C. MODEL EVALUATION CHARTS
# ════════════════════════════════════════════════

def plot_model_comparison(results: dict) -> plt.Figure:
    """
    Grouped bar chart — train vs test accuracy for all models.
    results = {model_name: {'train_acc': float, 'test_acc': float}}
    """
    names      = list(results.keys())
    train_accs = [results[n]["train_acc"] for n in names]
    test_accs  = [results[n]["test_acc"]  for n in names]
    x          = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, train_accs, width=0.4,
           label="Train", color=COLOR_TRAIN)
    ax.bar(x + 0.2, test_accs,  width=0.4,
           label="Test",  color=COLOR_TEST)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.75, 1.01)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison — train vs test accuracy")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list,
                          model_name: str = "Best model") -> plt.Figure:
    """
    Annotated heatmap confusion matrix.
    cm = sklearn confusion_matrix output.
    """
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", linewidths=0.3, ax=ax)
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.set_ylabel("Actual crop")
    ax.set_xlabel("Predicted crop")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray,
                            feature_names: list) -> plt.Figure:
    """
    Horizontal bar chart — feature importance scores.
    Most important feature highlighted in orange.
    """
    pairs  = sorted(zip(feature_names, importances),
                    key=lambda x: x[1], reverse=True)
    feats, imps = zip(*pairs)
    colors = [COLOR_P if i == 0 else COLOR_N for i in range(len(feats))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feats, imps, color=colors)
    ax.set_xlabel("Importance score")
    ax.set_title("Which soil/climate factor matters most?")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════
# D. PREDICTION OUTPUT CHART (for Streamlit app)
# ════════════════════════════════════════════════

def plot_top3_probabilities(top3: list) -> plt.Figure:
    """
    Horizontal bar chart for the app's prediction output.
    top3 = list of {'crop': str, 'confidence': float} dicts (3 items).
    """
    crops  = [d["crop"].capitalize() for d in top3]
    confs  = [d["confidence"] for d in top3]
    colors = [COLOR_K, COLOR_N, "#AAAAAA"]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(crops[::-1], confs[::-1], color=colors[::-1])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 3 crop recommendations")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    for bar, conf in zip(bars, confs[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{conf}%", va="center", fontsize=10)

    plt.tight_layout()
    return fig