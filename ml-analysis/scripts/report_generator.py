#!/usr/bin/env python3
"""
report_generator.py — ML Analysis Skill

Generates a Jupyter notebook from a completed ML analysis.
Produces tutorial-style notebooks with markdown explanations,
reproducible code cells, and embedded visualizations.

Usage:
    python report_generator.py --type eda --data <path> --output <notebook.ipynb>
    python report_generator.py --type modeling --data <path> --target <col> --output <notebook.ipynb>
    python report_generator.py --type full --data <path> --target <col> --output <notebook.ipynb>

Report types:
    eda      — Exploratory Data Analysis only
    modeling — Model training and evaluation
    full     — Complete end-to-end analysis (EDA + modeling + evaluation)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


# ── Notebook cell helpers ───────────────────────────────────────────────────

def md_cell(text: str) -> dict:
    """Create a markdown cell."""
    lines = [line + "\n" for line in text.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str, tags: list[str] | None = None) -> dict:
    """Create a code cell."""
    lines = [line + "\n" for line in code.strip().split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    metadata = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": lines,
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


# ── Section builders ────────────────────────────────────────────────────────

def setup_cells(data_path: str, target: str | None, random_seed: int) -> list[dict]:
    target_line = f'TARGET = "{target}"  # label column' if target else "TARGET = None  # unsupervised"
    return [
        md_cell(f"""# ML Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

This notebook walks through a complete machine learning analysis pipeline.
Each section includes explanations of *why* each step is taken, not just *what* is done.

---"""),
        md_cell("## 0. Setup\n\nImport libraries and configure global settings."),
        code_cell(f"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Reproducibility — always set seeds
RANDOM_SEED = {random_seed}
np.random.seed(RANDOM_SEED)

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100

# Paths
DATA_PATH = "{data_path}"
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

{target_line}

print("Setup complete.")""", tags=["parameters"]),
    ]


def eda_cells(data_path: str, target: str | None) -> list[dict]:
    target_block = ""
    if target:
        target_block = f"""
# ── Target variable ──────────────────────────────────────────────────────────
print(f"\\nTarget column: {target}")
print(df[TARGET].value_counts())
print(df[TARGET].describe())"""

    return [
        md_cell("""## 1. Exploratory Data Analysis

EDA is non-negotiable. Even if you "just want to train a model", understanding your data first:
- Reveals data quality issues that would silently corrupt training
- Informs the right preprocessing choices
- Prevents data leakage (a major source of over-optimistic results)"""),

        md_cell("### 1.1 Load & Basic Profile"),
        code_cell(f"""df = pd.read_csv(DATA_PATH) if DATA_PATH.endswith(".csv") else pd.read_parquet(DATA_PATH)

print(f"Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print(f"Memory: {{df.memory_usage(deep=True).sum() / 1e6:.1f}} MB")
print("\\nColumn types:")
print(df.dtypes.value_counts())
df.head()"""),

        md_cell("### 1.2 Missing Values\n\nUnderstanding missing patterns guides imputation strategy:\n- **MCAR** (Missing Completely At Random): safe to drop or mean-impute\n- **MAR** (Missing At Random): conditional imputation\n- **MNAR** (Missing Not At Random): requires domain knowledge, may need indicator feature"),
        code_cell("""missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct}).query("count > 0").sort_values("pct", ascending=False)

if missing_df.empty:
    print("No missing values found.")
else:
    display(missing_df)
    # Visualize
    fig, ax = plt.subplots(figsize=(10, max(3, len(missing_df) * 0.4)))
    missing_df["pct"].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png", bbox_inches="tight")
    plt.show()"""),

        md_cell("### 1.3 Distributions\n\nNumeric distributions reveal skew, outliers, and scaling needs.\nCategorical distributions reveal cardinality and imbalance."),
        code_cell(f"""numeric_cols = df.select_dtypes(include="number").columns.tolist()
if TARGET and TARGET in numeric_cols:
    numeric_cols = [c for c in numeric_cols if c != TARGET]

cat_cols = df.select_dtypes(include="object").columns.tolist()

# Numeric histograms
if numeric_cols:
    n = len(numeric_cols)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Feature Distributions", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", bbox_inches="tight")
    plt.show()

# Categorical counts (top 10 per column, max 4 columns shown)
if cat_cols:
    for col in cat_cols[:4]:
        top = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 3))
        top.plot(kind="barh", ax=ax, color="coral")
        ax.set_title(f"{{col}} — top {{len(top)}} values")
        ax.set_xlabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cat_{{col}}.png", bbox_inches="tight")
        plt.show()"""),

        md_cell(f"""### 1.4 Target Variable Analysis
{"" if target else "*No target column specified — skipping.*"}

Understanding the target distribution determines:
- Task type (classification vs. regression)
- Whether class imbalance handling is needed
- The right evaluation metric"""),
        code_cell(f"""if TARGET:
    if df[TARGET].dtype == "object" or df[TARGET].nunique() <= 20:
        # Classification
        counts = df[TARGET].value_counts()
        pct = df[TARGET].value_counts(normalize=True)
        display(pd.DataFrame({{"count": counts, "pct (%)": (pct*100).round(2)}}))

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Target Distribution: {{TARGET}}")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "target_distribution.png", bbox_inches="tight")
        plt.show()

        imbalance = counts.max() / counts.min()
        if imbalance > 5:
            print(f"⚠ Imbalance ratio: {{imbalance:.1f}}x — consider class_weight='balanced' or resampling")
    else:
        # Regression
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df[TARGET].hist(bins=50, ax=axes[0], color="steelblue")
        axes[0].set_title(f"{{TARGET}} Distribution")
        import scipy.stats as stats
        stats.probplot(df[TARGET].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot (vs Normal)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "target_distribution.png", bbox_inches="tight")
        plt.show()
        print(df[TARGET].describe())
else:
    print("No target column specified.")"""),

        md_cell("### 1.5 Correlations\n\nCorrelation analysis identifies:\n- Features strongly predictive of the target (good)\n- Highly correlated feature pairs (potential multicollinearity — may remove one)"),
        code_cell("""if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
    sns.heatmap(corr, mask=mask, annot=len(numeric_cols) <= 15, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", bbox_inches="tight")
    plt.show()

    # High correlation pairs
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.90:
                high_corr.append((corr.columns[i], corr.columns[j], round(r, 3)))
    if high_corr:
        print("Highly correlated pairs (|r| > 0.90):")
        for a, b, r in high_corr:
            print(f"  {a} ↔ {b}: r={r}")"""),
    ]


def modeling_cells(target: str, random_seed: int) -> list[dict]:
    return [
        md_cell(f"""## 2. Data Preparation

### 2.1 Train / Test Split

**Critical rule:** the test set is held out entirely. It is only used for final evaluation.
All preprocessing (scaling, encoding, imputation statistics) is fit on the training set only.
Fitting on the full dataset before splitting is *data leakage* — it produces overly optimistic results."""),

        code_cell(f"""from sklearn.model_selection import train_test_split

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Stratify for classification (ensures class ratios are preserved in both splits)
stratify = y if y.nunique() <= 20 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=stratify,
)

print(f"Train: {{X_train.shape[0]:,}} rows")
print(f"Test:  {{X_test.shape[0]:,}} rows")
if stratify is not None:
    print("\\nClass distribution (train):")
    print(y_train.value_counts(normalize=True).round(3))"""),

        md_cell("""### 2.2 Preprocessing Pipeline

Using `sklearn.Pipeline` ensures:
- Preprocessing is fit on training data only (no leakage)
- The same transformations are applied identically at inference time
- The model and preprocessor travel together as a single artifact"""),

        code_cell("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Identify column types
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(include="object").columns.tolist()

print(f"Numeric features  ({len(numeric_features)}): {numeric_features[:10]}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}")

# Numeric pipeline: impute then scale
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline: impute then encode
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

print("\\nPreprocessor defined.")"""),

        md_cell("""## 3. Modeling

### 3.1 Baseline Model

A baseline is mandatory. It answers the question: *"Is my complex model actually learning anything useful?"*

Even the best model in the world isn't impressive if a majority-class predictor achieves the same score."""),

        code_cell(f"""from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

is_classification = y_train.nunique() <= 20 or y_train.dtype == object

if is_classification:
    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)),
    ])
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    print("=== Baseline (most-frequent) ===")
    print(classification_report(y_test, y_pred_baseline))
else:
    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="mean")),
    ])
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_baseline)
    r2 = r2_score(y_test, y_pred_baseline)
    print(f"=== Baseline (mean predictor) ===")
    print(f"MAE: {{mae:.4f}}")
    print(f"R²:  {{r2:.4f}}")"""),

        md_cell("""### 3.2 Primary Model

**Why gradient boosting?**
For tabular data, gradient boosting (LightGBM, XGBoost) consistently outperforms other methods.
It handles missing values natively, requires minimal preprocessing, and is robust to outliers."""),

        code_cell(f"""try:
    import lightgbm as lgb
    if is_classification:
        model_cls = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                        num_leaves=63, random_state=RANDOM_SEED,
                                        n_jobs=-1, verbose=-1)
    else:
        model_cls = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                       num_leaves=63, random_state=RANDOM_SEED,
                                       n_jobs=-1, verbose=-1)
    print("Using LightGBM")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    if is_classification:
        model_cls = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    else:
        model_cls = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)
    print("LightGBM not found — using sklearn GradientBoosting")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model_cls),
])

pipeline.fit(X_train, y_train)
print("Training complete.")"""),

        md_cell("""## 4. Evaluation

### 4.1 Cross-Validation

Single train/test splits can be misleading — the result depends on which rows ended up in each set.
Cross-validation gives a distribution of scores, which is more reliable."""),

        code_cell(f"""from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np

if is_classification:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = "f1_weighted" if y_train.nunique() > 2 else "f1"
else:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = "neg_mean_absolute_error"

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

print(f"Cross-validation ({{scoring}}):")
print(f"  Scores: {{cv_scores.round(4)}}")
print(f"  Mean:   {{cv_scores.mean():.4f}}")
print(f"  Std:    {{cv_scores.std():.4f}}")
print(f"  95% CI: ({{cv_scores.mean() - 1.96*cv_scores.std():.4f}}, "
      f"{{cv_scores.mean() + 1.96*cv_scores.std():.4f}})")"""),

        md_cell("### 4.2 Test Set Performance\n\nThis is the final, held-out evaluation. Only look at this once."),
        code_cell("""y_pred = pipeline.predict(X_test)

if is_classification:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, ConfusionMatrixDisplay)

    print("=== Test Set Results ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.show()

    # AUC if binary
    if y_train.nunique() == 2 and hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC-ROC: {auc:.4f}")
else:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals.png", bbox_inches="tight")
    plt.show()"""),

        md_cell("""### 4.3 Feature Importance

SHAP (SHapley Additive exPlanations) is the gold standard for model interpretability.
Unlike built-in feature importances, SHAP values:
- Work for any model (black-box compatible)
- Show directional impact (positive/negative)
- Are consistent across features"""),
        code_cell("""try:
    import shap

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    shap_values = explainer.shap_values(X_test_transformed)

    # Feature names after preprocessing
    feature_names = (
        numeric_features +
        categorical_features
    )

    # For binary classification, use class 1 SHAP values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, X_test_transformed, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.show()
except ImportError:
    print("SHAP not installed. Using built-in feature importances instead.")
    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importances = pd.Series(
            model_step.feature_importances_,
            index=numeric_features + categorical_features
        ).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Feature Importances (built-in)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
        plt.show()
except Exception as e:
    print(f"SHAP failed: {e}. Using built-in importances.")"""),
    ]


def save_cells(random_seed: int) -> list[dict]:
    return [
        md_cell("""## 5. Save Model Artifact

The full pipeline (preprocessor + model) is saved together. This ensures that any new data
goes through identical preprocessing at inference time — no manual feature engineering needed."""),

        code_cell("""import joblib
from pathlib import Path

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "pipeline_v1.joblib"
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# Verify round-trip
loaded = joblib.load(model_path)
assert (loaded.predict(X_test[:5]) == pipeline.predict(X_test[:5])).all()
print("Round-trip verification: OK")"""),

        md_cell("""## 6. Summary

| Stage | Status |
|-------|--------|
| EDA | Complete |
| Preprocessing pipeline | Complete |
| Baseline model | Complete |
| Primary model (LightGBM) | Complete |
| Cross-validation | Complete |
| Test evaluation | Complete |
| Feature importance (SHAP) | Complete |
| Model artifact saved | Complete |

**Next steps:**
- Hyperparameter tuning with Optuna (see `references/automl-guide.md`)
- Error analysis: examine the worst predictions
- Deploy via FastAPI (see `references/mlops-deploy.md`)"""),
    ]


# ── Main ────────────────────────────────────────────────────────────────────

def build_notebook(report_type: str, data_path: str, target: str | None, random_seed: int) -> dict:
    cells = []

    cells.extend(setup_cells(data_path, target, random_seed))

    if report_type in ("eda", "full"):
        cells.extend(eda_cells(data_path, target))

    if report_type in ("modeling", "full") and target:
        cells.extend(modeling_cells(target, random_seed))
        cells.extend(save_cells(random_seed))
    elif report_type == "modeling" and not target:
        cells.append(md_cell("> **Error:** `--target` is required for modeling notebooks."))

    return make_notebook(cells)


def main():
    parser = argparse.ArgumentParser(description="ML Analysis Skill — Notebook Report Generator")
    parser.add_argument("--type", choices=["eda", "modeling", "full"], default="full",
                        help="Type of notebook to generate")
    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--target", default=None, help="Target/label column name")
    parser.add_argument("--output", default=None, help="Output notebook path (default: report_<type>.ipynb)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = args.output or f"report_{args.type}.ipynb"

    print(f"Generating {args.type} notebook...")
    notebook = build_notebook(args.type, args.data, args.target, args.seed)

    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=1)

    print(f"Notebook saved to: {output_path}")
    print(f"Run: jupyter notebook {output_path}")


if __name__ == "__main__":
    main()
