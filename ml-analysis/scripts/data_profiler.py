#!/usr/bin/env python3
"""
data_profiler.py — ML Analysis Skill

Generates a concise data profile report for any tabular dataset.
Prints results to stdout in a format Claude can read and interpret.

Usage:
    python data_profiler.py <data-path> [--target <column>] [--sample <n>]

Supports: .csv, .parquet, .json, .jsonl, .xlsx
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def load_data(path: str, sample: int | None = None):
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # Try CSV as fallback
        df = pd.read_csv(path, low_memory=False)

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
        print(f"[INFO] Sampled {sample:,} rows from {len(df):,} total\n")

    return df


def infer_data_modality(df) -> str:
    """Guess whether data is tabular, text-heavy, or time-series."""
    import pandas as pd

    col_names = [c.lower() for c in df.columns]

    # Time series signals
    time_keywords = {"date", "time", "timestamp", "datetime", "period", "year", "month", "day"}
    if any(any(kw in col for kw in time_keywords) for col in col_names):
        return "tabular/time-series"

    # Text-heavy signals
    text_cols = df.select_dtypes(include="object")
    if len(text_cols.columns) > 0:
        avg_len = text_cols.apply(lambda s: s.dropna().str.len().mean()).mean()
        if avg_len > 100:
            return "tabular/text-rich"

    return "tabular"


def profile_column(series, name: str) -> dict:
    import numpy as np
    import pandas as pd

    dtype = str(series.dtype)
    n = len(series)
    n_null = int(series.isnull().sum())
    null_pct = round(n_null / n * 100, 2) if n > 0 else 0.0
    n_unique = int(series.nunique())

    result = {
        "name": name,
        "dtype": dtype,
        "null_count": n_null,
        "null_pct": null_pct,
        "unique_count": n_unique,
    }

    if pd.api.types.is_bool_dtype(series):
        # Cast bool to int so quantile/arithmetic work correctly
        series = series.astype("int8")

    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna()
        if len(vals) > 0:
            result.update(
                {
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                    "min": round(float(vals.min()), 4),
                    "p25": round(float(vals.quantile(0.25)), 4),
                    "median": round(float(vals.median()), 4),
                    "p75": round(float(vals.quantile(0.75)), 4),
                    "max": round(float(vals.max()), 4),
                    "skew": round(float(vals.skew()), 4),
                    "kurtosis": round(float(vals.kurtosis()), 4),
                }
            )
            # Outlier detection via IQR
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            result["n_outliers_iqr"] = n_outliers
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        top = series.value_counts().head(5)
        result["top_values"] = {str(k): int(v) for k, v in top.items()}
        result["cardinality_level"] = (
            "low" if n_unique <= 10 else "medium" if n_unique <= 100 else "high"
        )
    elif pd.api.types.is_datetime64_any_dtype(series):
        vals = series.dropna()
        if len(vals) > 0:
            result["min_date"] = str(vals.min())
            result["max_date"] = str(vals.max())
            result["date_range_days"] = int((vals.max() - vals.min()).days)

    return result


def profile_target(series, name: str) -> dict:
    import pandas as pd

    result = {"name": name}

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 20:
            result["inferred_task"] = "classification"
            result["class_distribution"] = series.value_counts(normalize=True).round(4).to_dict()
            result["n_classes"] = int(n_unique)
            result["is_binary"] = n_unique == 2
            # Imbalance ratio
            counts = series.value_counts()
            ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
            result["imbalance_ratio"] = round(float(ratio), 2)
        else:
            result["inferred_task"] = "regression"
            vals = series.dropna()
            result["mean"] = round(float(vals.mean()), 4)
            result["std"] = round(float(vals.std()), 4)
            result["min"] = round(float(vals.min()), 4)
            result["max"] = round(float(vals.max()), 4)
            result["skew"] = round(float(vals.skew()), 4)
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        result["inferred_task"] = "classification"
        result["class_distribution"] = series.value_counts(normalize=True).round(4).to_dict()
        result["n_classes"] = int(series.nunique())
        result["is_binary"] = series.nunique() == 2
        counts = series.value_counts()
        ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
        result["imbalance_ratio"] = round(float(ratio), 2)

    return result


def detect_duplicates(df) -> dict:
    n_dup = int(df.duplicated().sum())
    return {"duplicate_rows": n_dup, "duplicate_pct": round(n_dup / len(df) * 100, 2)}


def detect_constant_columns(df) -> list[str]:
    return [col for col in df.columns if df[col].nunique() <= 1]


def detect_high_cardinality(df, threshold: float = 0.95) -> list[str]:
    import pandas as pd

    result = []
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) > threshold:
            result.append(col)
    return result


def compute_correlation_flags(df) -> list[dict]:
    """Find highly correlated numeric feature pairs (|r| > 0.95)."""
    import pandas as pd

    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs()
    flags = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r > 0.95:
                flags.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(float(r), 4)})
    return flags


def print_report(profile: dict) -> None:
    sep = "=" * 60

    print(sep)
    print("DATA PROFILE REPORT")
    print(sep)

    meta = profile["metadata"]
    print(f"\nFile       : {meta['source']}")
    print(f"Rows       : {meta['n_rows']:,}")
    print(f"Columns    : {meta['n_cols']}")
    print(f"Memory     : {meta['memory_mb']:.2f} MB")
    print(f"Modality   : {meta['modality']}")

    dup = profile["duplicates"]
    print(f"Duplicates : {dup['duplicate_rows']:,} rows ({dup['duplicate_pct']}%)")

    if profile.get("constant_columns"):
        print(f"\nWARNING — Constant columns (zero variance): {profile['constant_columns']}")

    if profile.get("high_cardinality_columns"):
        print(f"WARNING — High-cardinality string columns: {profile['high_cardinality_columns']}")

    if profile.get("correlation_flags"):
        print("\nHighly Correlated Feature Pairs (|r| > 0.95):")
        for pair in profile["correlation_flags"]:
            print(f"  {pair['col_a']} ↔ {pair['col_b']}  r={pair['correlation']}")

    print(f"\n{'─'*60}")
    print("COLUMN SUMMARY")
    print(f"{'─'*60}")

    header = f"{'Column':<30} {'Dtype':<12} {'Nulls%':>7} {'Unique':>8}"
    print(header)
    print("─" * 60)

    for col_info in profile["columns"]:
        null_flag = " ⚠" if col_info["null_pct"] > 20 else ""
        print(
            f"{col_info['name'][:29]:<30} {col_info['dtype'][:11]:<12} "
            f"{col_info['null_pct']:>6.1f}%{null_flag:2} {col_info['unique_count']:>8,}"
        )

    if profile.get("target"):
        t = profile["target"]
        print(f"\n{'─'*60}")
        print(f"TARGET COLUMN: {t['name']}")
        print(f"{'─'*60}")
        print(f"Inferred task : {t.get('inferred_task', 'unknown')}")
        if "n_classes" in t:
            print(f"Classes       : {t['n_classes']} ({'binary' if t.get('is_binary') else 'multi-class'})")
            print(f"Imbalance ratio: {t.get('imbalance_ratio', 'N/A')}")
            if t.get("imbalance_ratio", 1) > 5:
                print("  ⚠ Significant class imbalance detected — consider class_weight or resampling")
            print("  Class distribution:")
            for cls, pct in list(t.get("class_distribution", {}).items())[:10]:
                bar = "█" * int(pct * 30)
                print(f"    {str(cls)[:20]:<20} {pct*100:5.1f}% {bar}")
        elif t.get("inferred_task") == "regression":
            print(f"Mean ± Std    : {t['mean']} ± {t['std']}")
            print(f"Range         : [{t['min']}, {t['max']}]")
            if abs(t.get("skew", 0)) > 1:
                print(f"  ⚠ Skewed target (skew={t['skew']:.2f}) — consider log transform")

    print(f"\n{'─'*60}")
    print("RECOMMENDATIONS")
    print(f"{'─'*60}")

    for rec in profile.get("recommendations", []):
        print(f"  • {rec}")

    print(f"\n{sep}\n")


def generate_recommendations(profile: dict) -> list[str]:
    recs = []

    meta = profile["metadata"]
    if meta["n_rows"] < 1000:
        recs.append(
            f"Small dataset ({meta['n_rows']:,} rows) — prefer regularized models, use k-fold CV (k=5+)"
        )
    elif meta["n_rows"] > 100_000:
        recs.append(
            f"Large dataset ({meta['n_rows']:,} rows) — consider sampling for rapid iteration before full training"
        )

    high_null = [c for c in profile["columns"] if c["null_pct"] > 30]
    if high_null:
        cols = [c["name"] for c in high_null]
        recs.append(f"High missing rate (>30%) in: {', '.join(cols)} — investigate missingness mechanism (MCAR/MAR/MNAR)")

    if profile.get("constant_columns"):
        recs.append(f"Drop constant columns before modeling: {profile['constant_columns']}")

    if profile.get("high_cardinality_columns"):
        recs.append(
            f"High-cardinality string columns may be IDs — consider dropping or target-encoding: "
            f"{profile['high_cardinality_columns']}"
        )

    if profile.get("correlation_flags"):
        recs.append(
            f"Highly correlated feature pairs found — consider removing one from each pair to reduce multicollinearity"
        )

    if profile.get("target"):
        t = profile["target"]
        if t.get("imbalance_ratio", 1) > 5:
            recs.append(
                "Class imbalance detected — use class_weight='balanced', SMOTE, or adjust decision threshold"
            )
        if t.get("inferred_task") == "regression" and abs(t.get("skew", 0)) > 1:
            recs.append(
                f"Target is skewed (skew={t['skew']:.2f}) — try log1p transform or model on log scale"
            )

    dup_pct = profile["duplicates"]["duplicate_pct"]
    if dup_pct > 1:
        recs.append(f"Duplicate rows: {dup_pct}% — deduplicate before train/test split")

    outlier_cols = [
        c["name"]
        for c in profile["columns"]
        if c.get("n_outliers_iqr", 0) > 0
        and c.get("n_outliers_iqr", 0) / profile["metadata"]["n_rows"] > 0.02
    ]
    if outlier_cols:
        recs.append(
            f"Significant outliers (>2% of rows) in: {', '.join(outlier_cols[:5])} — inspect before clipping"
        )

    if not recs:
        recs.append("Data looks clean. Proceed to train/test split and modeling.")

    return recs


def run_profile(data_path: str, target_col: str | None = None, sample: int | None = None) -> dict:
    import pandas as pd

    df = load_data(data_path, sample=sample)

    profile = {
        "metadata": {
            "source": data_path,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
            "modality": infer_data_modality(df),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        },
        "duplicates": detect_duplicates(df),
        "constant_columns": detect_constant_columns(df),
        "high_cardinality_columns": detect_high_cardinality(df),
        "correlation_flags": compute_correlation_flags(df),
        "columns": [profile_column(df[col], col) for col in df.columns],
    }

    if target_col:
        if target_col not in df.columns:
            print(f"[WARN] Target column '{target_col}' not found in data. Available: {df.columns.tolist()}")
        else:
            profile["target"] = profile_target(df[target_col], target_col)

    profile["recommendations"] = generate_recommendations(profile)
    return profile


def main():
    parser = argparse.ArgumentParser(description="ML Analysis Skill — Data Profiler")
    parser.add_argument("data_path", help="Path to the data file (.csv, .parquet, .json, .xlsx)")
    parser.add_argument("--target", "-t", default=None, help="Target/label column name")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Sample N rows for large files")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    profile = run_profile(args.data_path, target_col=args.target, sample=args.sample)

    if args.json:
        print(json.dumps(profile, indent=2, default=str))
    else:
        print_report(profile)


if __name__ == "__main__":
    main()
