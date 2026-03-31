# Data Engineering Reference

Reference for building ETL pipelines, ensuring data quality, and processing large-scale data for ML.

## Table of Contents
1. [ETL Pipeline Design](#etl-pipeline-design)
2. [Data Quality](#data-quality)
3. [Large-Scale Processing (Spark / Dask)](#large-scale-processing-spark--dask)
4. [Feature Stores](#feature-stores)
5. [Data Versioning (DVC)](#data-versioning-dvc)

---

## ETL Pipeline Design

An ETL pipeline has three clearly separated stages. Never mix them — it makes debugging and reuse nearly impossible.

```
Raw data sources
      │
  [Extract]   ← read only, no transformations
      │
  [Transform] ← all business logic, cleaning, feature engineering
      │
   [Load]     ← write to target store, validate schema
      │
  ML-ready dataset
```

### Minimal Pipeline Template

```python
# src/pipeline.py
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(source_path: str) -> pd.DataFrame:
    """Load raw data. No transformations here."""
    logger.info(f"Extracting from {source_path}")
    df = pd.read_csv(source_path)
    logger.info(f"Extracted {len(df):,} rows, {df.shape[1]} columns")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """All cleaning and feature engineering. Must be pure/deterministic."""
    df = df.copy()

    # 1. Drop structural junk
    df = df.drop_duplicates()
    df = df.dropna(subset=["id"])  # required key

    # 2. Type coercions
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # 3. Business logic
    df["amount"] = df["amount"].clip(lower=0)  # no negative amounts
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 4. Imputation (fit on train only — see note below)
    df["amount"] = df["amount"].fillna(df["amount"].median())

    logger.info(f"Transform complete: {len(df):,} rows remaining")
    return df


def load(df: pd.DataFrame, target_path: str) -> None:
    """Write to target. Validate schema before writing."""
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    validate_schema(df)
    df.to_parquet(target_path, index=False)
    logger.info(f"Loaded {len(df):,} rows to {target_path}")


def validate_schema(df: pd.DataFrame) -> None:
    required_columns = {"id", "date", "amount", "year", "month"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    if df["id"].isnull().any():
        raise ValueError("Column 'id' must not have nulls after transform")


def run_pipeline(source: str, target: str) -> None:
    raw = extract(source)
    clean = transform(raw)
    load(clean, target)


if __name__ == "__main__":
    run_pipeline("data/raw/data.csv", "data/processed/data.parquet")
```

**Key rule:** Any stateful operation (computing medians, fitting encoders) must be fit on training data only, then applied to test/production data. Never compute statistics on the full dataset before the train/test split.

### Pipeline Patterns

| Pattern | When | Implementation |
|---------|------|----------------|
| **Snapshot** | Daily full reload | Overwrite target each run |
| **Incremental** | Large data, append-only | Filter by `updated_at > last_run` |
| **CDC** | Database replication | Capture insert/update/delete events |
| **Lambda** | Mix real-time + batch | Separate speed layer + batch layer |

---

## Data Quality

### Validation with Great Expectations

The gold standard for data quality checks in ML pipelines.

```python
import great_expectations as ge

# Load data
df = ge.read_csv("data/processed/data.parquet")

# Define expectations
df.expect_column_values_to_not_be_null("id")
df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
df.expect_column_values_to_be_in_set("status", ["active", "churned", "trial"])
df.expect_column_mean_to_be_between("revenue", min_value=0, max_value=10000)
df.expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")

# Run validation
results = df.validate()
print(f"Success: {results['success']}")
if not results["success"]:
    failed = [r for r in results["results"] if not r["success"]]
    for f in failed:
        print(f"  FAIL: {f['expectation_config']['expectation_type']} — {f['result']}")
```

### Lightweight Schema Validation (no extra library)

```python
def validate_dataframe(df: pd.DataFrame, schema: dict) -> list[str]:
    """
    schema = {
        "col_name": {"dtype": "float64", "nullable": False, "min": 0, "max": 100}
    }
    Returns list of error messages. Empty list = valid.
    """
    errors = []
    for col, rules in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue
        if not rules.get("nullable", True) and df[col].isnull().any():
            errors.append(f"Null values in non-nullable column: {col}")
        if "dtype" in rules and str(df[col].dtype) != rules["dtype"]:
            errors.append(f"Column {col}: expected {rules['dtype']}, got {df[col].dtype}")
        if "min" in rules and df[col].min() < rules["min"]:
            errors.append(f"Column {col}: value below min {rules['min']}")
        if "max" in rules and df[col].max() > rules["max"]:
            errors.append(f"Column {col}: value above max {rules['max']}")
    return errors

# Usage
schema = {
    "age": {"dtype": "int64", "nullable": False, "min": 0, "max": 120},
    "revenue": {"dtype": "float64", "nullable": True, "min": 0},
}
errors = validate_dataframe(df, schema)
if errors:
    raise ValueError("\n".join(errors))
```

### Data Quality Metrics to Track

| Metric | How to Compute | Alert Threshold |
|--------|----------------|-----------------|
| Null rate per column | `df.isnull().mean()` | > 20% change from baseline |
| Row count | `len(df)` | < 80% of yesterday's count |
| Unique rate (key columns) | `df["id"].nunique() / len(df)` | < 1.0 (duplicates crept in) |
| Value range violations | `(df["age"] < 0).sum()` | Any violation |
| Schema mismatch | column types vs. expected | Any mismatch |

---

## Large-Scale Processing (Spark / Dask)

### When to Switch from Pandas

| Data Size | Tool |
|-----------|------|
| < 1 GB | pandas |
| 1–50 GB, single machine | Dask or polars |
| > 50 GB, or distributed | Apache Spark |
| Streaming data | Spark Structured Streaming or Flink |

### Dask (pandas-like API, scales to multi-core / multi-machine)

```python
import dask.dataframe as dd

# Read (lazy — nothing loaded yet)
df = dd.read_parquet("data/raw/*.parquet")

# Same API as pandas
df = df[df["amount"] > 0]
df["log_amount"] = df["amount"].map(lambda x: np.log1p(x), meta=("log_amount", "float64"))

# Aggregations
result = df.groupby("category")["amount"].mean().compute()  # compute() triggers execution

# Write
df.to_parquet("data/processed/", write_index=False)
```

**Tips:**
- Use `dd.read_parquet` not `dd.read_csv` — parquet is partitioned and compressed
- Call `.compute()` once at the end, not inside loops
- Use `df.persist()` to cache hot datasets in memory across operations

### PySpark

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("ml-pipeline") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Read
df = spark.read.parquet("s3://bucket/data/raw/")

# Transform (lazy DAG — nothing runs yet)
df = df.filter(F.col("amount") > 0)
df = df.withColumn("log_amount", F.log1p(F.col("amount")))
df = df.groupBy("category").agg(F.mean("amount").alias("avg_amount"))

# Write (triggers execution)
df.write.mode("overwrite").parquet("s3://bucket/data/processed/")

spark.stop()
```

### Spark ML Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import GBTClassifier

# Feature engineering
indexer = StringIndexer(inputCol="category", outputCol="category_idx")
assembler = VectorAssembler(inputCols=["age", "amount", "category_idx"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Model
gbt = GBTClassifier(featuresCol="scaled_features", labelCol="label", maxIter=50)

# Pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, gbt])

# Train / evaluate
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")
```

### Polars (fast single-machine alternative to pandas)

```python
import polars as pl

df = pl.read_parquet("data/raw/*.parquet")

result = (
    df
    .filter(pl.col("amount") > 0)
    .with_columns([
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("date").cast(pl.Date),
    ])
    .group_by("category")
    .agg(pl.col("amount").mean().alias("avg_amount"))
    .sort("avg_amount", descending=True)
)

result.write_parquet("data/processed/aggregated.parquet")
```

Polars is typically 5–20× faster than pandas for single-machine workloads. Syntax is more verbose but very explicit.

---

## Feature Stores

Use a feature store when:
- Multiple models reuse the same features
- Features must be consistent between training and serving (training/serving skew)
- You need point-in-time correct historical features (no data leakage from the future)

### Feast (open-source)

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Retrieve training features (point-in-time correct)
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with entity keys and event_timestamp
    features=[
        "customer_stats:lifetime_value",
        "customer_stats:num_orders_30d",
        "product_features:avg_rating",
    ],
).to_df()

# Retrieve online features for serving (low-latency)
feature_vector = store.get_online_features(
    features=["customer_stats:lifetime_value", "customer_stats:num_orders_30d"],
    entity_rows=[{"customer_id": 1001}],
).to_dict()
```

**Simpler alternative:** If you don't need real-time serving, store precomputed features as versioned Parquet files and join them at training time. This covers 90% of use cases without the operational overhead.

---

## Data Versioning (DVC)

DVC tracks large data files and model artifacts in git without storing the binaries in git itself.

### Setup

```bash
pip install dvc dvc-s3  # or dvc-gcs, dvc-azure

dvc init
dvc remote add -d myremote s3://my-bucket/dvc-store
```

### Basic Workflow

```bash
# Track a data file
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset v1"
dvc push  # uploads to S3

# Later: pull data on a new machine
git pull
dvc pull  # downloads from S3

# Create a pipeline stage
dvc stage add -n preprocess \
  -d src/pipeline.py \
  -d data/raw/dataset.csv \
  -o data/processed/clean.parquet \
  "python src/pipeline.py"

dvc repro       # run the pipeline (only reruns changed stages)
dvc dag         # visualize the pipeline DAG
```

### Experiment Tracking

```bash
# Run experiments with different parameters
dvc exp run --set-param model.learning_rate=0.01
dvc exp run --set-param model.learning_rate=0.001

# Compare
dvc exp show
dvc exp diff
```

### DVC + MLflow Together

DVC handles data/model versioning; MLflow handles experiment metrics. They complement each other:

```python
import mlflow
import dvc.api

# Load a specific version of data via DVC
data_url = dvc.api.get_url("data/processed/clean.parquet", rev="v2.1")
df = pd.read_parquet(data_url)

with mlflow.start_run():
    mlflow.log_param("data_version", "v2.1")
    mlflow.log_param("learning_rate", 0.01)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    mlflow.log_metric("test_accuracy", score)
    mlflow.sklearn.log_model(model, "model")
```
