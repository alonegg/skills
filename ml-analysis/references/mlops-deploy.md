# MLOps & Deployment Guide

Reference for serializing, serving, containerizing, and monitoring ML models in production.

## Table of Contents
1. [Model Serialization](#model-serialization)
2. [REST API Serving](#rest-api-serving)
3. [Docker Containerization](#docker-containerization)
4. [Monitoring & Drift Detection](#monitoring--drift-detection)
5. [Deployment Checklist](#deployment-checklist)

---

## Model Serialization

### sklearn / LightGBM / XGBoost → joblib

```python
import joblib

# Save
joblib.dump(pipeline, "models/pipeline_v1.joblib")

# Load
pipeline = joblib.load("models/pipeline_v1.joblib")
predictions = pipeline.predict(X_new)
```

**Always save the full pipeline**, not just the model — the preprocessor must travel with it.

### PyTorch → torch.save

```python
import torch

# Save weights only (recommended — portable across refactors)
torch.save(model.state_dict(), "models/model_v1.pt")

# Load
model = MyModel(...)
model.load_state_dict(torch.load("models/model_v1.pt", map_location="cpu"))
model.eval()

# Save entire model (easier but fragile)
torch.save(model, "models/model_v1_full.pt")
```

### ONNX — Cross-Framework Export

Use when you need to deploy a PyTorch/TensorFlow model in a language-agnostic runtime (C++, Java, browser).

```python
import torch
import torch.onnx

# Export PyTorch model to ONNX
dummy_input = torch.randn(1, input_size)
torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

# Run with ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/model.onnx")
outputs = session.run(None, {"input": X_new.astype(np.float32)})
```

### TensorFlow / Keras → SavedModel

```python
# Save
model.save("models/model_v1")

# Load
import tensorflow as tf
model = tf.keras.models.load_model("models/model_v1")
```

### Format Decision Guide

| Situation | Format |
|-----------|--------|
| sklearn / LightGBM / XGBoost | joblib |
| PyTorch, Python-only deploy | torch.save (state_dict) |
| Cross-language or edge deploy | ONNX |
| TensorFlow/Keras | SavedModel |
| Sharing with non-ML teams | ONNX or PMML |

---

## REST API Serving

### FastAPI (recommended)

Minimal, async-capable, auto-generates OpenAPI docs.

```python
# src/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(title="ML Model API", version="1.0")

# Load model once at startup
pipeline = joblib.load("models/pipeline_v1.joblib")


class PredictRequest(BaseModel):
    features: List[float]  # or use a dict for named features


class PredictResponse(BaseModel):
    prediction: float
    probability: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = pipeline.predict(X)[0]
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(X)[0, 1])
    return PredictResponse(prediction=float(prediction), probability=probability)


# Run: uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

### Batch Prediction Endpoint

```python
class BatchRequest(BaseModel):
    records: List[dict]  # list of feature dicts

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    df = pd.DataFrame(request.records)
    predictions = pipeline.predict(df).tolist()
    return {"predictions": predictions}
```

### Input Validation Pattern

Always validate inputs at the API boundary:

```python
from pydantic import BaseModel, Field, validator

class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., ge=0)
    category: str = Field(..., pattern="^(A|B|C)$")

    @validator("income")
    def income_must_be_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError("income must be finite")
        return v
```

---

## Docker Containerization

### Dockerfile (Python / FastAPI)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

### Dockerfile for GPU (PyTorch)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8080
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build & Run

```bash
# Build
docker build -t ml-model:v1 .

# Run locally
docker run -p 8080:8080 ml-model:v1

# Test
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6]}'

# Push to registry
docker tag ml-model:v1 registry.example.com/ml-model:v1
docker push registry.example.com/ml-model:v1
```

### docker-compose for local development

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models  # hot-reload models without rebuild
    environment:
      - MODEL_PATH=models/pipeline_v1.joblib
      - LOG_LEVEL=info
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## Monitoring & Drift Detection

### Why Models Degrade

| Cause | Description | Detection |
|-------|-------------|-----------|
| **Data drift** | Input distribution shifts | Compare feature stats: mean, std, KS test |
| **Concept drift** | P(y\|x) changes | Monitor prediction accuracy over time |
| **Label drift** | Target distribution shifts | Monitor prediction distribution |
| **Upstream change** | Feature pipeline changes | Schema validation |

### Logging Predictions for Monitoring

```python
import json
import logging
from datetime import datetime

logger = logging.getLogger("predictions")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    prediction = pipeline.predict(X)[0]

    # Log every prediction for monitoring
    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": request.features,
        "prediction": float(prediction),
    }
    logger.info(json.dumps(log_record))

    return PredictResponse(prediction=float(prediction))
```

### Statistical Drift Detection with Evidently

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Reference: training data  |  Current: recent production data
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=train_df, current_data=production_df)

# Save as HTML dashboard
report.save_html("reports/drift_report.html")

# Check if drift detected
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    print("WARNING: Data drift detected — consider retraining")
```

### Simple KS-Test Drift Check (no extra library)

```python
from scipy import stats

def check_drift(reference: pd.Series, current: pd.Series, threshold: float = 0.05) -> bool:
    """Returns True if drift is detected (p-value < threshold)."""
    stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    return p_value < threshold

# Check each feature
for col in feature_columns:
    drifted = check_drift(train_df[col], production_df[col])
    if drifted:
        print(f"Drift detected in feature: {col}")
```

### Prediction Distribution Monitoring

```python
import matplotlib.pyplot as plt

# Compare training vs production prediction distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train_predictions, bins=50, alpha=0.7, label="Training")
axes[0].hist(prod_predictions, bins=50, alpha=0.7, label="Production")
axes[0].set_title("Prediction Distribution Shift")
axes[0].legend()

# PSI (Population Stability Index) — common in finance/credit
def psi(expected, actual, buckets=10):
    def scale_range(x, min_val, max_val):
        x = np.clip(x, min_val, max_val)
        return (x - min_val) / (max_val - min_val)
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_scaled = scale_range(expected, expected.min(), expected.max())
    actual_scaled = scale_range(actual, expected.min(), expected.max())
    e_pct = np.histogram(expected_scaled, breakpoints)[0] / len(expected)
    a_pct = np.histogram(actual_scaled, breakpoints)[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

psi_score = psi(train_predictions, prod_predictions)
print(f"PSI: {psi_score:.3f}")
# PSI < 0.1: no shift | 0.1–0.2: moderate | > 0.2: significant
```

### Retraining Triggers

Retrain when any of these conditions are met:
- PSI > 0.2 on key features
- KS test p-value < 0.05 on 20%+ of features
- Prediction distribution mean shifts > 2 standard deviations
- Monitored accuracy (if labels available) drops > 5% from baseline
- A scheduled retrain cadence (weekly, monthly) based on domain velocity

---

## Deployment Checklist

Before declaring a model ready for production:

### Model Validation
- [ ] Test set performance meets the business acceptance threshold
- [ ] Confidence intervals computed (k-fold CV)
- [ ] Model compared against current production model (if one exists)
- [ ] Edge cases tested: empty input, all-null features, single row, max-size batch

### Serving
- [ ] API returns correct predictions for known inputs
- [ ] Health endpoint returns 200
- [ ] Input validation rejects malformed requests with 422 (not 500)
- [ ] Response time < SLA under expected load (use `locust` or `k6` for load testing)
- [ ] Model artifact path is configurable via env var (not hardcoded)

### Operations
- [ ] Predictions are logged with timestamp and input features
- [ ] Drift detection pipeline is scheduled (daily or weekly)
- [ ] Alerting configured for p99 latency and error rate
- [ ] Rollback plan documented: old model artifact retained and re-deployable
- [ ] Model card written: training data, metrics, limitations, intended use

### Reproducibility
- [ ] Training code is versioned in git
- [ ] Model artifact is versioned (e.g., `pipeline_v1.2.joblib` or DVC tag)
- [ ] `requirements.txt` or `pyproject.toml` is pinned with exact versions
- [ ] Random seeds are set and documented
