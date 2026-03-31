"""
evaluate.py — Model evaluation utilities.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def is_classification_task(y) -> bool:
    return y.nunique() <= 20 or y.dtype == object


def evaluate_classification(pipeline, X_test, y_test, output_dir: str = "reports") -> dict:
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        average_precision_score, ConfusionMatrixDisplay,
    )

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": report["accuracy"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
    }

    if y_test.nunique() == 2 and hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["avg_precision"] = average_precision_score(y_test, y_prob)

    logger.info(f"Classification metrics: {metrics}")

    # Confusion matrix
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return metrics


def evaluate_regression(pipeline, X_test, y_test, output_dir: str = "reports") -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred),
    }

    logger.info(f"Regression metrics: {metrics}")

    # Residual plots
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    return metrics


def save_metrics(metrics: dict, path: str = "reports/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


def cross_validate(pipeline, X_train, y_train, cv: int = 5, seed: int = 42) -> dict:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

    is_clf = is_classification_task(y_train)

    if is_clf:
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scoring = "f1_weighted" if y_train.nunique() > 2 else "f1"
    else:
        kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)
        scoring = "neg_mean_absolute_error"

    scores = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
    result = {
        "scoring": scoring,
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci_95_low": float(scores.mean() - 1.96 * scores.std()),
        "ci_95_high": float(scores.mean() + 1.96 * scores.std()),
    }

    logger.info(
        f"CV {scoring}: {result['mean']:.4f} ± {result['std']:.4f} "
        f"(95% CI: [{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}])"
    )
    return result
