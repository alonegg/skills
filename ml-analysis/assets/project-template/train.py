#!/usr/bin/env python3
"""
train.py — End-to-end training script.

Usage:
    python train.py                          # uses configs/config.yaml
    python train.py --config configs/config.yaml
"""

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main(config_path: str) -> None:
    from src.model import load_config, build_model, build_pipeline, train, save
    from src.data import load_raw, split
    from src.features import detect_column_types, build_preprocessor
    from src.evaluate import (
        is_classification_task, evaluate_classification, evaluate_regression,
        save_metrics, cross_validate,
    )

    config = load_config(config_path)
    seed = config["data"]["random_seed"]
    np.random.seed(seed)

    # ── Load & split ────────────────────────────────────────────────────────
    df = load_raw(config["data"]["raw_path"])

    drop_cols = config["features"].get("drop", [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    target = config["data"]["target_column"]
    X_train, X_test, y_train, y_test = split(
        df, target=target,
        test_size=config["data"]["test_size"],
        seed=seed,
    )

    # ── Feature engineering ─────────────────────────────────────────────────
    numeric_cols = config["features"].get("numeric") or None
    cat_cols = config["features"].get("categorical") or None

    if not numeric_cols and not cat_cols:
        numeric_cols, cat_cols = detect_column_types(X_train)

    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:5]}...")
    logger.info(f"Categorical features ({len(cat_cols)}): {cat_cols[:5]}...")

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    # ── Build & train ───────────────────────────────────────────────────────
    is_clf = is_classification_task(y_train)
    model = build_model(config, is_classification=is_clf)
    pipeline = build_pipeline(preprocessor, model)

    # Cross-validate before final training
    cv_results = cross_validate(pipeline, X_train, y_train, cv=config["evaluation"]["cv_folds"], seed=seed)

    # Train on full training set
    pipeline = train(pipeline, X_train, y_train)

    # ── Evaluate on test set ────────────────────────────────────────────────
    output_dir = config["output"]["reports_dir"]

    if is_clf:
        metrics = evaluate_classification(pipeline, X_test, y_test, output_dir=output_dir)
    else:
        metrics = evaluate_regression(pipeline, X_test, y_test, output_dir=output_dir)

    metrics["cross_validation"] = cv_results
    save_metrics(metrics, path=f"{output_dir}/metrics.json")

    # ── Save model ──────────────────────────────────────────────────────────
    model_path = f"{config['output']['model_dir']}/pipeline_v1.joblib"
    save(pipeline, model_path)

    logger.info("Training complete.")
    logger.info(f"Model: {model_path}")
    logger.info(f"Metrics: {output_dir}/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
