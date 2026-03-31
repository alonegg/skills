"""
model.py — Model construction and training.
"""

import logging

import joblib
import yaml
from pathlib import Path
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_model(config: dict, is_classification: bool):
    """Instantiate the model specified in config."""
    model_type = config["model"]["type"]
    seed = config["data"]["random_seed"]
    n_est = config["model"]["n_estimators"]
    lr = config["model"]["learning_rate"]

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            if is_classification:
                return lgb.LGBMClassifier(
                    n_estimators=n_est, learning_rate=lr,
                    num_leaves=config["model"]["num_leaves"],
                    random_state=seed, n_jobs=-1, verbose=-1,
                )
            return lgb.LGBMRegressor(
                n_estimators=n_est, learning_rate=lr,
                num_leaves=config["model"]["num_leaves"],
                random_state=seed, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to sklearn GradientBoosting")
            model_type = "sklearn_gbm"

    if model_type in ("sklearn_gbm", "sklearn_rf"):
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            RandomForestClassifier, RandomForestRegressor,
        )
        if model_type == "sklearn_rf":
            cls = RandomForestClassifier if is_classification else RandomForestRegressor
        else:
            cls = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
        return cls(n_estimators=n_est, random_state=seed)

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge()

    raise ValueError(f"Unknown model type: {model_type}")


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")
    return pipeline


def save(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved to {path}")


def load(path: str) -> Pipeline:
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
