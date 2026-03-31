"""
data.py — Data loading and preprocessing utilities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """Load raw data from CSV or Parquet. No transformations."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif p.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif p.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")
    return df


def split(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42):
    """
    Train/test split. Stratifies automatically for classification targets.
    All preprocessing must be fit AFTER this call, on X_train only.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Stratify for classification (few unique values)
    stratify = y if y.nunique() <= 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def save_processed(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path}")
