import numpy as np
import pandas as pd
import logging
import time
from typing import Tuple, Dict, Any

# =========================
# LOGGER SETUP
# =========================
def setup_logger(name: str = "NeuroFlow", level=logging.INFO):
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


logger = setup_logger()


# =========================
# TIMER DECORATOR
# =========================
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.info(f"{func.__name__} executed in {end - start:.4f}s")
        return result

    return wrapper


# =========================
# DATA VALIDATION
# =========================
def validate_data(X, y=None) -> Tuple[np.ndarray, Any]:
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(X, list):
        X = np.array(X)

    if y is not None:
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)

    if not isinstance(X, np.ndarray):
        raise ValueError("X must be numpy array / DataFrame / list")

    return X, y


# =========================
# MISSING VALUE HANDLER
# =========================
def handle_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == "mean" and df[col].dtype != "object":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(0, inplace=True)

    logger.info("Missing values handled.")
    return df


# =========================
# FEATURE SCALING (MANUAL)
# =========================
def normalize(X: np.ndarray) -> np.ndarray:
    return (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-8)


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X)) / (np.std(X) + 1e-8)


# =========================
# FEATURE ENGINEERING
# =========================
def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    features = [X]

    for d in range(2, degree + 1):
        features.append(np.power(X, d))

    return np.concatenate(features, axis=1)


# =========================
# OUTLIER DETECTION
# =========================
def detect_outliers_zscore(X: np.ndarray, threshold: float = 3.0):
    z_scores = (X - np.mean(X)) / (np.std(X) + 1e-8)
    return np.abs(z_scores) > threshold


# =========================
# DATA SPLIT (CUSTOM)
# =========================
def split_data(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# =========================
# BATCH PROCESSING
# =========================
def batch_predict(model, X: np.ndarray, batch_size: int = 32):
    results = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        preds = model.predict(batch)
        results.extend(preds)

    return np.array(results)


# =========================
# MODEL INSPECTOR
# =========================
def model_info(model) -> Dict[str, Any]:
    return {
        "type": type(model).__name__,
        "params": model.get_params() if hasattr(model, "get_params") else "N/A",
    }


# =========================
# SAFE SAVE / LOAD
# =========================
def save_model(model, path: str):
    import joblib
    joblib.dump(model, path)
    logger.info(f"Model saved at {path}")


def load_model(path: str):
    import joblib
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
