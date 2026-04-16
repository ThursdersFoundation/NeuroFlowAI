import numpy as np
import pandas as pd
from typing import Optional, Any, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class NeuroFlow:
    """
    NeuroFlowAI Core Engine

    Advanced AI pipeline for:
    - preprocessing
    - training
    - evaluation
    - prediction
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        auto_scale: bool = True,
        verbose: bool = True,
    ):
        self.model = model or LogisticRegression()
        self.test_size = test_size
        self.random_state = random_state
        self.auto_scale = auto_scale
        self.verbose = verbose

        self.pipeline = None
        self.metrics: Dict[str, float] = {}
        self.is_trained = False

    # =========================
    # Internal Logging
    # =========================
    def _log(self, message: str):
        if self.verbose:
            print(f"[NeuroFlow] {message}")

    # =========================
    # Data Validation
    # =========================
    def _validate_data(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
        return X, y

    # =========================
    # Build Pipeline
    # =========================
    def _build_pipeline(self):
        steps = []

        if self.auto_scale:
            steps.append(("scaler", StandardScaler()))

        steps.append(("model", self.model))

        self.pipeline = Pipeline(steps)
        self._log("Pipeline built successfully.")

    # =========================
    # Train Model
    # =========================
    def train(self, X, y):
        X, y = self._validate_data(X, y)

        self._log("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self._build_pipeline()

        self._log("Training model...")
        self.pipeline.fit(X_train, y_train)

        self._log("Evaluating model...")
        predictions = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        self.metrics["accuracy"] = acc
        self.is_trained = True

        self._log(f"Training complete. Accuracy: {acc:.4f}")

        return self.metrics

    # =========================
    # Predict
    # =========================
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        X, _ = self._validate_data(X)

        self._log("Running prediction...")
        return self.pipeline.predict(X)

    # =========================
    # Predict Probabilities
    # =========================
    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("Model does not support probability prediction.")

        X, _ = self._validate_data(X)

        return self.pipeline.predict_proba(X)

    # =========================
    # Evaluate on Custom Data
    # =========================
    def evaluate(self, X, y):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        X, y = self._validate_data(X, y)

        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)

        self._log(f"Evaluation Accuracy: {acc:.4f}")
        return {"accuracy": acc}

    # =========================
    # Model Info
    # =========================
    def summary(self):
        if not self.is_trained:
            return "Model not trained yet."

        return {
            "model": self.model.__class__.__name__,
            "metrics": self.metrics,
            "pipeline": self.pipeline.steps,
        }

    # =========================
    # Save Model
    # =========================
    def save(self, path: str = "neuroflow_model.pkl"):
        import joblib

        if not self.is_trained:
            raise RuntimeError("Train model before saving.")

        joblib.dump(self.pipeline, path)
        self._log(f"Model saved to {path}")

    # =========================
    # Load Model
    # =========================
    def load(self, path: str = "neuroflow_model.pkl"):
        import joblib

        self.pipeline = joblib.load(path)
        self.is_trained = True
        self._log(f"Model loaded from {path}")

    # =========================
    # Reset Model
    # =========================
    def reset(self):
        self.pipeline = None
        self.metrics = {}
        self.is_trained = False
        self._log("Model reset complete.")
