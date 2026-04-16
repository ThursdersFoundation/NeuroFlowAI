import numpy as np
import os

from neuroflowai.core import NeuroFlow
from neuroflowai.utils import compute_metrics, batch_predict


# =========================
# BASIC TRAINING TEST
# =========================
def test_training_and_prediction():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = NeuroFlow(verbose=False)
    metrics = model.train(X, y)

    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.5

    preds = model.predict(X)
    assert len(preds) == len(X)


# =========================
# EVALUATION TEST
# =========================
def test_evaluation():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = NeuroFlow(verbose=False)
    model.train(X, y)

    results = model.evaluate(X, y)
    assert "accuracy" in results


# =========================
# PROBABILITY TEST
# =========================
def test_predict_proba():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = NeuroFlow(verbose=False)
    model.train(X, y)

    probs = model.predict_proba(X)
    assert probs.shape[0] == len(X)


# =========================
# SAVE & LOAD TEST
# =========================
def test_save_and_load():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = NeuroFlow(verbose=False)
    model.train(X, y)

    path = "test_model.pkl"
    model.save(path)

    assert os.path.exists(path)

    new_model = NeuroFlow(verbose=False)
    new_model.load(path)

    preds = new_model.predict(X)
    assert len(preds) == len(X)

    os.remove(path)


# =========================
# BATCH PREDICTION TEST
# =========================
def test_batch_prediction():
    X = np.array([[i] for i in range(100)])
    y = np.array([0]*50 + [1]*50)

    model = NeuroFlow(verbose=False)
    model.train(X, y)

    preds = batch_predict(model.pipeline, X, batch_size=10)

    assert len(preds) == len(X)


# =========================
# METRICS TEST
# =========================
def test_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = compute_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "f1_score" in metrics


# =========================
# EDGE CASE: UNTRAINED MODEL
# =========================
def test_untrained_model_error():
    model = NeuroFlow(verbose=False)

    try:
        model.predict([[1]])
        assert False  # should not reach here
    except RuntimeError:
        assert True


# =========================
# SUMMARY TEST
# =========================
def test_summary():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = NeuroFlow(verbose=False)
    model.train(X, y)

    summary = model.summary()

    assert "model" in summary
    assert "metrics" in summary
