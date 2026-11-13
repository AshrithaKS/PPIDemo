"""Metrics for imbalanced residue-level classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at(y, prob, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(prob) >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if pred.any() else 0.0,
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


def best_mcc_threshold(y, prob, grid=None) -> float:
    """Threshold maximising MCC. Choose it on VALIDATION data only, never on test."""
    grid = np.linspace(0.02, 0.9, 89) if grid is None else grid
    y = np.asarray(y)
    scores = [matthews_corrcoef(y, (prob >= t).astype(int)) if (prob >= t).any() else -1 for t in grid]
    return float(grid[int(np.argmax(scores))])


def trivial_baseline(y) -> dict:
    """Predict every residue non-interacting."""
    y = np.asarray(y)
    return {"accuracy": float(1 - y.mean()), "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0,
            "pr_auc": float(y.mean()), "roc_auc": 0.5}
