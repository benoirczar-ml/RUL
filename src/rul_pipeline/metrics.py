from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08 score for C-MAPSS.

    d = y_pred - y_true
    - d < 0 (early prediction): exp(-d / 13) - 1
    - d >= 0 (late prediction): exp(d / 10) - 1
    """
    d = y_pred - y_true
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))

