from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class HistGBRConfig:
    max_iter: int = 400
    learning_rate: float = 0.05
    max_depth: int = 8
    min_samples_leaf: int = 30
    l2_regularization: float = 0.0
    random_state: int = 42


def train_hist_gbr(X: pd.DataFrame, y: pd.Series, cfg: HistGBRConfig) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state,
    )
    model.fit(X, y)
    return model


def predict(model: HistGradientBoostingRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def save_model(model: HistGradientBoostingRegressor, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> HistGradientBoostingRegressor:
    return joblib.load(path)

