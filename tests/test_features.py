from __future__ import annotations

import numpy as np
import pandas as pd

from rul_pipeline.data import BASE_COLS
from rul_pipeline.features import build_features, feature_columns


def _toy_df() -> pd.DataFrame:
    rows = []
    for unit in [1, 2]:
        for cycle in [1, 2, 3]:
            row = {
                "unit": unit,
                "cycle": cycle,
                "op_setting_1": 0.1 * unit,
                "op_setting_2": 0.2 * cycle,
                "op_setting_3": 0.3,
            }
            for i in range(1, 22):
                row[f"sensor_{i}"] = float(unit + cycle + i)
            rows.append(row)
    return pd.DataFrame(rows)[BASE_COLS]


def test_build_features_shape_and_nan_free() -> None:
    df = _toy_df()
    X = build_features(df)

    assert list(X.columns) == feature_columns()
    assert X.shape[0] == 6
    assert np.isfinite(X.to_numpy()).all()

