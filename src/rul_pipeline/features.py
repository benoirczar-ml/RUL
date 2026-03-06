from __future__ import annotations

import pandas as pd

from .data import OPS_COLS, SENSOR_COLS


def feature_columns() -> list[str]:
    delta_cols = [f"{col}_d1" for col in SENSOR_COLS]
    return ["cycle", "cycle_norm", *OPS_COLS, *SENSOR_COLS, *delta_cols]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)

    max_cycle_per_unit = out.groupby("unit")["cycle"].transform("max")
    out["cycle_norm"] = out["cycle"] / max_cycle_per_unit.clip(lower=1)

    for col in SENSOR_COLS:
        out[f"{col}_d1"] = out.groupby("unit")[col].diff().fillna(0.0)

    cols = feature_columns()
    return out[cols].astype("float32")

