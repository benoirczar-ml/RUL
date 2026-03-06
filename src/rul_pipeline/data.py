from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPS_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
BASE_COLS = ["unit", "cycle", *OPS_COLS, *SENSOR_COLS]


def _fd_suffix(fd_id: str) -> str:
    fd_clean = fd_id.upper().replace("FD", "")
    if fd_clean not in {"001", "002", "003", "004"}:
        raise ValueError(f"Unsupported fd_id={fd_id}. Expected FD001..FD004.")
    return fd_clean


def load_split(data_dir: str | Path, fd_id: str, split: str) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split={split}. Expected train or test.")

    fd_suffix = _fd_suffix(fd_id)
    file_path = Path(data_dir) / f"{split}_FD{fd_suffix}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=BASE_COLS,
        engine="python",
    )
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_rul_targets(data_dir: str | Path, fd_id: str) -> pd.DataFrame:
    fd_suffix = _fd_suffix(fd_id)
    file_path = Path(data_dir) / f"RUL_FD{fd_suffix}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing RUL file: {file_path}")

    return pd.read_csv(file_path, sep=r"\s+", header=None, names=["rul"], engine="python")


def add_train_rul(df: pd.DataFrame, max_rul: int | None = None) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = max_cycle - out["cycle"]
    if max_rul is not None:
        out["rul"] = out["rul"].clip(upper=max_rul)
    return out


def select_last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


def build_truncated_validation(
    df_with_rul: pd.DataFrame,
    min_prefix_cycles: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create pseudo-test validation by truncating each validation unit trajectory.

    Input dataframe must include columns: unit, cycle, rul.
    Returns:
    - observed_df: concatenated observed prefixes (up to cut cycle per unit)
    - cuts_df: per-unit truncation metadata including true RUL at cut
    """
    required_cols = {"unit", "cycle", "rul"}
    if not required_cols.issubset(df_with_rul.columns):
        raise ValueError(f"df_with_rul must include columns {required_cols}")

    if min_prefix_cycles < 1:
        raise ValueError("min_prefix_cycles must be >= 1.")

    rng = np.random.default_rng(random_state)
    observed_parts: list[pd.DataFrame] = []
    cut_rows: list[dict] = []

    for unit, grp in df_with_rul.groupby("unit", sort=True):
        g = grp.sort_values("cycle").reset_index(drop=True)
        min_cycle = int(g["cycle"].iloc[0])
        max_cycle = int(g["cycle"].iloc[-1])

        min_cut = min_cycle + min_prefix_cycles - 1
        max_cut = max_cycle - 1
        if max_cut < min_cycle:
            max_cut = max_cycle

        if max_cut < min_cut:
            cut_cycle = max_cut
        else:
            cut_cycle = int(rng.integers(min_cut, max_cut + 1))

        observed = g[g["cycle"] <= cut_cycle].copy()
        observed_parts.append(observed)
        cut_rows.append(
            {
                "unit": int(unit),
                "cut_cycle": int(cut_cycle),
                "max_cycle": int(max_cycle),
                "observed_cycles": int(len(observed)),
                "true_rul_at_cut": float(observed.iloc[-1]["rul"]),
            }
        )

    observed_df = pd.concat(observed_parts, axis=0).reset_index(drop=True)
    cuts_df = pd.DataFrame(cut_rows).sort_values("unit").reset_index(drop=True)
    return observed_df, cuts_df
