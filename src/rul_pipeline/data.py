from __future__ import annotations

from pathlib import Path

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

