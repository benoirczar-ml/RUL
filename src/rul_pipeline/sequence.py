from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequence_samples(
    features: pd.DataFrame,
    units: np.ndarray,
    targets: np.ndarray | None,
    seq_len: int,
    sample_step: int = 1,
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")
    if sample_step < 1:
        raise ValueError("sample_step must be >= 1.")

    x_values = features.to_numpy(dtype=np.float32)
    unit_values = units.astype(int)
    y_values = targets.astype(np.float32) if targets is not None else None

    windows: list[np.ndarray] = []
    y_out: list[float] = []
    unit_out: list[int] = []

    for unit in pd.unique(unit_values):
        idx = np.flatnonzero(unit_values == unit)
        x_unit = x_values[idx]
        y_unit = y_values[idx] if y_values is not None else None

        if last_only:
            positions = [len(idx) - 1]
        else:
            positions = list(range(0, len(idx), sample_step))
            if positions[-1] != len(idx) - 1:
                positions.append(len(idx) - 1)

        for pos in positions:
            win = np.zeros((seq_len, x_unit.shape[1]), dtype=np.float32)
            start = max(0, pos - seq_len + 1)
            chunk = x_unit[start : pos + 1]
            win[-len(chunk) :, :] = chunk
            windows.append(win)
            unit_out.append(int(unit))
            if y_unit is not None:
                y_out.append(float(y_unit[pos]))

    x_seq = np.stack(windows)
    unit_arr = np.asarray(unit_out, dtype=np.int64)
    if y_values is None:
        return x_seq, None, unit_arr
    return x_seq, np.asarray(y_out, dtype=np.float32), unit_arr


def fit_window_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_window_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)

