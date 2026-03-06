from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_split
from .features import build_features
from .io_utils import read_json
from .modeling import load_model, predict as predict_hist
from .sequence import apply_window_standardizer, build_sequence_samples
from .sequence_model import load_lstm_checkpoint, predict_lstm


def predict_last_cycle(
    model_dir: str | Path,
    data_dir: str | Path | None = None,
    fd: str | None = None,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    model_dir = Path(model_dir)
    metadata = read_json(model_dir / "metadata.json")

    run_fd = fd or metadata["fd"]
    run_data_dir = Path(data_dir) if data_dir is not None else Path(metadata["data_dir"])
    model_type = metadata["model_type"]

    test_raw = load_split(run_data_dir, fd_id=run_fd, split="test")
    x_all = build_features(test_raw)

    if model_type == "hist_gbr":
        model = load_model(str(model_dir / "model.joblib"))
        last_idx = test_raw.groupby("unit")["cycle"].idxmax().sort_values()
        units = test_raw.loc[last_idx, "unit"].to_numpy().astype(int)
        x_last = x_all.loc[last_idx, metadata["feature_columns"]]
        y_pred = predict_hist(model, x_last)

    elif model_type == "lstm_regressor":
        feature_cols = metadata["feature_columns"]
        seq_len = int(metadata["seq_len"])
        x_seq, _, units = build_sequence_samples(
            x_all[feature_cols],
            units=test_raw["unit"].to_numpy(),
            targets=None,
            seq_len=seq_len,
            sample_step=1,
            last_only=True,
        )
        mean = np.asarray(metadata["scaler_mean"], dtype=np.float32)
        std = np.asarray(metadata["scaler_std"], dtype=np.float32)
        x_seq = apply_window_standardizer(x_seq, mean, std)

        model, resolved_device = load_lstm_checkpoint(str(model_dir / "model.pt"), device=device)
        batch_size = int(metadata["params"].get("batch_size", 256))
        y_pred = predict_lstm(model, x_seq, batch_size=batch_size, device=resolved_device)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    out = pd.DataFrame({"unit": units.astype(int), "pred_rul": y_pred.astype(float)})
    return out, metadata

