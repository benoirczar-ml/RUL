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


def _predict_on_dataframe(
    model_dir: Path,
    metadata: dict,
    data_df: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    model_type = metadata["model_type"]
    df_sorted = data_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    x_all = build_features(df_sorted)

    if model_type == "hist_gbr":
        model = load_model(str(model_dir / "model.joblib"))
        y_pred = predict_hist(model, x_all[metadata["feature_columns"]])
        out = pd.DataFrame(
            {
                "unit": df_sorted["unit"].astype(int).to_numpy(),
                "cycle": df_sorted["cycle"].astype(int).to_numpy(),
                "pred_rul": y_pred.astype(float),
            }
        )
        return out

    if model_type == "lstm_regressor":
        feature_cols = metadata["feature_columns"]
        seq_len = int(metadata["seq_len"])
        x_seq, _, units = build_sequence_samples(
            x_all[feature_cols],
            units=df_sorted["unit"].to_numpy(),
            targets=None,
            seq_len=seq_len,
            sample_step=1,
            last_only=False,
        )
        mean = np.asarray(metadata["scaler_mean"], dtype=np.float32)
        std = np.asarray(metadata["scaler_std"], dtype=np.float32)
        x_seq = apply_window_standardizer(x_seq, mean, std)

        model, resolved_device = load_lstm_checkpoint(str(model_dir / "model.pt"), device=device)
        batch_size = int(metadata["params"].get("batch_size", 256))
        y_pred = predict_lstm(model, x_seq, batch_size=batch_size, device=resolved_device)

        cycles = df_sorted["cycle"].astype(int).to_numpy()
        if len(units) != len(cycles):
            raise ValueError(
                f"Sequence prediction size mismatch: units={len(units)} cycles={len(cycles)} "
                f"for model {model_dir}"
            )
        out = pd.DataFrame({"unit": units.astype(int), "cycle": cycles, "pred_rul": y_pred.astype(float)})
        return out

    raise ValueError(f"Unsupported model_type={model_type}")


def predict_on_dataframe(
    model_dir: str | Path,
    data_df: pd.DataFrame,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    model_dir = Path(model_dir)
    metadata = read_json(model_dir / "metadata.json")
    pred_df = _predict_on_dataframe(model_dir=model_dir, metadata=metadata, data_df=data_df, device=device)
    return pred_df, metadata


def predict_all_cycles(
    model_dir: str | Path,
    data_dir: str | Path | None = None,
    fd: str | None = None,
    split: str = "test",
    device: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    model_dir = Path(model_dir)
    metadata = read_json(model_dir / "metadata.json")
    run_fd = fd or metadata["fd"]
    run_data_dir = Path(data_dir) if data_dir is not None else Path(metadata["data_dir"])
    split_df = load_split(run_data_dir, fd_id=run_fd, split=split)
    pred_df = _predict_on_dataframe(model_dir=model_dir, metadata=metadata, data_df=split_df, device=device)
    return pred_df, metadata


def predict_last_cycle(
    model_dir: str | Path,
    data_dir: str | Path | None = None,
    fd: str | None = None,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    pred_all, metadata = predict_all_cycles(
        model_dir=model_dir,
        data_dir=data_dir,
        fd=fd,
        split="test",
        device=device,
    )
    idx = pred_all.groupby("unit")["cycle"].idxmax().sort_values()
    out = pred_all.loc[idx, ["unit", "pred_rul"]].reset_index(drop=True)
    return out, metadata
