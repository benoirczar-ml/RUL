from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import add_train_rul, load_split
from rul_pipeline.features import build_features, feature_columns
from rul_pipeline.io_utils import ensure_dir, read_json, write_json
from rul_pipeline.metrics import mae, phm_score, rmse
from rul_pipeline.sequence import apply_window_standardizer, build_sequence_samples, fit_window_standardizer
from rul_pipeline.sequence_model import LSTMConfig, predict_lstm, save_lstm_checkpoint, train_lstm_regressor


def _resolve(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def _pick(cli_value, config_dict: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_dict.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C-MAPSS LSTM sequence regressor.")
    parser.add_argument("--config", default="config/train_lstm.json", help="JSON config path.")
    parser.add_argument("--data-dir", default=None, help="Path to CMAPSSData directory.")
    parser.add_argument("--fd", default=None, help="Dataset id: FD001..FD004.")
    parser.add_argument("--max-rul", type=int, default=None, help="Clip train RUL target.")
    parser.add_argument("--val-fraction", type=float, default=None, help="Validation unit split fraction.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--seq-len", type=int, default=None, help="Window length.")
    parser.add_argument("--sample-step", type=int, default=None, help="Training window stride.")
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=None, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda.")
    parser.add_argument(
        "--val-last-cycle-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use last-cycle validation windows for early stopping and primary metrics (default: true).",
    )
    parser.add_argument("--model-dir", default=None, help="Output model directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = _resolve(args.config)
    cfg_data = read_json(cfg_path) if cfg_path and cfg_path.exists() else {}

    data_dir = _pick(args.data_dir, cfg_data, "data_dir", "RULdata/CMAPSSData")
    fd = _pick(args.fd, cfg_data, "fd", "FD001")
    max_rul = int(_pick(args.max_rul, cfg_data, "max_rul", 125))
    val_fraction = float(_pick(args.val_fraction, cfg_data, "val_fraction", 0.2))
    seed = int(_pick(args.seed, cfg_data, "seed", 42))
    seq_len = int(_pick(args.seq_len, cfg_data, "seq_len", 30))
    sample_step = int(_pick(args.sample_step, cfg_data, "sample_step", 1))
    device = _pick(args.device, cfg_data, "device", "auto")
    val_last_cycle_only = bool(_pick(args.val_last_cycle_only, cfg_data, "val_last_cycle_only", True))

    train_raw = load_split(_resolve(data_dir), fd_id=fd, split="train")
    train_with_target = add_train_rul(train_raw, max_rul=max_rul)

    units = train_with_target["unit"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])
    if not train_units:
        raise ValueError("Validation split consumed all units. Lower --val-fraction.")

    train_df = train_with_target[train_with_target["unit"].isin(train_units)].sort_values(["unit", "cycle"]).reset_index(
        drop=True
    )
    valid_df = train_with_target[train_with_target["unit"].isin(val_units)].sort_values(["unit", "cycle"]).reset_index(
        drop=True
    )

    x_train_tab = build_features(train_df)
    x_valid_tab = build_features(valid_df)
    feat_cols = feature_columns()

    x_train_seq, y_train_seq, _ = build_sequence_samples(
        x_train_tab[feat_cols],
        units=train_df["unit"].to_numpy(),
        targets=train_df["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=sample_step,
        last_only=False,
    )
    x_valid_seq, y_valid_seq, _ = build_sequence_samples(
        x_valid_tab[feat_cols],
        units=valid_df["unit"].to_numpy(),
        targets=valid_df["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=1,
        last_only=False,
    )
    x_valid_last_seq, y_valid_last_seq, _ = build_sequence_samples(
        x_valid_tab[feat_cols],
        units=valid_df["unit"].to_numpy(),
        targets=valid_df["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=1,
        last_only=True,
    )

    mean, std = fit_window_standardizer(x_train_seq)
    x_train_seq = apply_window_standardizer(x_train_seq, mean, std)
    x_valid_seq = apply_window_standardizer(x_valid_seq, mean, std)
    x_valid_last_seq = apply_window_standardizer(x_valid_last_seq, mean, std)

    model_cfg = LSTMConfig(
        input_size=x_train_seq.shape[2],
        hidden_size=int(_pick(args.hidden_size, cfg_data, "hidden_size", 96)),
        num_layers=int(_pick(args.num_layers, cfg_data, "num_layers", 2)),
        dropout=float(_pick(args.dropout, cfg_data, "dropout", 0.2)),
        learning_rate=float(_pick(args.learning_rate, cfg_data, "learning_rate", 1e-3)),
        weight_decay=float(_pick(args.weight_decay, cfg_data, "weight_decay", 1e-5)),
        epochs=int(_pick(args.epochs, cfg_data, "epochs", 12)),
        batch_size=int(_pick(args.batch_size, cfg_data, "batch_size", 256)),
        patience=int(_pick(args.patience, cfg_data, "patience", 3)),
        random_state=seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_model_dir = ROOT / "models" / f"lstm_{fd}_{ts}"
    model_dir = _resolve(args.model_dir) if args.model_dir else default_model_dir
    ensure_dir(model_dir)

    model, history, resolved_device = train_lstm_regressor(
        x_train_seq,
        y_train_seq,
        x_valid_last_seq if val_last_cycle_only else x_valid_seq,
        y_valid_last_seq if val_last_cycle_only else y_valid_seq,
        cfg=model_cfg,
        device=device,
    )
    pred_valid_all = predict_lstm(model, x_valid_seq, batch_size=model_cfg.batch_size, device=resolved_device)
    pred_valid_last = predict_lstm(model, x_valid_last_seq, batch_size=model_cfg.batch_size, device=resolved_device)
    metrics_all = {
        "rmse": rmse(y_valid_seq, pred_valid_all),
        "mae": mae(y_valid_seq, pred_valid_all),
        "phm_score": phm_score(y_valid_seq, pred_valid_all),
    }
    metrics_last = {
        "rmse": rmse(y_valid_last_seq, pred_valid_last),
        "mae": mae(y_valid_last_seq, pred_valid_last),
        "phm_score": phm_score(y_valid_last_seq, pred_valid_last),
    }
    metrics_primary = metrics_last if val_last_cycle_only else metrics_all

    save_lstm_checkpoint(model, str(model_dir / "model.pt"))
    write_json(
        model_dir / "metadata.json",
        {
            "model_type": "lstm_regressor",
            "fd": fd,
            "data_dir": str(_resolve(data_dir)),
            "max_rul": max_rul,
            "seq_len": seq_len,
            "feature_columns": feat_cols,
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_units": int(len(train_units)),
            "valid_units": int(len(val_units)),
            "train_windows": int(len(x_train_seq)),
            "valid_windows": int(len(x_valid_seq)),
            "valid_last_cycle_windows": int(len(x_valid_last_seq)),
            "metrics_valid": metrics_primary,
            "metrics_valid_last_cycle": metrics_last,
            "metrics_valid_all_cycles": metrics_all,
            "history": history,
            "params": {
                "hidden_size": model_cfg.hidden_size,
                "num_layers": model_cfg.num_layers,
                "dropout": model_cfg.dropout,
                "learning_rate": model_cfg.learning_rate,
                "weight_decay": model_cfg.weight_decay,
                "epochs": model_cfg.epochs,
                "batch_size": model_cfg.batch_size,
                "patience": model_cfg.patience,
                "seed": seed,
                "val_fraction": val_fraction,
                "sample_step": sample_step,
                "device": resolved_device,
                "val_last_cycle_only": val_last_cycle_only,
            },
        },
    )

    print(f"Model directory: {model_dir}")
    print(f"Device: {resolved_device}")
    print(f"Validation mode: {'last_cycle' if val_last_cycle_only else 'all_cycles'}")
    print(f"Validation RMSE: {metrics_primary['rmse']:.4f}")
    print(f"Validation MAE: {metrics_primary['mae']:.4f}")
    print(f"Validation PHM score: {metrics_primary['phm_score']:.4f}")


if __name__ == "__main__":
    main()
