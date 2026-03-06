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

from rul_pipeline.data import add_train_rul, build_truncated_validation, load_split
from rul_pipeline.features import build_features, feature_columns
from rul_pipeline.hybrid_sequence_model import (
    ConvAttentionLSTMConfig,
    predict_conv_attention_lstm,
    save_conv_attention_lstm_checkpoint,
    train_conv_attention_lstm_regressor,
)
from rul_pipeline.io_utils import ensure_dir, read_json, write_json
from rul_pipeline.metrics import mae, phm_score, rmse
from rul_pipeline.sequence import apply_window_standardizer, build_sequence_samples, fit_window_standardizer

FD_INDEX_MAP = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}


def _resolve(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def _pick(cli_value, config_dict: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_dict.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C-MAPSS Conv+Attention+LSTM sequence regressor.")
    parser.add_argument("--config", default="config/train_hybrid.json", help="JSON config path.")
    parser.add_argument("--data-dir", default=None, help="Path to CMAPSSData directory.")
    parser.add_argument("--fd", default=None, help="Dataset id: FD001..FD004.")
    parser.add_argument("--max-rul", type=int, default=None, help="Clip train RUL target.")
    parser.add_argument("--val-fraction", type=float, default=None, help="Validation unit split fraction.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--seq-len", type=int, default=None, help="Window length.")
    parser.add_argument("--sample-step", type=int, default=None, help="Training window stride.")
    parser.add_argument("--conv-channels", type=int, default=None, help="Temporal Conv1D output channels.")
    parser.add_argument("--kernel-size", type=int, default=None, help="Temporal Conv1D kernel size.")
    parser.add_argument("--attention-heads", type=int, default=None, help="Multi-head attention heads.")
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=None, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout.")
    parser.add_argument("--num-fd-heads", type=int, default=None, help="Number of FD-specific heads.")
    parser.add_argument("--loss-name", choices=["mse", "huber_asymmetric"], default=None, help="Training loss.")
    parser.add_argument("--huber-delta", type=float, default=None, help="Huber delta for asymmetric loss.")
    parser.add_argument("--late-error-weight", type=float, default=None, help="Extra weight for late errors.")
    parser.add_argument("--late-error-margin", type=float, default=None, help="Late error threshold on (pred-true).")
    parser.add_argument(
        "--emphasize-failure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use weighted sampling to emphasize low-RUL windows.",
    )
    parser.add_argument("--failure-rul-threshold", type=int, default=None, help="Low-RUL threshold for sampling weight.")
    parser.add_argument("--failure-weight", type=float, default=None, help="Sample weight multiplier for low-RUL windows.")
    parser.add_argument(
        "--sampling-strategy",
        choices=["auto", "shuffle", "failure_weighted", "balanced_fd_failure"],
        default=None,
        help="Batch sampling strategy.",
    )
    parser.add_argument("--fd-balance-power", type=float, default=None, help="Power for inverse-FD-frequency weighting.")
    parser.add_argument("--warmup-mse-epochs", type=int, default=None, help="Use MSE for first N epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--device", default=None, help="auto/cpu/cuda.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned-memory DataLoader buffers on CUDA.",
    )
    parser.add_argument(
        "--non-blocking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable non-blocking CPU->GPU copies.",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--enable-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TF32 matmul on CUDA.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable cuDNN benchmark autotuning.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile acceleration.",
    )
    parser.add_argument("--compile-mode", default=None, help="torch.compile mode, e.g. reduce-overhead/max-autotune.")
    parser.add_argument("--compile-backend", default=None, help="torch.compile backend, e.g. inductor.")
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable fullgraph mode in torch.compile.",
    )
    parser.add_argument(
        "--log-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print one metrics line per epoch.",
    )
    parser.add_argument("--val-strategy", choices=["truncation", "last_cycle"], default=None, help="Validation strategy.")
    parser.add_argument("--val-min-prefix", type=int, default=None, help="Min observed cycles in truncation validation.")
    parser.add_argument("--model-dir", default=None, help="Output model directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = _resolve(args.config)
    cfg_data = read_json(cfg_path) if cfg_path and cfg_path.exists() else {}

    data_dir = _pick(args.data_dir, cfg_data, "data_dir", "RULdata/CMAPSSData")
    fd = str(_pick(args.fd, cfg_data, "fd", "FD001")).upper()
    if fd not in FD_INDEX_MAP:
        raise ValueError(f"Unsupported fd={fd}. Expected one of {sorted(FD_INDEX_MAP)}.")
    fd_index = FD_INDEX_MAP[fd]
    max_rul = int(_pick(args.max_rul, cfg_data, "max_rul", 125))
    val_fraction = float(_pick(args.val_fraction, cfg_data, "val_fraction", 0.2))
    seed = int(_pick(args.seed, cfg_data, "seed", 42))
    seq_len = int(_pick(args.seq_len, cfg_data, "seq_len", 30))
    sample_step = int(_pick(args.sample_step, cfg_data, "sample_step", 1))
    device = _pick(args.device, cfg_data, "device", "auto")
    val_strategy = str(_pick(args.val_strategy, cfg_data, "val_strategy", "truncation"))
    val_min_prefix = int(_pick(args.val_min_prefix, cfg_data, "val_min_prefix", 20))

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
    valid_df_full = train_with_target[train_with_target["unit"].isin(val_units)].sort_values(["unit", "cycle"]).reset_index(
        drop=True
    )

    if val_strategy == "truncation":
        valid_df_eval, valid_cuts = build_truncated_validation(
            valid_df_full,
            min_prefix_cycles=val_min_prefix,
            random_state=seed,
        )
    elif val_strategy == "last_cycle":
        valid_df_eval = valid_df_full.copy()
        valid_cuts = None
    else:
        raise ValueError(f"Unsupported val_strategy={val_strategy}")

    x_train_tab = build_features(train_df)
    x_valid_eval_tab = build_features(valid_df_eval)
    x_valid_full_tab = build_features(valid_df_full)
    feat_cols = feature_columns()

    x_train_seq, y_train_seq, _ = build_sequence_samples(
        x_train_tab[feat_cols],
        units=train_df["unit"].to_numpy(),
        targets=train_df["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=sample_step,
        last_only=False,
    )
    x_valid_eval_seq, y_valid_eval_seq, _ = build_sequence_samples(
        x_valid_eval_tab[feat_cols],
        units=valid_df_eval["unit"].to_numpy(),
        targets=valid_df_eval["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=1,
        last_only=False,
    )
    x_valid_eval_last_seq, y_valid_eval_last_seq, _ = build_sequence_samples(
        x_valid_eval_tab[feat_cols],
        units=valid_df_eval["unit"].to_numpy(),
        targets=valid_df_eval["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=1,
        last_only=True,
    )
    x_valid_full_last_seq, y_valid_full_last_seq, _ = build_sequence_samples(
        x_valid_full_tab[feat_cols],
        units=valid_df_full["unit"].to_numpy(),
        targets=valid_df_full["rul"].to_numpy(),
        seq_len=seq_len,
        sample_step=1,
        last_only=True,
    )

    mean, std = fit_window_standardizer(x_train_seq)
    x_train_seq = apply_window_standardizer(x_train_seq, mean, std)
    x_valid_eval_seq = apply_window_standardizer(x_valid_eval_seq, mean, std)
    x_valid_eval_last_seq = apply_window_standardizer(x_valid_eval_last_seq, mean, std)
    x_valid_full_last_seq = apply_window_standardizer(x_valid_full_last_seq, mean, std)

    fd_train_idx = np.full(len(x_train_seq), fd_index, dtype=np.int64)
    fd_valid_eval_last_idx = np.full(len(x_valid_eval_last_seq), fd_index, dtype=np.int64)
    fd_valid_full_last_idx = np.full(len(x_valid_full_last_seq), fd_index, dtype=np.int64)

    model_cfg = ConvAttentionLSTMConfig(
        input_size=x_train_seq.shape[2],
        conv_channels=int(_pick(args.conv_channels, cfg_data, "conv_channels", 96)),
        kernel_size=int(_pick(args.kernel_size, cfg_data, "kernel_size", 5)),
        attention_heads=int(_pick(args.attention_heads, cfg_data, "attention_heads", 4)),
        hidden_size=int(_pick(args.hidden_size, cfg_data, "hidden_size", 96)),
        num_layers=int(_pick(args.num_layers, cfg_data, "num_layers", 2)),
        dropout=float(_pick(args.dropout, cfg_data, "dropout", 0.2)),
        num_fd_heads=int(_pick(args.num_fd_heads, cfg_data, "num_fd_heads", 4)),
        loss_name=str(_pick(args.loss_name, cfg_data, "loss_name", "huber_asymmetric")),
        huber_delta=float(_pick(args.huber_delta, cfg_data, "huber_delta", 10.0)),
        late_error_weight=float(_pick(args.late_error_weight, cfg_data, "late_error_weight", 0.35)),
        late_error_margin=float(_pick(args.late_error_margin, cfg_data, "late_error_margin", 0.0)),
        emphasize_failure=bool(_pick(args.emphasize_failure, cfg_data, "emphasize_failure", True)),
        failure_rul_threshold=int(_pick(args.failure_rul_threshold, cfg_data, "failure_rul_threshold", 30)),
        failure_weight=float(_pick(args.failure_weight, cfg_data, "failure_weight", 2.0)),
        sampling_strategy=str(_pick(args.sampling_strategy, cfg_data, "sampling_strategy", "auto")),
        fd_balance_power=float(_pick(args.fd_balance_power, cfg_data, "fd_balance_power", 1.0)),
        warmup_mse_epochs=int(_pick(args.warmup_mse_epochs, cfg_data, "warmup_mse_epochs", 0)),
        learning_rate=float(_pick(args.learning_rate, cfg_data, "learning_rate", 1e-3)),
        weight_decay=float(_pick(args.weight_decay, cfg_data, "weight_decay", 1e-5)),
        epochs=int(_pick(args.epochs, cfg_data, "epochs", 12)),
        batch_size=int(_pick(args.batch_size, cfg_data, "batch_size", 256)),
        patience=int(_pick(args.patience, cfg_data, "patience", 3)),
        random_state=seed,
        num_workers=int(_pick(args.num_workers, cfg_data, "num_workers", 2)),
        pin_memory=bool(_pick(args.pin_memory, cfg_data, "pin_memory", True)),
        non_blocking=bool(_pick(args.non_blocking, cfg_data, "non_blocking", True)),
        use_amp=bool(_pick(args.use_amp, cfg_data, "use_amp", True)),
        enable_tf32=bool(_pick(args.enable_tf32, cfg_data, "enable_tf32", True)),
        cudnn_benchmark=bool(_pick(args.cudnn_benchmark, cfg_data, "cudnn_benchmark", True)),
        use_torch_compile=bool(_pick(args.use_torch_compile, cfg_data, "use_torch_compile", False)),
        compile_mode=str(_pick(args.compile_mode, cfg_data, "compile_mode", "reduce-overhead")),
        compile_backend=str(_pick(args.compile_backend, cfg_data, "compile_backend", "inductor")),
        compile_fullgraph=bool(_pick(args.compile_fullgraph, cfg_data, "compile_fullgraph", False)),
        log_every_epoch=bool(_pick(args.log_every_epoch, cfg_data, "log_every_epoch", True)),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_model_dir = ROOT / "models" / f"caelstm_{fd}_{ts}"
    model_dir = _resolve(args.model_dir) if args.model_dir else default_model_dir
    ensure_dir(model_dir)

    model, history, resolved_device = train_conv_attention_lstm_regressor(
        x_train_seq,
        y_train_seq,
        x_valid_eval_last_seq,
        y_valid_eval_last_seq,
        cfg=model_cfg,
        device=device,
        fd_train=fd_train_idx,
        fd_valid=fd_valid_eval_last_idx,
    )
    pred_valid_eval = predict_conv_attention_lstm(
        model,
        x_valid_eval_last_seq,
        batch_size=model_cfg.batch_size,
        device=resolved_device,
        non_blocking=model_cfg.non_blocking,
        pin_memory=model_cfg.pin_memory,
        fd_idx=fd_valid_eval_last_idx,
    )
    pred_valid_full_last = predict_conv_attention_lstm(
        model,
        x_valid_full_last_seq,
        batch_size=model_cfg.batch_size,
        device=resolved_device,
        non_blocking=model_cfg.non_blocking,
        pin_memory=model_cfg.pin_memory,
        fd_idx=fd_valid_full_last_idx,
    )
    metrics_primary = {
        "rmse": rmse(y_valid_eval_last_seq, pred_valid_eval),
        "mae": mae(y_valid_eval_last_seq, pred_valid_eval),
        "phm_score": phm_score(y_valid_eval_last_seq, pred_valid_eval),
    }
    metrics_full_last = {
        "rmse": rmse(y_valid_full_last_seq, pred_valid_full_last),
        "mae": mae(y_valid_full_last_seq, pred_valid_full_last),
        "phm_score": phm_score(y_valid_full_last_seq, pred_valid_full_last),
    }

    save_conv_attention_lstm_checkpoint(model, str(model_dir / "model.pt"))
    write_json(
        model_dir / "metadata.json",
        {
            "model_type": "conv_attn_lstm_regressor",
            "fd": fd,
            "data_dir": str(_resolve(data_dir)),
            "max_rul": max_rul,
            "seq_len": seq_len,
            "feature_columns": feat_cols,
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df_full)),
            "valid_eval_rows": int(len(valid_df_eval)),
            "train_units": int(len(train_units)),
            "valid_units": int(len(val_units)),
            "train_windows": int(len(x_train_seq)),
            "valid_eval_windows": int(len(x_valid_eval_seq)),
            "valid_eval_last_cycle_windows": int(len(x_valid_eval_last_seq)),
            "metrics_valid": metrics_primary,
            "metrics_valid_full_last_cycle": metrics_full_last,
            "history": history,
            "params": {
                "conv_channels": model_cfg.conv_channels,
                "kernel_size": model_cfg.kernel_size,
                "attention_heads": model_cfg.attention_heads,
                "hidden_size": model_cfg.hidden_size,
                "num_layers": model_cfg.num_layers,
                "dropout": model_cfg.dropout,
                "num_fd_heads": model_cfg.num_fd_heads,
                "loss_name": model_cfg.loss_name,
                "huber_delta": model_cfg.huber_delta,
                "late_error_weight": model_cfg.late_error_weight,
                "late_error_margin": model_cfg.late_error_margin,
                "emphasize_failure": model_cfg.emphasize_failure,
                "failure_rul_threshold": model_cfg.failure_rul_threshold,
                "failure_weight": model_cfg.failure_weight,
                "sampling_strategy": model_cfg.sampling_strategy,
                "fd_balance_power": model_cfg.fd_balance_power,
                "warmup_mse_epochs": model_cfg.warmup_mse_epochs,
                "learning_rate": model_cfg.learning_rate,
                "weight_decay": model_cfg.weight_decay,
                "epochs": model_cfg.epochs,
                "batch_size": model_cfg.batch_size,
                "patience": model_cfg.patience,
                "num_workers": model_cfg.num_workers,
                "pin_memory": model_cfg.pin_memory,
                "non_blocking": model_cfg.non_blocking,
                "use_amp": model_cfg.use_amp,
                "enable_tf32": model_cfg.enable_tf32,
                "cudnn_benchmark": model_cfg.cudnn_benchmark,
                "use_torch_compile": model_cfg.use_torch_compile,
                "compile_mode": model_cfg.compile_mode,
                "compile_backend": model_cfg.compile_backend,
                "compile_fullgraph": model_cfg.compile_fullgraph,
                "log_every_epoch": model_cfg.log_every_epoch,
                "seed": seed,
                "val_fraction": val_fraction,
                "sample_step": sample_step,
                "device": resolved_device,
                "val_strategy": val_strategy,
                "val_min_prefix": val_min_prefix,
                "fd_index": fd_index,
                "fd_index_map": FD_INDEX_MAP,
            },
            "validation_cuts": valid_cuts.to_dict(orient="records") if valid_cuts is not None else [],
        },
    )

    print(f"Model directory: {model_dir}")
    print(f"Device: {resolved_device}")
    print(f"Validation strategy: {val_strategy}")
    print(f"Validation RMSE: {metrics_primary['rmse']:.4f}")
    print(f"Validation MAE: {metrics_primary['mae']:.4f}")
    print(f"Validation PHM score: {metrics_primary['phm_score']:.4f}")


if __name__ == "__main__":
    main()
