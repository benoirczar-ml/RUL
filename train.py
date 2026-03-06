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
from rul_pipeline.modeling import HistGBRConfig, predict, save_model, train_hist_gbr


def _resolve(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def _pick(cli_value, config_dict: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_dict.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C-MAPSS baseline RUL regressor.")
    parser.add_argument("--config", default="config/train_baseline.json", help="JSON config path.")
    parser.add_argument("--data-dir", default=None, help="Path to CMAPSSData directory.")
    parser.add_argument("--fd", default=None, help="Dataset id: FD001..FD004.")
    parser.add_argument("--max-rul", type=int, default=None, help="Clip train RUL target.")
    parser.add_argument("--val-fraction", type=float, default=None, help="Validation unit split fraction.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=None, help="HistGBR max_iter.")
    parser.add_argument("--learning-rate", type=float, default=None, help="HistGBR learning_rate.")
    parser.add_argument("--max-depth", type=int, default=None, help="HistGBR max_depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=None, help="HistGBR min_samples_leaf.")
    parser.add_argument("--l2-regularization", type=float, default=None, help="HistGBR l2_regularization.")
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

    model_cfg = HistGBRConfig(
        max_iter=int(_pick(args.max_iter, cfg_data, "max_iter", 400)),
        learning_rate=float(_pick(args.learning_rate, cfg_data, "learning_rate", 0.05)),
        max_depth=int(_pick(args.max_depth, cfg_data, "max_depth", 8)),
        min_samples_leaf=int(_pick(args.min_samples_leaf, cfg_data, "min_samples_leaf", 30)),
        l2_regularization=float(_pick(args.l2_regularization, cfg_data, "l2_regularization", 0.0)),
        random_state=seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_model_dir = ROOT / "models" / f"hist_gbr_{fd}_{ts}"
    model_dir = _resolve(args.model_dir) if args.model_dir else default_model_dir
    ensure_dir(model_dir)

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

    train_df = train_with_target[train_with_target["unit"].isin(train_units)]
    valid_df = train_with_target[train_with_target["unit"].isin(val_units)]

    X_train = build_features(train_df)
    y_train = train_df["rul"].astype("float32")
    X_valid = build_features(valid_df)
    y_valid = valid_df["rul"].astype("float32")

    model = train_hist_gbr(X_train, y_train, model_cfg)
    pred_valid = predict(model, X_valid)

    metrics = {
        "rmse": rmse(y_valid.to_numpy(), pred_valid),
        "mae": mae(y_valid.to_numpy(), pred_valid),
        "phm_score": phm_score(y_valid.to_numpy(), pred_valid),
    }

    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    save_model(model, str(model_path))
    write_json(
        meta_path,
        {
            "model_type": "hist_gbr",
            "fd": fd,
            "data_dir": str(_resolve(data_dir)),
            "max_rul": max_rul,
            "feature_columns": feature_columns(),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_units": int(len(train_units)),
            "valid_units": int(len(val_units)),
            "metrics_valid": metrics,
            "params": {
                "max_iter": model_cfg.max_iter,
                "learning_rate": model_cfg.learning_rate,
                "max_depth": model_cfg.max_depth,
                "min_samples_leaf": model_cfg.min_samples_leaf,
                "l2_regularization": model_cfg.l2_regularization,
                "seed": seed,
                "val_fraction": val_fraction,
            },
        },
    )

    print(f"Model directory: {model_dir}")
    print(f"Validation RMSE: {metrics['rmse']:.4f}")
    print(f"Validation MAE: {metrics['mae']:.4f}")
    print(f"Validation PHM score: {metrics['phm_score']:.4f}")


if __name__ == "__main__":
    main()
