from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import load_rul_targets, load_split, select_last_cycle_rows
from rul_pipeline.inference import predict_last_cycle
from rul_pipeline.io_utils import read_json, write_json
from rul_pipeline.metrics import mae, phm_score, rmse


def _resolve(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None
    p = Path(path_arg)
    return (ROOT / p).resolve() if not p.is_absolute() else p


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for LSTM RUL model.")
    parser.add_argument("--config", default="config/tune_lstm.json", help="JSON config path.")
    parser.add_argument("--output-dir", default="outputs/tuning", help="Output directory for ranking artifacts.")
    parser.add_argument("--models-root", default="models/tuning_lstm", help="Models root for trial checkpoints.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on trials per FD.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for trial shuffle when max-trials is used.")
    return parser.parse_args()


def build_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [_as_list(grid[k]) for k in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def run_train(fd: str, trial_cfg: dict[str, Any], model_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "train_sequence.py"),
        "--fd",
        fd,
        "--model-dir",
        str(model_dir),
    ]
    for k, v in trial_cfg.items():
        cli_name = f"--{k.replace('_', '-')}"
        cmd.extend([cli_name, str(v)])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def evaluate_test(model_dir: Path, data_dir: Path, fd: str) -> dict[str, float]:
    pred_df, _ = predict_last_cycle(model_dir=model_dir, data_dir=data_dir, fd=fd, device="auto")
    test_df = load_split(data_dir, fd_id=fd, split="test")
    test_last = select_last_cycle_rows(test_df)
    true_rul = load_rul_targets(data_dir, fd_id=fd)
    gt = pd.DataFrame({"unit": test_last["unit"].astype(int), "true_rul": true_rul["rul"].astype(float)})
    merged = gt.merge(pred_df, on="unit", how="left")
    if merged["pred_rul"].isna().any():
        raise ValueError(f"Missing predictions for FD {fd} in model {model_dir}")
    y_true = merged["true_rul"].to_numpy()
    y_pred = merged["pred_rul"].to_numpy()
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
    }


def main() -> None:
    args = parse_args()
    cfg = read_json(_resolve(args.config))

    data_dir = _resolve(cfg.get("data_dir", "RULdata/CMAPSSData"))
    fd_list = [str(x).upper() for x in _as_list(cfg.get("fds", ["FD001", "FD002", "FD003", "FD004"]))]
    grid = cfg["grid"]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = _resolve(args.models_root)
    models_root.mkdir(parents=True, exist_ok=True)

    base_trials = build_trials(grid)
    if not base_trials:
        raise ValueError("Grid produced zero trials.")

    rng = np.random.default_rng(args.seed)

    all_rows: list[dict[str, Any]] = []
    for fd in fd_list:
        fd_trials = list(base_trials)
        if args.max_trials is not None and args.max_trials < len(fd_trials):
            order = np.arange(len(fd_trials))
            rng.shuffle(order)
            fd_trials = [fd_trials[i] for i in order[: args.max_trials]]

        print(f"=== TUNING {fd}: {len(fd_trials)} trials ===")
        fd_rows: list[dict[str, Any]] = []

        for i, trial in enumerate(fd_trials, start=1):
            trial_name = f"trial_{i:03d}"
            model_dir = models_root / fd / trial_name
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{fd}] {trial_name}/{len(fd_trials)} params={trial}")
            run_train(fd=fd, trial_cfg=trial, model_dir=model_dir)

            metadata = read_json(model_dir / "metadata.json")
            metrics_valid = metadata["metrics_valid"]
            metrics_test = evaluate_test(model_dir=model_dir, data_dir=data_dir, fd=fd)

            row = {
                "fd": fd,
                "trial": trial_name,
                "model_dir": str(model_dir),
                "val_rmse": metrics_valid["rmse"],
                "val_mae": metrics_valid["mae"],
                "val_phm_score": metrics_valid["phm_score"],
                "test_rmse": metrics_test["rmse"],
                "test_mae": metrics_test["mae"],
                "test_phm_score": metrics_test["phm_score"],
                "params_json": json.dumps(trial, sort_keys=True),
            }
            fd_rows.append(row)
            all_rows.append(row)

        fd_df = pd.DataFrame(fd_rows)
        fd_df_ranked_val = fd_df.sort_values(["val_rmse", "val_phm_score", "val_mae"]).reset_index(drop=True)
        fd_df_ranked_test = fd_df.sort_values(["test_rmse", "test_phm_score", "test_mae"]).reset_index(drop=True)
        fd_csv = output_dir / f"lstm_tuning_{fd}.csv"
        fd_json = output_dir / f"lstm_tuning_{fd}.json"
        fd_df_ranked_val.to_csv(fd_csv, index=False)
        write_json(
            fd_json,
            {
                "fd": fd,
                "n_trials": int(len(fd_df_ranked_val)),
                "best_trial_by_val": fd_df_ranked_val.iloc[0].to_dict() if len(fd_df_ranked_val) else None,
                "best_trial_by_test": fd_df_ranked_test.iloc[0].to_dict() if len(fd_df_ranked_test) else None,
                "trials_ranked_by_val": fd_df_ranked_val.to_dict(orient="records"),
            },
        )
        best_val = fd_df_ranked_val.iloc[0]
        best_test = fd_df_ranked_test.iloc[0]
        print(
            f"[{fd}] best_by_val={best_val['trial']} val_rmse={best_val['val_rmse']:.4f} "
            f"test_rmse={best_val['test_rmse']:.4f} | "
            f"best_by_test={best_test['trial']} test_rmse={best_test['test_rmse']:.4f}"
        )

    all_df = pd.DataFrame(all_rows).sort_values(["fd", "val_rmse", "val_phm_score", "val_mae"]).reset_index(drop=True)
    all_csv = output_dir / "lstm_tuning_all_fd.csv"
    all_json = output_dir / "lstm_tuning_all_fd.json"
    all_df.to_csv(all_csv, index=False)
    write_json(
        all_json,
        {
            "n_rows": int(len(all_df)),
            "by_fd_best_by_val": {
                fd: all_df[all_df["fd"] == fd].sort_values(["val_rmse", "val_phm_score", "val_mae"]).iloc[0].to_dict()
                for fd in sorted(all_df["fd"].unique())
            },
            "by_fd_best_by_test": {
                fd: all_df[all_df["fd"] == fd].sort_values(["test_rmse", "test_phm_score", "test_mae"]).iloc[0].to_dict()
                for fd in sorted(all_df["fd"].unique())
            },
            "rows": all_df.to_dict(orient="records"),
        },
    )
    print(f"Saved: {all_csv}")
    print(f"Saved: {all_json}")


if __name__ == "__main__":
    main()
