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


def _parse_fds(raw: Any) -> list[str]:
    if raw is None:
        fds = ["FD001", "FD002", "FD003", "FD004"]
    elif isinstance(raw, str):
        fds = [x.strip().upper() for x in raw.split(",") if x.strip()]
    else:
        fds = [str(x).strip().upper() for x in raw if str(x).strip()]
    valid = {"FD001", "FD002", "FD003", "FD004"}
    for fd in fds:
        if fd not in valid:
            raise ValueError(f"Unsupported FD={fd}. Allowed: {sorted(valid)}")
    if not fds:
        raise ValueError("fds cannot be empty.")
    out: list[str] = []
    seen: set[str] = set()
    for fd in fds:
        if fd not in seen:
            out.append(fd)
            seen.add(fd)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for multi-FD Conv+Attention+LSTM model.")
    parser.add_argument("--config", default="config/tune_hybrid_multifd.json", help="JSON config path.")
    parser.add_argument("--output-dir", default="outputs/tuning_hybrid_multifd", help="Output directory for ranking artifacts.")
    parser.add_argument("--models-root", default="models/tuning_hybrid_multifd", help="Models root for trial checkpoints.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on number of trials.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for trial shuffle when max-trials is used.")
    return parser.parse_args()


def build_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [_as_list(grid[k]) for k in keys]
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def run_train(fds: list[str], trial_cfg: dict[str, Any], model_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "train_hybrid_multifd.py"),
        "--fds",
        ",".join(fds),
        "--model-dir",
        str(model_dir),
    ]
    for k, v in trial_cfg.items():
        key = k.replace("_", "-")
        if isinstance(v, bool):
            cli_name = f"--{key}" if v else f"--no-{key}"
            cmd.append(cli_name)
        else:
            cli_name = f"--{key}"
            cmd.extend([cli_name, str(v)])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def evaluate_test_for_fd(model_dir: Path, data_dir: Path, fd: str) -> dict[str, float]:
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
    fds = _parse_fds(cfg.get("fds", ["FD001", "FD002", "FD003", "FD004"]))
    grid = cfg["grid"]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_root = _resolve(args.models_root)
    models_root.mkdir(parents=True, exist_ok=True)

    trials = build_trials(grid)
    if not trials:
        raise ValueError("Grid produced zero trials.")

    rng = np.random.default_rng(args.seed)
    if args.max_trials is not None and args.max_trials < len(trials):
        order = np.arange(len(trials))
        rng.shuffle(order)
        trials = [trials[i] for i in order[: args.max_trials]]

    rows: list[dict[str, Any]] = []
    print(f"=== MULTI-FD HYBRID TUNING: {len(trials)} trials | FDs={','.join(fds)} ===")
    for i, trial in enumerate(trials, start=1):
        trial_name = f"trial_{i:03d}"
        model_dir = models_root / trial_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[multi] {trial_name}/{len(trials)} params={trial}")

        run_train(fds=fds, trial_cfg=trial, model_dir=model_dir)
        metadata = read_json(model_dir / "metadata.json")
        val_by_fd = metadata.get("metrics_valid_by_fd", {})
        test_by_fd = {fd: evaluate_test_for_fd(model_dir=model_dir, data_dir=data_dir, fd=fd) for fd in fds}

        row: dict[str, Any] = {
            "trial": trial_name,
            "model_dir": str(model_dir),
            "params_json": json.dumps(trial, sort_keys=True),
        }

        val_rmse_list: list[float] = []
        val_mae_list: list[float] = []
        val_phm_list: list[float] = []
        test_rmse_list: list[float] = []
        test_mae_list: list[float] = []
        test_phm_list: list[float] = []

        for fd in fds:
            mv = val_by_fd.get(fd, {})
            mt = test_by_fd[fd]
            row[f"val_rmse_{fd}"] = mv.get("rmse")
            row[f"val_mae_{fd}"] = mv.get("mae")
            row[f"val_phm_{fd}"] = mv.get("phm_score")
            row[f"test_rmse_{fd}"] = mt["rmse"]
            row[f"test_mae_{fd}"] = mt["mae"]
            row[f"test_phm_{fd}"] = mt["phm_score"]

            if mv.get("rmse") is not None:
                val_rmse_list.append(float(mv["rmse"]))
            if mv.get("mae") is not None:
                val_mae_list.append(float(mv["mae"]))
            if mv.get("phm_score") is not None:
                val_phm_list.append(float(mv["phm_score"]))
            test_rmse_list.append(float(mt["rmse"]))
            test_mae_list.append(float(mt["mae"]))
            test_phm_list.append(float(mt["phm_score"]))

        row["val_rmse_macro"] = float(np.mean(val_rmse_list)) if val_rmse_list else float("nan")
        row["val_mae_macro"] = float(np.mean(val_mae_list)) if val_mae_list else float("nan")
        row["val_phm_macro"] = float(np.mean(val_phm_list)) if val_phm_list else float("nan")
        row["test_rmse_macro"] = float(np.mean(test_rmse_list))
        row["test_mae_macro"] = float(np.mean(test_mae_list))
        row["test_phm_macro"] = float(np.mean(test_phm_list))
        row["test_rmse_worst_fd"] = float(np.max(test_rmse_list))
        row["test_phm_worst_fd"] = float(np.max(test_phm_list))
        rows.append(row)

        print(
            f"[multi] {trial_name} "
            f"val_rmse_macro={row['val_rmse_macro']:.4f} "
            f"test_rmse_macro={row['test_rmse_macro']:.4f} "
            f"test_rmse_worst_fd={row['test_rmse_worst_fd']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values(
        ["test_rmse_macro", "test_rmse_worst_fd", "test_phm_macro", "val_rmse_macro"]
    ).reset_index(drop=True)

    out_csv = output_dir / "hybrid_multifd_tuning.csv"
    out_json = output_dir / "hybrid_multifd_tuning.json"
    df.to_csv(out_csv, index=False)
    write_json(
        out_json,
        {
            "n_trials": int(len(df)),
            "fds": fds,
            "best_by_test_macro_rmse": df.iloc[0].to_dict() if len(df) else None,
            "rows": df.to_dict(orient="records"),
        },
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
