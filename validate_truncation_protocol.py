from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import add_train_rul, build_truncated_validation, load_split
from rul_pipeline.inference import predict_on_dataframe
from rul_pipeline.io_utils import read_json, write_json
from rul_pipeline.metrics import mae, phm_score, rmse
from rul_pipeline.operations import parse_int_list_csv


def _resolve(path_arg: str) -> Path:
    p = Path(path_arg)
    return (ROOT / p).resolve() if not p.is_absolute() else p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check truncation-validation stability across cut seeds.")
    parser.add_argument("--model-dir", required=True, help="Model directory with metadata.")
    parser.add_argument("--fd", default=None, help="FD001..FD004 (defaults to model metadata).")
    parser.add_argument("--data-dir", default=None, help="CMAPSSData directory (defaults to model metadata).")
    parser.add_argument("--split-seed", type=int, default=None, help="Unit split seed (defaults to metadata seed).")
    parser.add_argument("--val-fraction", type=float, default=None, help="Validation fraction (defaults to metadata).")
    parser.add_argument("--max-rul", type=int, default=None, help="RUL clipping (defaults to metadata).")
    parser.add_argument("--min-prefix", type=int, default=20, help="Min prefix cycles for truncation.")
    parser.add_argument("--cut-seeds", default="11,22,33,44,55", help="Comma-separated truncation seeds.")
    parser.add_argument("--output-csv", default="outputs/truncation_stability.csv", help="Per-seed metrics CSV.")
    parser.add_argument("--output-json", default="outputs/truncation_stability.json", help="Summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = _resolve(args.model_dir)
    metadata = read_json(model_dir / "metadata.json")

    fd = (args.fd or metadata["fd"]).upper()
    data_dir = _resolve(args.data_dir) if args.data_dir else Path(metadata["data_dir"])
    params = metadata.get("params", {})
    split_seed = int(args.split_seed if args.split_seed is not None else params.get("seed", 42))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else params.get("val_fraction", 0.2))
    max_rul = int(args.max_rul if args.max_rul is not None else metadata.get("max_rul", 125))
    cut_seeds = parse_int_list_csv(args.cut_seeds)
    if not cut_seeds:
        raise ValueError("cut-seeds cannot be empty.")

    train_raw = load_split(data_dir, fd_id=fd, split="train")
    train_df = add_train_rul(train_raw, max_rul=max_rul)

    units = train_df["unit"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(split_seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    valid_df_full = train_df[train_df["unit"].isin(val_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)

    rows: list[dict] = []
    for cut_seed in cut_seeds:
        observed_df, cuts_df = build_truncated_validation(
            valid_df_full,
            min_prefix_cycles=args.min_prefix,
            random_state=int(cut_seed),
        )
        pred_df, _ = predict_on_dataframe(model_dir=model_dir, data_df=observed_df, device="auto")
        idx = pred_df.groupby("unit")["cycle"].idxmax().sort_values()
        pred_last = pred_df.loc[idx, ["unit", "pred_rul"]].reset_index(drop=True)
        merged = cuts_df[["unit", "true_rul_at_cut"]].merge(pred_last, on="unit", how="left")
        y_true = merged["true_rul_at_cut"].to_numpy(dtype=float)
        y_pred = merged["pred_rul"].to_numpy(dtype=float)

        rows.append(
            {
                "cut_seed": int(cut_seed),
                "n_units": int(len(merged)),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "phm_score": phm_score(y_true, y_pred),
            }
        )

    df = pd.DataFrame(rows).sort_values("cut_seed").reset_index(drop=True)
    out_csv = _resolve(args.output_csv)
    out_json = _resolve(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "fd": fd,
        "model_dir": str(model_dir),
        "n_seeds": int(len(df)),
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "min_prefix": int(args.min_prefix),
        "cut_seeds": cut_seeds,
        "metrics_mean": {
            "rmse": float(df["rmse"].mean()),
            "mae": float(df["mae"].mean()),
            "phm_score": float(df["phm_score"].mean()),
        },
        "metrics_std": {
            "rmse": float(df["rmse"].std(ddof=0)),
            "mae": float(df["mae"].std(ddof=0)),
            "phm_score": float(df["phm_score"].std(ddof=0)),
        },
        "rows": df.to_dict(orient="records"),
    }
    write_json(out_json, summary)

    print(f"Saved truncation stability CSV: {out_csv}")
    print(f"Saved truncation stability JSON: {out_json}")
    print(df.to_string(index=False))
    print("mean:", summary["metrics_mean"])
    print("std:", summary["metrics_std"])


if __name__ == "__main__":
    main()

