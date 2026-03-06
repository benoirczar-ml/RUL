from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import load_rul_targets, load_split, select_last_cycle_rows
from rul_pipeline.io_utils import write_json
from rul_pipeline.metrics import mae, phm_score, rmse


def _resolve(path_arg: str) -> Path:
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C-MAPSS test predictions.")
    parser.add_argument("--predictions-csv", required=True, help="CSV with columns: unit,pred_rul.")
    parser.add_argument("--data-dir", default="RULdata/CMAPSSData", help="Path to CMAPSSData directory.")
    parser.add_argument("--fd", required=True, help="Dataset id: FD001..FD004.")
    parser.add_argument("--output-json", default="outputs/metrics.json", help="Output metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = _resolve(args.predictions_csv)
    data_dir = _resolve(args.data_dir)
    out_json = _resolve(args.output_json)

    pred_df = pd.read_csv(pred_path)
    required_cols = {"unit", "pred_rul"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"Predictions must include columns {required_cols}, got {list(pred_df.columns)}")

    test_df = load_split(data_dir, fd_id=args.fd, split="test")
    test_last = select_last_cycle_rows(test_df)
    true_rul = load_rul_targets(data_dir, fd_id=args.fd)

    if len(test_last) != len(true_rul):
        raise ValueError(
            f"Length mismatch: test_last={len(test_last)} and true_rul={len(true_rul)}. "
            "Check dataset consistency."
        )

    gt = pd.DataFrame({"unit": test_last["unit"].astype(int), "true_rul": true_rul["rul"].astype(float)})
    merged = gt.merge(pred_df[["unit", "pred_rul"]], on="unit", how="left")

    missing = int(merged["pred_rul"].isna().sum())
    if missing > 0:
        raise ValueError(f"Missing predictions for {missing} units.")

    y_true = merged["true_rul"].to_numpy()
    y_pred = merged["pred_rul"].to_numpy()

    result = {
        "fd": args.fd.upper(),
        "n_units": int(len(merged)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
    }
    write_json(out_json, result)
    print(f"Saved metrics: {out_json}")
    print(result)


if __name__ == "__main__":
    main()
