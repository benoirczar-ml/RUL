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
from rul_pipeline.inference import predict_last_cycle
from rul_pipeline.io_utils import write_json
from rul_pipeline.metrics import mae, phm_score, rmse


def _resolve(path_arg: str) -> Path:
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple RUL model directories on one FD split.")
    parser.add_argument("--model-dirs", nargs="+", required=True, help="List of model directories.")
    parser.add_argument("--fd", required=True, help="Dataset id: FD001..FD004.")
    parser.add_argument("--data-dir", default="RULdata/CMAPSSData", help="Path to CMAPSSData directory.")
    parser.add_argument("--output-csv", default="outputs/benchmark.csv", help="Benchmark CSV output path.")
    parser.add_argument("--output-json", default="outputs/benchmark.json", help="Benchmark JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    run_fd = args.fd.upper()

    test_df = load_split(data_dir, fd_id=run_fd, split="test")
    test_last = select_last_cycle_rows(test_df)
    true_rul = load_rul_targets(data_dir, fd_id=run_fd)
    gt = pd.DataFrame({"unit": test_last["unit"].astype(int), "true_rul": true_rul["rul"].astype(float)})

    rows: list[dict] = []
    for model_dir_raw in args.model_dirs:
        model_dir = _resolve(model_dir_raw)
        pred_df, metadata = predict_last_cycle(model_dir=model_dir, data_dir=data_dir, fd=run_fd, device="auto")
        merged = gt.merge(pred_df[["unit", "pred_rul"]], on="unit", how="left")
        if merged["pred_rul"].isna().any():
            raise ValueError(f"Missing predictions for some units in model: {model_dir}")

        y_true = merged["true_rul"].to_numpy()
        y_pred = merged["pred_rul"].to_numpy()
        rows.append(
            {
                "model_dir": str(model_dir),
                "model_name": model_dir.name,
                "model_type": metadata["model_type"],
                "fd": run_fd,
                "n_units": int(len(merged)),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "phm_score": phm_score(y_true, y_pred),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["rmse", "phm_score", "mae"]).reset_index(drop=True)

    output_csv = _resolve(args.output_csv)
    output_json = _resolve(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv, index=False)
    write_json(
        output_json,
        {
            "fd": run_fd,
            "models": result_df.to_dict(orient="records"),
            "best_by_rmse": result_df.iloc[0].to_dict() if len(result_df) else None,
        },
    )

    print(f"Saved benchmark CSV: {output_csv}")
    print(f"Saved benchmark JSON: {output_json}")
    print(result_df[["model_name", "model_type", "rmse", "mae", "phm_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

