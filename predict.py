from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.data import load_split
from rul_pipeline.features import build_features
from rul_pipeline.io_utils import read_json
from rul_pipeline.modeling import load_model, predict


def _resolve(path_arg: str) -> Path:
    return (ROOT / path_arg).resolve() if not Path(path_arg).is_absolute() else Path(path_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict C-MAPSS test RUL values.")
    parser.add_argument("--model-dir", required=True, help="Directory with model.joblib and metadata.json.")
    parser.add_argument("--data-dir", default=None, help="Path to CMAPSSData directory.")
    parser.add_argument("--fd", default=None, help="Dataset id: FD001..FD004 (defaults to metadata fd).")
    parser.add_argument("--output-csv", default="outputs/predictions.csv", help="Output predictions CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = _resolve(args.model_dir)
    metadata = read_json(model_dir / "metadata.json")
    model = load_model(str(model_dir / "model.joblib"))

    fd = args.fd or metadata["fd"]
    data_dir = _resolve(args.data_dir) if args.data_dir else Path(metadata["data_dir"])
    output_path = _resolve(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_raw = load_split(data_dir, fd_id=fd, split="test")
    X_all = build_features(test_raw)

    last_idx = test_raw.groupby("unit")["cycle"].idxmax().sort_values()
    units = test_raw.loc[last_idx, "unit"].to_numpy()
    feature_cols = metadata["feature_columns"]
    X_last = X_all.loc[last_idx, feature_cols]

    y_pred = predict(model, X_last)
    out = pd.DataFrame({"unit": units.astype(int), "pred_rul": y_pred.astype(float)})
    out.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
