from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.inference import predict_last_cycle


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
    output_path = _resolve(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df, metadata = predict_last_cycle(
        model_dir=model_dir,
        data_dir=_resolve(args.data_dir) if args.data_dir else None,
        fd=args.fd,
        device="auto",
    )
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    print(f"Rows: {len(pred_df)}")
    print(f"Model type: {metadata['model_type']}")


if __name__ == "__main__":
    main()
