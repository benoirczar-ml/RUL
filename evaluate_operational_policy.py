from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rul_pipeline.inference import predict_all_cycles
from rul_pipeline.io_utils import write_json
from rul_pipeline.operations import (
    evaluate_alert_policy,
    iter_policy_grid,
    parse_float_list_csv,
    parse_int_list_csv,
)


def _resolve(path_arg: str) -> Path:
    p = Path(path_arg)
    return (ROOT / p).resolve() if not p.is_absolute() else p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate operational alert policy on cycle-level RUL predictions.")
    parser.add_argument("--model-dir", required=True, help="Model directory.")
    parser.add_argument("--fd", required=True, help="FD001..FD004.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--data-dir", default="RULdata/CMAPSSData", help="CMAPSSData directory.")
    parser.add_argument("--trigger-ruls", default="30", help="Comma list, e.g. 20,30,40.")
    parser.add_argument(
        "--exit-ruls",
        default="",
        help="Comma list for hysteresis exit threshold, e.g. 120,140. Empty disables hysteresis.",
    )
    parser.add_argument("--consecutives", default="1", help="Comma list, e.g. 1,2.")
    parser.add_argument("--cooldowns", default="0", help="Comma list in cycles, e.g. 0,5,10.")
    parser.add_argument("--trend-windows", default="0", help="Comma list in cycles, e.g. 0,3,5.")
    parser.add_argument("--trend-deltas", default="0", help="Comma list in RUL units, e.g. 0,5,10.")
    parser.add_argument("--min-lead", type=int, default=5, help="Min lead time cycles for true detection.")
    parser.add_argument("--max-lead", type=int, default=130, help="Max lead time cycles for true detection.")
    parser.add_argument("--output-csv", default="outputs/operational_policy_grid.csv", help="Grid result CSV.")
    parser.add_argument("--output-json", default="outputs/operational_policy_grid.json", help="Grid result JSON.")
    parser.add_argument("--per-unit-csv", default="outputs/operational_policy_per_unit.csv", help="Per-unit CSV for best policy.")
    parser.add_argument("--alerts-csv", default="outputs/operational_policy_alerts.csv", help="Alerts CSV for best policy.")
    return parser.parse_args()


def parse_optional_float_list_csv(csv_text: str) -> list[float | None]:
    raw = csv_text.strip()
    if raw == "":
        return [None]
    out: list[float | None] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"none", "null", "na"}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def is_better_policy(cur: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if cur["recall"] != best["recall"]:
        return cur["recall"] > best["recall"]
    if cur["false_alerts"] != best["false_alerts"]:
        return cur["false_alerts"] < best["false_alerts"]
    cur_med = cur["median_lead_time_cycles"]
    best_med = best["median_lead_time_cycles"]
    if math.isnan(cur_med) and not math.isnan(best_med):
        return False
    if not math.isnan(cur_med) and math.isnan(best_med):
        return True
    if cur_med != best_med:
        return cur_med > best_med
    return cur["mean_lead_time_cycles"] > best["mean_lead_time_cycles"]


def main() -> None:
    args = parse_args()
    model_dir = _resolve(args.model_dir)
    data_dir = _resolve(args.data_dir)
    out_csv = _resolve(args.output_csv)
    out_json = _resolve(args.output_json)
    per_unit_csv = _resolve(args.per_unit_csv)
    alerts_csv = _resolve(args.alerts_csv)

    pred_df, metadata = predict_all_cycles(
        model_dir=model_dir,
        data_dir=data_dir,
        fd=args.fd,
        split=args.split,
        device="auto",
    )

    trigger_ruls = parse_float_list_csv(args.trigger_ruls)
    exit_ruls = parse_optional_float_list_csv(args.exit_ruls)
    consecutives = parse_int_list_csv(args.consecutives)
    cooldowns = parse_int_list_csv(args.cooldowns)
    trend_windows = parse_int_list_csv(args.trend_windows)
    trend_deltas = parse_float_list_csv(args.trend_deltas)
    if not trigger_ruls or not consecutives or not cooldowns or not exit_ruls or not trend_windows or not trend_deltas:
        raise ValueError("Policy grids must be non-empty.")

    rows: list[dict] = []
    best_summary = None
    best_per_unit = None
    best_alerts = None
    for trigger_rul, exit_rul, consecutive, cooldown, trend_window, trend_delta in iter_policy_grid(
        trigger_ruls,
        consecutives,
        cooldowns,
        exit_ruls=exit_ruls,
        trend_windows=trend_windows,
        trend_deltas=trend_deltas,
    ):
        if exit_rul is not None and exit_rul < trigger_rul:
            continue
        if trend_window == 0 and trend_delta > 0:
            continue
        summary, per_unit_df, alerts_df = evaluate_alert_policy(
            pred_df=pred_df,
            trigger_rul=trigger_rul,
            exit_rul=exit_rul,
            consecutive=consecutive,
            cooldown_cycles=cooldown,
            trend_window=trend_window,
            trend_delta=trend_delta,
            min_lead=args.min_lead,
            max_lead=args.max_lead,
        )
        row = {
            "trigger_rul": trigger_rul,
            "exit_rul": exit_rul,
            "consecutive": consecutive,
            "cooldown_cycles": cooldown,
            "trend_window": trend_window,
            "trend_delta": trend_delta,
            "recall": summary["recall"],
            "missed_units": summary["missed_units"],
            "false_alerts": summary["false_alerts"],
            "false_alert_rate": summary["false_alert_rate"],
            "median_lead_time_cycles": summary["median_lead_time_cycles"],
            "mean_lead_time_cycles": summary["mean_lead_time_cycles"],
            "total_alerts": summary["total_alerts"],
        }
        rows.append(row)

        if is_better_policy(row, None if best_summary is None else {
            "recall": best_summary["recall"],
            "false_alerts": best_summary["false_alerts"],
            "median_lead_time_cycles": best_summary["median_lead_time_cycles"],
            "mean_lead_time_cycles": best_summary["mean_lead_time_cycles"],
        }):
            best_summary = summary
            best_per_unit = per_unit_df
            best_alerts = alerts_df
    if not rows:
        raise ValueError("No valid policy combinations after constraints.")

    result_df = pd.DataFrame(rows).sort_values(
        ["recall", "false_alerts", "median_lead_time_cycles", "mean_lead_time_cycles"],
        ascending=[False, True, False, False],
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    per_unit_csv.parent.mkdir(parents=True, exist_ok=True)
    alerts_csv.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_csv, index=False)
    best_per_unit.to_csv(per_unit_csv, index=False)
    best_alerts.to_csv(alerts_csv, index=False)
    write_json(
        out_json,
        {
            "fd": args.fd.upper(),
            "split": args.split,
            "model_dir": str(model_dir),
            "model_type": metadata["model_type"],
            "grid_size": int(len(result_df)),
            "best_policy_summary": best_summary,
            "grid_results": result_df.to_dict(orient="records"),
            "outputs": {
                "grid_csv": str(out_csv),
                "per_unit_csv": str(per_unit_csv),
                "alerts_csv": str(alerts_csv),
            },
        },
    )

    print(f"Saved policy grid CSV: {out_csv}")
    print(f"Saved policy grid JSON: {out_json}")
    print(f"Saved best-policy per-unit CSV: {per_unit_csv}")
    print(f"Saved best-policy alerts CSV: {alerts_csv}")
    print(result_df.head(10).to_string(index=False))
    print("Best policy summary:", best_summary)


if __name__ == "__main__":
    main()
