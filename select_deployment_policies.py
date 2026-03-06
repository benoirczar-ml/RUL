from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent


def _resolve(path_arg: str) -> Path:
    p = Path(path_arg)
    return (ROOT / p).resolve() if not p.is_absolute() else p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select deployment-ready alert policies under business constraints.")
    parser.add_argument(
        "--policy-dir",
        default="outputs/ops_calibration_v2",
        help="Directory containing FDxxx_policy_grid.csv files.",
    )
    parser.add_argument(
        "--fds",
        default="FD001,FD002,FD003,FD004",
        help="Comma-separated FD list.",
    )
    parser.add_argument("--min-recall", type=float, default=0.98, help="Minimum acceptable recall.")
    parser.add_argument(
        "--max-false-alert-rate",
        type=float,
        default=0.30,
        help="Maximum acceptable false alert rate.",
    )
    parser.add_argument(
        "--min-median-lead",
        type=float,
        default=60.0,
        help="Minimum acceptable median lead time (cycles).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/deployment_policies",
        help="Output dir for selected policy configs and summary files.",
    )
    return parser.parse_args()


def _parse_fd_list(csv_text: str) -> list[str]:
    return [x.strip().upper() for x in csv_text.split(",") if x.strip()]


def _to_optional_float(v: Any) -> float | None:
    if pd.isna(v):
        return None
    return float(v)


def _select_policy_row(
    grid: pd.DataFrame,
    min_recall: float,
    max_false_alert_rate: float,
    min_median_lead: float,
) -> tuple[pd.Series, bool]:
    filtered = grid.copy()
    filtered = filtered[filtered["recall"] >= min_recall]
    filtered = filtered[filtered["false_alert_rate"] <= max_false_alert_rate]
    filtered = filtered[filtered["median_lead_time_cycles"] >= min_median_lead]

    if len(filtered) > 0:
        chosen = filtered.sort_values(
            ["false_alerts", "false_alert_rate", "median_lead_time_cycles"],
            ascending=[True, True, False],
        ).iloc[0]
        return chosen, True

    fallback = grid.sort_values(
        ["recall", "false_alerts", "false_alert_rate", "median_lead_time_cycles"],
        ascending=[False, True, True, False],
    ).iloc[0]
    return fallback, False


def main() -> None:
    args = parse_args()
    policy_dir = _resolve(args.policy_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fds = _parse_fd_list(args.fds)

    rows: list[dict[str, Any]] = []
    for fd in fds:
        grid_path = policy_dir / f"{fd}_policy_grid.csv"
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing grid file: {grid_path}")
        grid = pd.read_csv(grid_path)
        if len(grid) == 0:
            raise ValueError(f"Empty grid file: {grid_path}")

        selected, meets = _select_policy_row(
            grid=grid,
            min_recall=args.min_recall,
            max_false_alert_rate=args.max_false_alert_rate,
            min_median_lead=args.min_median_lead,
        )

        constraints = {
            "min_recall": float(args.min_recall),
            "max_false_alert_rate": float(args.max_false_alert_rate),
            "min_median_lead_time_cycles": float(args.min_median_lead),
        }
        policy_cfg = {
            "fd": fd,
            "meets_constraints": bool(meets),
            "constraints": constraints,
            "policy": {
                "trigger_rul": float(selected["trigger_rul"]),
                "exit_rul": _to_optional_float(selected["exit_rul"]),
                "consecutive": int(selected["consecutive"]),
                "cooldown_cycles": int(selected["cooldown_cycles"]),
                "trend_window": int(selected["trend_window"]),
                "trend_delta": float(selected["trend_delta"]),
                "min_lead": 5,
                "max_lead": 120,
            },
            "metrics": {
                "recall": float(selected["recall"]),
                "missed_units": int(selected["missed_units"]),
                "false_alerts": int(selected["false_alerts"]),
                "false_alert_rate": float(selected["false_alert_rate"]),
                "median_lead_time_cycles": float(selected["median_lead_time_cycles"]),
                "mean_lead_time_cycles": float(selected["mean_lead_time_cycles"]),
                "total_alerts": int(selected["total_alerts"]),
            },
            "source_grid_csv": str(grid_path),
        }

        cfg_path = output_dir / f"policy_config_{fd}.json"
        cfg_path.write_text(json.dumps(policy_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        rows.append(
            {
                "fd": fd,
                "meets_constraints": bool(meets),
                "trigger_rul": float(selected["trigger_rul"]),
                "exit_rul": _to_optional_float(selected["exit_rul"]),
                "consecutive": int(selected["consecutive"]),
                "cooldown_cycles": int(selected["cooldown_cycles"]),
                "trend_window": int(selected["trend_window"]),
                "trend_delta": float(selected["trend_delta"]),
                "recall": float(selected["recall"]),
                "false_alert_rate": float(selected["false_alert_rate"]),
                "median_lead_time_cycles": float(selected["median_lead_time_cycles"]),
                "false_alerts": int(selected["false_alerts"]),
                "total_alerts": int(selected["total_alerts"]),
                "config_path": str(cfg_path),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("fd").reset_index(drop=True)
    summary_csv = output_dir / "deployment_policy_selection.csv"
    summary_json = output_dir / "deployment_policy_selection.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "n_fd": int(len(summary_df)),
                "n_meeting_constraints": int(summary_df["meets_constraints"].sum()),
                "constraints": {
                    "min_recall": float(args.min_recall),
                    "max_false_alert_rate": float(args.max_false_alert_rate),
                    "min_median_lead_time_cycles": float(args.min_median_lead),
                },
                "rows": summary_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved deployment summary CSV: {summary_csv}")
    print(f"Saved deployment summary JSON: {summary_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
