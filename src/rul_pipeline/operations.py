from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def generate_alerts(
    pred_df: pd.DataFrame,
    trigger_rul: float,
    consecutive: int = 1,
    cooldown_cycles: int = 0,
) -> pd.DataFrame:
    required_cols = {"unit", "cycle", "pred_rul"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"pred_df must include columns {required_cols}")
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    if cooldown_cycles < 0:
        raise ValueError("cooldown_cycles must be >= 0")

    rows: list[dict] = []
    sorted_df = pred_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    for unit, grp in sorted_df.groupby("unit", sort=True):
        g = grp.sort_values("cycle")
        streak = 0
        last_alert_cycle = -10**9
        for _, row in g.iterrows():
            cycle = int(row["cycle"])
            pred_rul = float(row["pred_rul"])
            if pred_rul <= trigger_rul:
                streak += 1
            else:
                streak = 0

            can_alert = cycle - last_alert_cycle > cooldown_cycles
            if streak >= consecutive and can_alert:
                rows.append({"unit": int(unit), "cycle": cycle, "pred_rul": pred_rul})
                last_alert_cycle = cycle
                streak = 0

    return pd.DataFrame(rows, columns=["unit", "cycle", "pred_rul"])


def evaluate_alert_policy(
    pred_df: pd.DataFrame,
    trigger_rul: float,
    consecutive: int = 1,
    cooldown_cycles: int = 0,
    min_lead: int = 1,
    max_lead: int | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required_cols = {"unit", "cycle", "pred_rul"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"pred_df must include columns {required_cols}")
    if min_lead < 0:
        raise ValueError("min_lead must be >= 0")
    if max_lead is not None and max_lead < min_lead:
        raise ValueError("max_lead must be >= min_lead or None")

    sorted_df = pred_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    alerts_df = generate_alerts(
        pred_df=sorted_df,
        trigger_rul=trigger_rul,
        consecutive=consecutive,
        cooldown_cycles=cooldown_cycles,
    )

    per_unit_rows: list[dict] = []
    total_alerts = 0
    true_alerts = 0
    false_alerts = 0
    lead_times: list[float] = []

    for unit, grp in sorted_df.groupby("unit", sort=True):
        g = grp.sort_values("cycle")
        failure_cycle = int(g["cycle"].max())
        unit_alerts = alerts_df[alerts_df["unit"] == int(unit)].sort_values("cycle")

        unit_total_alerts = int(len(unit_alerts))
        total_alerts += unit_total_alerts

        selected_lead = None
        unit_true_alerts = 0
        unit_false_alerts = 0
        for _, arow in unit_alerts.iterrows():
            alert_cycle = int(arow["cycle"])
            lead = failure_cycle - alert_cycle
            in_window = lead >= min_lead and (max_lead is None or lead <= max_lead)
            if in_window:
                unit_true_alerts += 1
                if selected_lead is None:
                    selected_lead = float(lead)
            else:
                unit_false_alerts += 1

        true_alerts += unit_true_alerts
        false_alerts += unit_false_alerts

        detected = selected_lead is not None
        if detected:
            lead_times.append(float(selected_lead))

        per_unit_rows.append(
            {
                "unit": int(unit),
                "failure_cycle": failure_cycle,
                "n_alerts": unit_total_alerts,
                "n_true_alerts": unit_true_alerts,
                "n_false_alerts": unit_false_alerts,
                "detected": bool(detected),
                "lead_time_cycles": selected_lead,
            }
        )

    per_unit_df = pd.DataFrame(per_unit_rows).sort_values("unit").reset_index(drop=True)
    total_units = int(len(per_unit_df))
    detected_units = int(per_unit_df["detected"].sum())
    missed_units = total_units - detected_units
    recall = float(detected_units / total_units) if total_units else 0.0
    median_lead = float(pd.Series(lead_times).median()) if lead_times else math.nan
    mean_lead = float(pd.Series(lead_times).mean()) if lead_times else math.nan
    alerts_per_unit = float(total_alerts / total_units) if total_units else 0.0
    false_alert_rate = float(false_alerts / total_alerts) if total_alerts else 0.0

    summary = {
        "total_units": total_units,
        "detected_units": detected_units,
        "missed_units": missed_units,
        "recall": recall,
        "median_lead_time_cycles": median_lead,
        "mean_lead_time_cycles": mean_lead,
        "total_alerts": int(total_alerts),
        "true_alerts": int(true_alerts),
        "false_alerts": int(false_alerts),
        "false_alert_rate": false_alert_rate,
        "alerts_per_unit": alerts_per_unit,
        "policy": {
            "trigger_rul": float(trigger_rul),
            "consecutive": int(consecutive),
            "cooldown_cycles": int(cooldown_cycles),
            "min_lead": int(min_lead),
            "max_lead": None if max_lead is None else int(max_lead),
        },
    }
    return summary, per_unit_df, alerts_df


def parse_int_list_csv(csv_text: str) -> list[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def parse_float_list_csv(csv_text: str) -> list[float]:
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def iter_policy_grid(
    trigger_ruls: Iterable[float],
    consecutives: Iterable[int],
    cooldown_cycles: Iterable[int],
) -> Iterable[tuple[float, int, int]]:
    for tr in trigger_ruls:
        for c in consecutives:
            for cd in cooldown_cycles:
                yield float(tr), int(c), int(cd)

