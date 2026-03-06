from __future__ import annotations

import pandas as pd

from rul_pipeline.operations import evaluate_alert_policy, generate_alerts


def test_generate_alerts_consecutive_logic() -> None:
    pred_df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "pred_rul": [50.0, 25.0, 20.0, 15.0],
        }
    )
    alerts = generate_alerts(pred_df, trigger_rul=25.0, consecutive=2, cooldown_cycles=0)
    assert alerts["cycle"].tolist() == [3]


def test_evaluate_alert_policy_basic_metrics() -> None:
    pred_df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "pred_rul": [100.0, 60.0, 30.0, 10.0, 100.0, 70.0, 50.0, 30.0],
        }
    )
    summary, per_unit, alerts = evaluate_alert_policy(
        pred_df=pred_df,
        trigger_rul=40.0,
        consecutive=1,
        cooldown_cycles=0,
        min_lead=1,
        max_lead=3,
    )

    assert int(summary["total_units"]) == 2
    assert int(summary["detected_units"]) == 1
    assert int(summary["missed_units"]) == 1
    assert abs(float(summary["recall"]) - 0.5) < 1e-9
    assert int(summary["false_alerts"]) >= 1
    assert len(per_unit) == 2
    assert len(alerts) >= 1


def test_generate_alerts_hysteresis_blocks_repeated_alerts() -> None:
    pred_df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "pred_rul": [45.0, 35.0, 30.0, 25.0, 20.0],
        }
    )
    alerts = generate_alerts(
        pred_df,
        trigger_rul=40.0,
        exit_rul=60.0,
        consecutive=1,
        cooldown_cycles=0,
    )
    assert alerts["cycle"].tolist() == [2]


def test_generate_alerts_trend_gate_requires_drop() -> None:
    pred_df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "pred_rul": [120.0, 100.0, 95.0, 92.0, 90.0],
        }
    )
    alerts = generate_alerts(
        pred_df,
        trigger_rul=100.0,
        consecutive=1,
        cooldown_cycles=0,
        trend_window=2,
        trend_delta=20.0,
    )
    assert alerts["cycle"].tolist() == [3]
