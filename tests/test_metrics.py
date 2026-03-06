from __future__ import annotations

import numpy as np

from rul_pipeline.metrics import phm_score


def test_phm_score_penalizes_late_predictions_more() -> None:
    y_true = np.array([100.0, 100.0], dtype=float)
    y_pred_early = np.array([90.0, 90.0], dtype=float)
    y_pred_late = np.array([110.0, 110.0], dtype=float)

    early_score = phm_score(y_true, y_pred_early)
    late_score = phm_score(y_true, y_pred_late)

    assert late_score > early_score

