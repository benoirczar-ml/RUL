from __future__ import annotations

import pandas as pd

from rul_pipeline.data import add_train_rul, build_truncated_validation


def test_add_train_rul_with_clipping() -> None:
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 5, 1, 3],
        }
    )
    out = add_train_rul(df, max_rul=2)
    assert out["rul"].tolist() == [2, 2, 0, 2, 0]


def test_build_truncated_validation_produces_prefixes() -> None:
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "rul": [3, 2, 1, 0, 3, 2, 1, 0],
        }
    )

    observed, cuts = build_truncated_validation(df, min_prefix_cycles=2, random_state=7)
    assert set(observed["unit"].unique()) == {1, 2}
    assert len(cuts) == 2

    for _, row in cuts.iterrows():
        assert row["cut_cycle"] >= 2
        assert row["cut_cycle"] <= 3
        unit_obs = observed[observed["unit"] == row["unit"]]
        assert int(unit_obs["cycle"].max()) == int(row["cut_cycle"])
