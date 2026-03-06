from __future__ import annotations

import pandas as pd

from rul_pipeline.data import add_train_rul


def test_add_train_rul_with_clipping() -> None:
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 5, 1, 3],
        }
    )
    out = add_train_rul(df, max_rul=2)
    assert out["rul"].tolist() == [2, 2, 0, 2, 0]

