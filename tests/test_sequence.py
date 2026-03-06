from __future__ import annotations

import numpy as np
import pandas as pd

from rul_pipeline.sequence import build_sequence_samples


def test_build_sequence_samples_last_only_shapes() -> None:
    features = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 10.0, 11.0],
            "f2": [0.5, 0.6, 0.7, 1.0, 1.1],
        }
    )
    units = np.array([1, 1, 1, 2, 2])
    targets = np.array([4.0, 3.0, 2.0, 1.0, 0.0])

    x_seq, y_seq, unit_seq = build_sequence_samples(
        features=features,
        units=units,
        targets=targets,
        seq_len=4,
        sample_step=1,
        last_only=True,
    )

    assert x_seq.shape == (2, 4, 2)
    assert y_seq.shape == (2,)
    assert unit_seq.tolist() == [1, 2]
    assert np.isclose(y_seq[0], 2.0)
    assert np.isclose(y_seq[1], 0.0)

