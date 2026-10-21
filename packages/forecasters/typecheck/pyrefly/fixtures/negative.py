"""Intentional static failures for the Pyrefly tensor-shape overlay.

Not executed. Each EXPECT comment names the error kind that must be reported on the
following line, and no other errors may appear. ``run.py`` fails if an expected error
disappears.
"""

from __future__ import annotations

import numpy as np
import torch

from forecasters import ChronosBolt


def check(
    model: ChronosBolt,
    context: np.ndarray[[7], np.dtype[np.float64]],
    matrix: np.ndarray[[7, 2], np.dtype[np.float64]],
    tensor: torch.Tensor[[7]],
) -> None:
    median = model.predict(context, horizon=5)
    quantiles = model.predict(context, horizon=5, quantile_levels=[0.5])

    # EXPECT: bad-assignment
    wrong_horizon: np.ndarray[[6], np.dtype[np.float64]] = median
    # EXPECT: bad-assignment
    wrong_rank: np.ndarray[[5, 2], np.dtype[np.float64]] = median
    # EXPECT: bad-assignment
    wrong_quantile_rank: np.ndarray[[5], np.dtype[np.float64]] = quantiles
    # EXPECT: bad-assignment
    wrong_dtype: np.ndarray[[5], np.dtype[np.float32]] = median
    # EXPECT: bad-assignment
    wrong_quantile_horizon: np.ndarray[[6, int], np.dtype[np.float64]] = quantiles
    # EXPECT: no-matching-overload
    model.predict(matrix, horizon=5)
    # EXPECT: no-matching-overload
    model.predict([1.0, 2.0], horizon=5)
    # EXPECT: no-matching-overload
    model.predict(context, horizon=5.0)
    # EXPECT: no-matching-overload
    model.predict(context)
    # EXPECT: no-matching-overload
    model.predict(context, horizon=5, quantile_levels=["0.5"])
    # EXPECT: bad-assignment
    wrong_backend: torch.Tensor[[5]] = median
    # EXPECT: bad-assignment
    torch_wrong_rank: torch.Tensor[[5, 2]] = model.predict(tensor, horizon=5)
    # EXPECT: no-matching-overload
    model.predict(np.zeros(7, dtype=np.int64), horizon=5)
    # EXPECT: no-matching-overload
    model.predict(np.zeros(7, dtype=np.bool_), horizon=5)
