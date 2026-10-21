"""Static consumer checks, run with mypy (not collected by pytest)."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from forecasters import ChronosBolt, Forecaster, Timestral


def check_structural_typing(model: ChronosBolt | Timestral) -> None:
    forecaster: Forecaster = model
    check_forecaster(forecaster)


def check_forecaster(model: Forecaster) -> None:
    level_tuple: tuple[float, ...] = model.supported_quantile_levels
    context_length: int = model.max_context_length
    horizon_limit: int = model.max_horizon
    assert level_tuple and context_length and horizon_limit


def check_adapter_typing(
    model: ChronosBolt | Timestral,
    np64: NDArray[np.float64],
    np32: NDArray[np.float32],
    tensor: torch.Tensor,
) -> None:
    ndarray64 = model.predict(np64, horizon=2)
    assert_type(ndarray64, NDArray[np.float64])

    ndarray32 = model.predict(np32, horizon=2, quantile_levels=[0.5])
    assert_type(ndarray32, NDArray[np.float32])

    tensor_out = model.predict(tensor, horizon=2)
    assert_type(tensor_out, torch.Tensor)

    # A Python sequence context is outside the public contract.
    model.predict([1.0, 2.0], horizon=2)  # type: ignore[call-overload]  # ty: ignore[no-matching-overload]
