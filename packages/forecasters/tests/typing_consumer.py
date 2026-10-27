"""Static consumer checks, run with ty (not collected by pytest)."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import numpy as np
import torch

from forecasters import BaseForecaster, ChronosBolt, Forecaster, Timestral

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CustomForecaster(BaseForecaster):
    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        return context[:horizon, None]


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
    npld: NDArray[np.longdouble],
    nps: NDArray[np.str_],
    nbool: NDArray[np.bool_],
    tensor: torch.Tensor,
) -> None:
    median64 = model.predict(np64, horizon=2)
    assert_type(median64, np.ndarray[tuple[int], np.dtype[np.float64]])

    quantiles32 = model.predict(np32, horizon=2, quantile_levels=[0.5])
    assert_type(quantiles32, np.ndarray[tuple[int, int], np.dtype[np.float32]])

    assert_type(model.predict(npld, horizon=2), np.ndarray[tuple[int], np.dtype[np.longdouble]])

    tensor_out = model.predict(tensor, horizon=2)
    assert_type(tensor_out, torch.Tensor)

    # A Python sequence context is outside the public contract.
    model.predict([1.0, 2.0], horizon=2)  # ty: ignore[no-matching-overload]

    # String arrays are not numeric and fail the dtype bound.
    model.predict(nps, horizon=2)  # ty: ignore[no-matching-overload]

    # Boolean arrays are not numeric and fail the dtype bound.
    model.predict(nbool, horizon=2)  # ty: ignore[no-matching-overload]


def check_extension_typing(
    model: CustomForecaster,
    np64: NDArray[np.float64],
    tensor: torch.Tensor,
) -> None:
    forecaster: Forecaster = model
    check_forecaster(forecaster)
    assert_type(model.predict(np64, horizon=2), np.ndarray[tuple[int], np.dtype[np.float64]])
    assert_type(
        model.predict(np64, horizon=2, quantile_levels=[0.5]),
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    )
    assert_type(model.predict(tensor, horizon=2, quantile_levels=[0.5]), torch.Tensor)
