"""Positive static fixtures for the Pyrefly tensor-shape overlay.

Not executed. ``run.py`` requires a clean Pyrefly check: every ``assert_type`` below
fails if inference regresses. The shape-aware stubs are installed only in the isolated
Pyrefly environment.
"""

from __future__ import annotations

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from forecasters import ChronosBolt, Forecaster, Timestral
from forecasters._forecaster import Forecaster as ForecasterFromModule
from forecasters.chronos_bolt import ChronosBolt as ChronosBoltFromModule
from forecasters.timestral import Timestral as TimestralFromModule


def check_concrete_adapters(
    model: ChronosBolt | Timestral,
    context: np.ndarray[[7], np.dtype[np.float64]],
) -> None:
    median = model.predict(context, horizon=5)
    assert_type(median, np.ndarray[[5], np.dtype[np.float64]])

    quantiles = model.predict(context, horizon=5, quantile_levels=[0.9, 0.5, 0.1])
    # The quantile axis is gradual; rank, horizon and dtype are proven.
    assert_type(quantiles, np.ndarray[[5, int], np.dtype[np.float64]])

    # The overlay composes with the shape-aware NumPy stubs rather than only decorating
    # the call site.
    assert_type(quantiles[:, 0], np.ndarray[[5], np.dtype[np.float64]])
    assert_type(quantiles.T, np.ndarray[[int, 5], np.dtype[np.float64]])


def check_numpy_dtype_preservation(
    model: ChronosBolt,
    f32: np.ndarray[[12], np.dtype[np.float32]],
    f64: np.ndarray[[12], np.dtype[np.float64]],
    ld: np.ndarray[[12], np.dtype[np.longdouble]],
    generic: NDArray[np.float64],
) -> None:
    assert_type(model.predict(f32, horizon=3), np.ndarray[[3], np.dtype[np.float32]])
    assert_type(model.predict(f64, horizon=3), np.ndarray[[3], np.dtype[np.float64]])
    assert_type(model.predict(ld, horizon=3), np.ndarray[[3], np.dtype[np.longdouble]])
    assert_type(model.predict(generic, horizon=4), np.ndarray[[4], np.dtype[np.float64]])
    assert_type(
        model.predict(generic, horizon=4, quantile_levels=[0.5]),
        np.ndarray[[4, int], np.dtype[np.float64]],
    )


def check_torch_backend(model: Timestral, context: torch.Tensor[[8]]) -> None:
    median = model.predict(context, horizon=5)
    assert_type(median, torch.Tensor[[5]])

    quantiles = model.predict(context, horizon=5, quantile_levels=[0.1, 0.5])
    assert_type(quantiles, torch.Tensor[[5, int]])
    assert_type(quantiles[:, 0], torch.Tensor[[5]])
    assert_type(quantiles.T, torch.Tensor[[int, 5]])


def check_protocol(model: Forecaster, context: np.ndarray[[6], np.dtype[np.float64]]) -> None:
    levels: tuple[float, ...] = model.supported_quantile_levels
    context_limit: int = model.max_context_length
    horizon_limit: int = model.max_horizon
    assert levels and context_limit and horizon_limit

    assert_type(model.predict(context, horizon=2), np.ndarray[[2], np.dtype[np.float64]])


def check_dynamic_horizon(
    model: ChronosBolt,
    context: np.ndarray[[7], np.dtype[np.float64]],
    horizon: int,
) -> None:
    # The horizon dimension stays gradual when it is not a literal, but the rank does not.
    assert_type(model.predict(context, horizon=horizon), np.ndarray[[int], np.dtype[np.float64]])


def check_module_imports(
    model: ChronosBoltFromModule | TimestralFromModule,
    context: np.ndarray[[7], np.dtype[np.float64]],
) -> None:
    # The overlay also covers the submodule import paths used by the package itself.
    assert_type(model.predict(context, horizon=4), np.ndarray[[4], np.dtype[np.float64]])
    assert_type(ChronosBoltFromModule.from_pretrained(), ChronosBoltFromModule)
    forecaster: ForecasterFromModule = model
    assert_type(forecaster.predict(context, horizon=3), np.ndarray[[3], np.dtype[np.float64]])
