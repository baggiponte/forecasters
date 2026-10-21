"""Real checkpoint parity tests, enabled with FORECASTERS_NATIVE=1.

These tests use the downloaded tiny checkpoints in ``.hf-cache`` and compare the adapter
against a direct call to the official SDK on the same frozen context.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("FORECASTERS_NATIVE") != "1",
        reason="set FORECASTERS_NATIVE=1 to run real-checkpoint parity tests",
    ),
]

CACHE_DIR = Path(__file__).resolve().parents[3] / ".hf-cache"


def frozen_context() -> np.ndarray:
    return (
        2.5
        + np.sin(np.arange(48, dtype=np.float64) / 5.0)
        + np.linspace(0.0, 0.4, 48, dtype=np.float64)
    )


def load_chronos() -> tuple[Any, Any]:
    from chronos import ChronosBoltPipeline

    from forecasters import ChronosBolt

    adapter = ChronosBolt.from_pretrained(
        "amazon/chronos-bolt-tiny", cache_dir=CACHE_DIR, local_files_only=True
    )
    native = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", cache_dir=str(CACHE_DIR), local_files_only=True
    )
    return adapter, native


def load_timestral() -> tuple[Any, Any]:
    from timestral import TimestralForecaster

    from forecasters import Timestral

    adapter = Timestral.from_pretrained(
        "mistralai/Timestral-5M", cache_dir=CACHE_DIR, local_files_only=True
    )
    native = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M", cache_dir=str(CACHE_DIR), local_files_only=True
    )
    return adapter, native


def chronos_raw(native: Any, context: np.ndarray, horizon: int) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(context, dtype=torch.float32)
    raw = native.predict(tensor, prediction_length=horizon)
    return raw.detach()[0].transpose(0, 1).numpy()


def timestral_raw(native: Any, context: np.ndarray, horizon: int) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(context)
    raw = native.forecast(tensor, prediction_length=horizon).quantiles
    return raw.detach()[0, 0].numpy()


LOADERS: list[
    tuple[str, Callable[[], tuple[Any, Any]], Callable[[Any, np.ndarray, int], np.ndarray]]
] = [
    ("chronos-bolt-tiny", load_chronos, chronos_raw),
    ("timestral-5m", load_timestral, timestral_raw),
]


@pytest.mark.parametrize(
    ("name", "loader", "raw_forecast"), LOADERS, ids=[item[0] for item in LOADERS]
)
def test_median_and_requested_quantiles_match_native_sdk(
    name: str,
    loader: Callable[[], tuple[Any, Any]],
    raw_forecast: Callable[[Any, np.ndarray, int], np.ndarray],
) -> None:
    if not CACHE_DIR.exists():
        pytest.skip(f"downloaded checkpoint cache not found at {CACHE_DIR}")
    adapter, native = loader()
    context = frozen_context()
    horizon = 8

    raw = raw_forecast(native, context, horizon)
    levels = adapter.supported_quantile_levels
    median_index = levels.index(0.5)

    median = adapter.predict(context, horizon=horizon)
    assert median.shape == (horizon,)
    assert median.dtype == np.float64
    np.testing.assert_array_equal(median, raw[:, median_index].astype(np.float64))

    requested = [0.9, 0.1, 0.5]
    quantiles = adapter.predict(context, horizon=horizon, quantile_levels=requested)
    assert quantiles.shape == (horizon, len(requested))
    expected = raw[:, [levels.index(level) for level in requested]].astype(np.float64)
    np.testing.assert_array_equal(quantiles, expected)

    singleton = adapter.predict(context, horizon=horizon, quantile_levels=[0.5])
    assert singleton.shape == (horizon, 1)
    np.testing.assert_array_equal(singleton[:, 0], median)
