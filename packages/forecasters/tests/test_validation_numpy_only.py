"""Tests that collect and run without PyTorch or either native SDK."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest

from forecasters._deps import import_torch
from forecasters._validation import validate_context, validate_horizon, validate_quantile_levels

SUPPORTED = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@pytest.mark.parametrize(
    "context",
    [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1, 2], dtype=np.int64),
        np.array([True, False]),
        np.array([1.0 + 0j], dtype=np.complex128),
        np.array(1.0),
        np.ones((2, 2), dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_invalid_numpy_contexts_are_rejected(context: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_context(context)


def test_arbitrary_array_api_object_is_not_in_scope() -> None:
    class ForeignArray:
        ndim = 1
        size = 2
        shape = (2,)
        dtype = "float32"

        def __array_namespace__(self, *, api_version: str | None = None) -> Any:
            raise AssertionError("scope check must happen before namespace use")

    with pytest.raises(TypeError, match="NumPy ndarray or PyTorch Tensor"):
        validate_context(ForeignArray())


@pytest.mark.parametrize("horizon", [3, np.int64(3)])
def test_valid_horizons(horizon: Any) -> None:
    assert validate_horizon(horizon) == 3


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "2"])
def test_invalid_horizons(horizon: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_horizon(horizon)


def test_quantile_matching_preserves_order_and_float32_representation() -> None:
    indices = validate_quantile_levels(
        [0.9, float(np.float32(0.1)), 0.5], SUPPORTED, model_name="fake"
    )
    assert indices == [8, 0, 4]


def test_quantile_failures_identify_model_and_levels() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_quantile_levels([0.1, 0.33], SUPPORTED, model_name="fake/checkpoint")
    text = str(excinfo.value)
    assert "fake/checkpoint" in text and "0.33" in text and "supported native levels" in text

    with pytest.raises(ValueError, match="unique"):
        validate_quantile_levels([0.1, float(np.float32(0.1))], SUPPORTED, model_name="fake")


def test_import_torch_failure_hint_avoids_unpublished_timestral_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError) as excinfo:
        import_torch()
    text = str(excinfo.value)
    assert "forecasters[chronos]" in text
    assert "forecasters[timestral]" not in text
    assert "git+https://github.com/mistralai/timestral.git" in text
