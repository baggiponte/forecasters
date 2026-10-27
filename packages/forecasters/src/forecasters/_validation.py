"""Validate requests before native inference."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from array_api_compat import device, is_numpy_array, is_torch_array

if TYPE_CHECKING:
    import numpy as np
    import torch

# Match float32 encodings of grid levels without interpolating between probabilities.
_NATIVE_LEVEL_ABS_TOL = 1e-7


@dataclass(frozen=True)
class NumpySpec:
    """Validated NumPy context metadata used to restore the caller's representation."""

    family: Literal["numpy"]
    dtype: np.dtype[Any]


@dataclass(frozen=True)
class TorchSpec:
    """Validated Torch context metadata used to restore the caller's representation."""

    family: Literal["torch"]
    dtype: torch.dtype
    device: torch.device | str


InputSpec = NumpySpec | TorchSpec


def validate_context(context: object) -> InputSpec:
    if is_numpy_array(context):
        array = context
        if array.ndim != 1:
            raise ValueError(f"context must be one-dimensional; got shape {tuple(array.shape)}")
        if array.shape[0] == 0:
            raise ValueError("context must contain at least one observation")
        return NumpySpec(family="numpy", dtype=array.dtype)

    if is_torch_array(context):
        tensor = context
        if tensor.ndim != 1:
            raise ValueError(f"context must be one-dimensional; got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 0:
            raise ValueError("context must contain at least one observation")
        return TorchSpec(family="torch", dtype=tensor.dtype, device=device(tensor))

    raise TypeError(
        "context must be a NumPy ndarray or PyTorch Tensor; "
        f"got {type(context).__name__}. Python sequences and other array libraries are "
        "not accepted as context (only quantile_levels may be a sequence)."
    )


def validate_horizon(horizon: object) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral):
        raise TypeError(f"horizon must be a positive integer; got {horizon!r}")
    value = int(horizon)
    if value <= 0:
        raise ValueError(f"horizon must be positive; got {value}")
    return value


def _find_native_level(level: float, supported: tuple[float, ...]) -> int | None:
    for index, native_level in enumerate(supported):
        if math.isclose(level, native_level, rel_tol=0.0, abs_tol=_NATIVE_LEVEL_ABS_TOL):
            return index
    return None


def median_index(supported: tuple[float, ...], *, model_name: str) -> int:
    index = _find_native_level(0.5, supported)
    if index is None:
        raise RuntimeError(
            f"{model_name} does not expose a native 0.5 quantile; supported levels are {supported!r}"
        )
    return index


def validate_quantile_levels(
    levels: object,
    supported: tuple[float, ...],
    *,
    model_name: str,
) -> list[int] | None:
    """Validate requested levels and return their indices in the native grid.

    Matching tolerates float32 encodings of the native levels but never substitutes a
    nearby statistical level. Requested order is retained.
    """
    if levels is None:
        return None

    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise TypeError(
            "quantile_levels must be a nonempty sequence of probabilities between 0 and 1; "
            f"got {type(levels).__name__}"
        )

    requested: list[float] = []
    for item in levels:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise TypeError(
                f"quantile levels must be real numbers, not {type(item).__name__}: {item!r}"
            )
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"quantile levels must be finite; got {value!r}")
        if not 0.0 < value < 1.0:
            raise ValueError(f"quantile levels must lie strictly between 0 and 1; got {value!r}")
        requested.append(value)

    if not requested:
        raise ValueError("quantile_levels must not be empty")

    indices: list[int] = []
    unavailable: list[float] = []
    for value in requested:
        index = _find_native_level(value, supported)
        if index is None:
            unavailable.append(value)
        else:
            indices.append(index)

    if unavailable:
        raise ValueError(
            f"{model_name} does not support the requested quantile levels {unavailable!r}; "
            f"supported native levels are {supported!r}"
        )

    if len(set(indices)) != len(indices):
        raise ValueError(
            "quantile_levels must be unique after matching the loaded checkpoint's native levels; "
            f"got {requested!r} for native levels {supported!r}"
        )

    return indices
