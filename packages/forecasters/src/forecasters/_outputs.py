"""Native input normalization and output restoration helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from array_api_compat import array_namespace, is_torch_array

from forecasters._validation import InputSpec

if TYPE_CHECKING:
    import torch


def numpy_native_context(context: np.ndarray) -> np.ndarray:
    """Make a NumPy context safe for the SDK's Torch bridge without mutating the caller."""
    array = np.ascontiguousarray(context)
    if not array.flags.writeable:
        array = array.copy()
    if array.dtype == np.dtype(np.longdouble) and array.dtype.itemsize > 8:
        converted = array.astype(np.float64, copy=False)
        if not np.isfinite(converted).all():
            raise ValueError(
                "NumPy longdouble context cannot be represented in float64 for native "
                "inference without overflow"
            )
        array = converted
    return array


def require_native_tensor(values: object, *, model_name: str) -> torch.Tensor:
    """Reject malformed SDK return types before any shape or value access."""
    if not is_torch_array(values):
        raise RuntimeError(
            f"{model_name} native SDK returned {type(values).__name__}; expected a torch.Tensor"
        )
    return cast("torch.Tensor", values)


def _finite(values: object) -> bool:
    xp = array_namespace(values)
    return bool(xp.all(xp.isfinite(values)))


def restore_output(values: torch.Tensor, spec: InputSpec, *, model_name: str) -> object:
    """Validate selected values, then restore the caller's family, dtype and device."""
    if not _finite(values):
        raise RuntimeError(
            f"{model_name} native SDK returned non-finite selected forecast values"
        )

    restored: object
    if spec.family == "torch":
        restored = values.detach().to(
            device=cast("str | torch.device | None", spec.device),
            dtype=cast("torch.dtype", spec.dtype),
        )
        destination = f"device {spec.device}"
    else:
        array = values.detach().to(device="cpu").numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            restored = array.astype(cast("np.dtype[Any]", spec.dtype), copy=False)
        destination = f"NumPy dtype {spec.dtype}"

    if not _finite(restored):
        raise RuntimeError(
            f"{model_name} forecast values are not representable in the requested {destination}; "
            "conversion produced non-finite values"
        )
    return restored
