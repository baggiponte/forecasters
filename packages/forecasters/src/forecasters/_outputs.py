"""Native input normalization and output restoration helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from array_api_compat import array_namespace, is_torch_array

from forecasters._deps import import_torch

if TYPE_CHECKING:
    import torch

    from forecasters._validation import InputSpec

NativeLayout = tuple[Literal["batch", "variate", "horizon", "quantile"], ...]


def validate_native_layout(layout: NativeLayout) -> NativeLayout:
    axes = set(layout)
    if (
        len(axes) != len(layout)
        or not {"horizon", "quantile"} <= axes
        or not axes <= {"batch", "variate", "horizon", "quantile"}
    ):
        raise ValueError(
            "native_layout must contain one 'horizon' and one 'quantile' axis, "
            "with optional unique 'batch' and 'variate' axes"
        )
    return tuple(layout)


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


def to_native_context(
    context: object, context_spec: InputSpec, *, dtype: torch.dtype | None
) -> torch.Tensor:
    """Bridge a validated context to Torch using the SDK's requested compute dtype."""
    values = (
        numpy_native_context(cast("np.ndarray", context))
        if context_spec.family == "numpy"
        else context
    )
    return cast("torch.Tensor", import_torch().as_tensor(values, dtype=dtype))


def require_native_tensor(values: object, *, model_name: str) -> torch.Tensor:
    if not is_torch_array(values):
        raise RuntimeError(
            f"{model_name} native SDK returned {type(values).__name__}; expected a torch.Tensor"
        )
    return values


def normalize_native_quantiles(
    values: object,
    *,
    layout: NativeLayout,
    horizon: int,
    quantile_count: int,
    model_name: str,
) -> torch.Tensor:
    """Check the complete SDK tensor before normalizing its axes to (H, Q)."""
    tensor = require_native_tensor(values, model_name=model_name)
    sizes = {"horizon": horizon, "quantile": quantile_count, "batch": 1, "variate": 1}
    expected_shape = tuple(sizes[axis] for axis in layout)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{model_name} native SDK returned shape {tuple(tensor.shape)}; "
            f"expected exactly {expected_shape}"
        )
    # All other axes have already been checked to be singleton dimensions.
    return (
        tensor.detach()
        .movedim((layout.index("horizon"), layout.index("quantile")), (0, 1))
        .reshape(horizon, quantile_count)
    )


def _all_finite(values: object) -> bool:
    xp = array_namespace(values)
    return bool(xp.all(xp.isfinite(values)))


def restore_output(values: torch.Tensor, context_spec: InputSpec, *, model_name: str) -> object:
    """Validate selected values, then restore the caller's family, dtype and device."""
    if not _all_finite(values):
        raise RuntimeError(f"{model_name} native SDK returned non-finite selected forecast values")

    restored: object
    if context_spec.family == "torch":
        restored = values.detach().to(
            device=context_spec.device,
            dtype=cast("torch.dtype", context_spec.dtype),
        )
        destination = f"device {context_spec.device}"
    else:
        cpu_values = values.detach().to(device="cpu")
        # NumPy has no bfloat16 dtype; widening to float32 preserves every value.
        if cpu_values.dtype == import_torch().bfloat16:
            cpu_values = cpu_values.float()
        array = cpu_values.numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            restored = array.astype(cast("np.dtype[Any]", context_spec.dtype), copy=False)
        destination = f"NumPy dtype {context_spec.dtype}"

    if not _all_finite(restored):
        raise RuntimeError(
            f"{model_name} forecast values are not representable in the requested {destination}; "
            "conversion produced non-finite values"
        )
    return restored
