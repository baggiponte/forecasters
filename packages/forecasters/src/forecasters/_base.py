"""Shared prediction contract for native Torch quantile models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final, overload

from forecasters._outputs import (
    NativeLayout,
    normalize_native_quantiles,
    require_native_tensor,
    to_torch_context,
    validate_native_layout,
)
from forecasters._validation import (
    median_index,
    validate_context,
    validate_horizon,
    validate_quantile_levels,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import numpy.typing as npt
    import torch

    from forecasters._typing import DTypeT


class BaseForecaster(ABC):
    """Implement loading and ``_predict_native``; inherit the prediction contract.

    The native hook receives a one-dimensional Torch tensor and returns the full
    native quantile tensor. Declare its axes with ``native_layout``; batch and
    variate axes, when present, must have size one. No validation, quantile selection,
    or caller-representation restoration belongs in the hook.

    This is an optional authoring interface for Torch quantile models, not a
    requirement for objects satisfying the structural ``Forecaster`` protocol.
    """

    def __init__(
        self,
        *,
        model_name: str,
        supported_quantile_levels: Iterable[float],
        max_context_length: int,
        max_horizon: int,
        native_layout: NativeLayout,
        native_dtype: torch.dtype | None = None,
    ) -> None:
        self._native_layout = validate_native_layout(native_layout)
        if native_dtype is not None and not native_dtype.is_floating_point:
            raise ValueError("native_dtype must be a floating Torch dtype")
        self._model_name = model_name
        self.supported_quantile_levels = tuple(float(level) for level in supported_quantile_levels)
        self.max_context_length = int(max_context_length)
        self.max_horizon = int(max_horizon)
        self._median_index = median_index(self.supported_quantile_levels, model_name=model_name)
        self._native_dtype = native_dtype

    @overload
    def predict(
        self,
        context: npt.NDArray[DTypeT],
        *,
        horizon: int,
        quantile_levels: None = None,
    ) -> np.ndarray[tuple[int], np.dtype[DTypeT]]: ...

    @overload
    def predict(
        self,
        context: npt.NDArray[DTypeT],
        *,
        horizon: int,
        quantile_levels: Sequence[float],
    ) -> np.ndarray[tuple[int, int], np.dtype[DTypeT]]: ...

    @overload
    def predict(
        self,
        context: torch.Tensor,
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> torch.Tensor: ...

    @final
    def predict(
        self,
        context: npt.NDArray[np.number] | torch.Tensor,
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> npt.NDArray[np.number] | torch.Tensor:
        """Return the native median (H,) or requested quantiles (H, Q).

        Preserve the caller's array family, floating dtype and Torch device.
        Reject invalid requests before inference and malformed native output before
        selecting columns. Only selected forecast values must be finite.
        """
        context_spec = validate_context(context)
        horizon = validate_horizon(horizon)
        if context.shape[-1] > self.max_context_length:
            raise ValueError(
                f"context length {context.shape[-1]} exceeds {self._model_name}'s supported "
                f"maximum of {self.max_context_length}",
            )
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds {self._model_name}'s native maximum of {self.max_horizon}",
            )
        indices = validate_quantile_levels(
            quantile_levels,
            self.supported_quantile_levels,
            model_name=self._model_name,
        )

        native_context = to_torch_context(context, dtype=self._native_dtype)
        raw = require_native_tensor(
            self._predict_native(native_context, horizon=horizon),
            model_name=self._model_name,
        )
        quantiles = normalize_native_quantiles(
            raw,
            layout=self._native_layout,
            horizon=horizon,
            quantile_count=len(self.supported_quantile_levels),
            model_name=self._model_name,
        )
        selected = quantiles[:, self._median_index] if indices is None else quantiles[:, indices]
        return context_spec.restore(selected, model_name=self._model_name)

    @abstractmethod
    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        """Call the native SDK and return its full quantile tensor in declared axis order."""
