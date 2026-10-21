"""Adapter for Mistral AI's Timestral checkpoints through the official SDK."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

from forecasters._deps import import_timestral_forecaster, import_torch
from forecasters._native import timestral_quantiles
from forecasters._outputs import numpy_native_context, require_native_tensor, restore_output
from forecasters._validation import (
    InputSpec,
    median_index,
    validate_context,
    validate_horizon,
    validate_quantile_levels,
)

if TYPE_CHECKING:
    import torch

_FloatDType = TypeVar("_FloatDType", bound=np.floating[Any])

_DEFAULT_CHECKPOINT = "mistralai/Timestral-5M"


class Timestral:
    """A loaded Timestral checkpoint exposed through the minimal forecasting interface.

    Construction goes through :meth:`from_pretrained`; the native forecaster is retained
    and reused for every prediction.
    """

    _forecaster: Any
    _model_name: str
    _levels: tuple[float, ...]
    _max_context_length: int
    _max_horizon: int
    _median_index: int

    def __init__(self, native: Any, *, model_name: str) -> None:
        """Wrap an already-loaded SDK object; use from_pretrained to load a checkpoint."""
        self._forecaster = native
        self._model_name = model_name
        self._levels = tuple(float(level) for level in native.quantile_levels)
        self._max_context_length = int(native.max_context)
        self._max_horizon = int(native.per_shot_horizon_patches) * int(native.patch_size)
        self._median_index = median_index(self._levels, model_name=self._model_name)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = _DEFAULT_CHECKPOINT,
        *,
        device: str = "cpu",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> Timestral:
        """Load an official Timestral checkpoint via the Mistral AI ``timestral`` package.

        Args:
            model_name: Hugging Face repository id or local export directory.
            device: Native PyTorch compute device, for example ``"cpu"`` or ``"cuda"``.
            revision: Optional Hugging Face revision.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Read the checkpoint from cache without network access.
        """
        forecaster_class = import_timestral_forecaster()
        native = forecaster_class.from_pretrained(
            model_name,
            device=device,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        return cls(native, model_name=model_name)

    @property
    def supported_quantile_levels(self) -> tuple[float, ...]:
        return self._levels

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @property
    def max_horizon(self) -> int:
        return self._max_horizon

    @overload
    def predict(
        self,
        context: NDArray[_FloatDType],
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> NDArray[_FloatDType]: ...

    @overload
    def predict(
        self,
        context: torch.Tensor,
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> torch.Tensor: ...

    def predict(
        self,
        context: NDArray[np.floating[Any]] | torch.Tensor,
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> NDArray[np.floating[Any]] | torch.Tensor:
        """Forecast a one-dimensional floating context with the loaded Timestral model.

        Without ``quantile_levels`` the result is the native median vector ``(H,)``. With
        requested levels it is ``(H, Q)`` in the caller's order, selected directly from
        the same native quantile tensor.
        """
        spec: InputSpec = validate_context(context)
        horizon = validate_horizon(horizon)

        if context.shape[-1] > self._max_context_length:
            raise ValueError(
                f"context length {context.shape[-1]} exceeds {self._model_name}'s supported "
                f"maximum of {self._max_context_length}"
            )
        if horizon > self._max_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds {self._model_name}'s native maximum of {self._max_horizon}"
            )

        indices = validate_quantile_levels(
            quantile_levels,
            self._levels,
            model_name=self._model_name,
        )

        torch_module = import_torch()
        if spec.family == "numpy":
            native_values = numpy_native_context(cast(NDArray[Any], context))
            native_context = torch_module.as_tensor(native_values)
        else:
            native_context = torch_module.as_tensor(context)

        forecast = self._forecaster.forecast(native_context, prediction_length=horizon)
        raw = require_native_tensor(
            getattr(forecast, "quantiles", None), model_name=self._model_name
        )
        # Timestral returns exactly (batch=1, variates=1, horizon, loaded quantile grid).
        expected_shape = (1, 1, horizon, len(self._levels))
        if tuple(raw.shape) != expected_shape:
            raise RuntimeError(
                f"{self._model_name} native SDK returned shape {tuple(raw.shape)}; "
                f"expected exactly {expected_shape}"
            )
        selected = timestral_quantiles(raw, median_index=self._median_index, indices=indices)
        return cast(
            "NDArray[np.floating[Any]] | torch.Tensor",
            restore_output(selected, spec, model_name=self._model_name),
        )
