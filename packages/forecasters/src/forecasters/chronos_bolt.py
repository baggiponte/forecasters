"""Adapter for Amazon's Chronos-Bolt pipelines through the official Chronos SDK."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray

from forecasters._deps import import_chronos_pipeline, import_torch
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

_DEFAULT_CHECKPOINT = "amazon/chronos-bolt-tiny"


class ChronosBolt:
    """A loaded Chronos-Bolt checkpoint."""

    _pipeline: Any
    _model_name: str
    _levels: tuple[float, ...]
    _max_context_length: int
    _max_horizon: int
    _median_index: int

    def __init__(self, native: Any, *, model_name: str) -> None:
        self._pipeline = native
        self._model_name = model_name
        self._levels = tuple(float(level) for level in native.quantiles)
        self._max_context_length = int(native.model_context_length)
        self._max_horizon = int(native.model_prediction_length)
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
    ) -> ChronosBolt:
        """Load weights once, using device for inference independently of the input device."""
        pipeline_class = import_chronos_pipeline()
        pipeline = pipeline_class.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        pipeline.model.to(device)
        return cls(pipeline, model_name=model_name)

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
        """Return the native median (H,) or requested quantiles (H, Q)."""
        context_spec: InputSpec = validate_context(context)
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

        quantile_indices = validate_quantile_levels(
            quantile_levels,
            self._levels,
            model_name=self._model_name,
        )

        torch_module = import_torch()
        if context_spec.family == "numpy":
            native_values = numpy_native_context(cast(NDArray[Any], context))
            native_context = torch_module.as_tensor(native_values, dtype=torch_module.float32)
        else:
            native_context = torch_module.as_tensor(context, dtype=torch_module.float32)

        native_output = require_native_tensor(
            self._pipeline.predict(
                native_context,
                prediction_length=horizon,
                limit_prediction_length=True,
            ),
            model_name=self._model_name,
        )
        # Chronos-Bolt returns exactly (batch=1, loaded quantile grid, horizon).
        expected_shape = (1, len(self._levels), horizon)
        if tuple(native_output.shape) != expected_shape:
            raise RuntimeError(
                f"{self._model_name} native SDK returned shape {tuple(native_output.shape)}; "
                f"expected exactly {expected_shape}"
            )
        native_quantiles = native_output.detach()[0].transpose(0, 1)
        selected_forecast = (
            native_quantiles[:, self._median_index]
            if quantile_indices is None
            else native_quantiles[:, quantile_indices]
        )
        return cast(
            "NDArray[np.floating[Any]] | torch.Tensor",
            restore_output(selected_forecast, context_spec, model_name=self._model_name),
        )
