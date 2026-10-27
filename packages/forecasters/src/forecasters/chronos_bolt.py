"""Adapter for Amazon's Chronos-Bolt pipelines through the official Chronos SDK."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, final

from forecasters._base import BaseForecaster
from forecasters._deps import import_chronos_pipeline

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

    import torch

_DEFAULT_CHECKPOINT = "amazon/chronos-bolt-tiny"


@final
class ChronosBolt(BaseForecaster):
    """A loaded Chronos-Bolt checkpoint."""

    def __init__(self, native: Any, *, model_name: str) -> None:
        import torch

        self._pipeline = native
        super().__init__(
            model_name=model_name,
            supported_quantile_levels=native.quantiles,
            max_context_length=native.model_context_length,
            max_horizon=native.model_prediction_length,
            native_layout=("batch", "quantile", "horizon"),
            native_dtype=torch.float32,
        )

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

    @override
    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        return self._pipeline.predict(
            context,
            prediction_length=horizon,
            limit_prediction_length=True,
        )
