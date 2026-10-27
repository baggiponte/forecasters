"""Adapter for Mistral AI's Timestral checkpoints through the official SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, final

from forecasters._base import BaseForecaster
from forecasters._deps import import_timestral_forecaster

if TYPE_CHECKING:
    from pathlib import Path

    import torch

_DEFAULT_CHECKPOINT = "mistralai/Timestral-5M"


@final
class Timestral(BaseForecaster):
    """A loaded Timestral checkpoint."""

    def __init__(self, native: Any, *, model_name: str) -> None:
        self._forecaster = native
        super().__init__(
            model_name=model_name,
            supported_quantile_levels=native.quantile_levels,
            max_context_length=native.max_context,
            max_horizon=int(native.per_shot_horizon_patches) * int(native.patch_size),
            native_layout=("batch", "variate", "horizon", "quantile"),
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
    ) -> Timestral:
        """Load weights once, using device for inference independently of the input device."""
        forecaster_class = import_timestral_forecaster()
        native = forecaster_class.from_pretrained(
            model_name,
            device=device,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        return cls(native, model_name=model_name)

    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        forecast = self._forecaster.forecast(context, prediction_length=horizon)
        return getattr(forecast, "quantiles", None)
