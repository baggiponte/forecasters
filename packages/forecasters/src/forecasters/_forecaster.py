"""The small structural interface shared by concrete forecasters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from numpy.typing import NDArray

_FloatDType = TypeVar("_FloatDType", bound=np.floating[Any])


class Forecaster(Protocol):
    """A loaded, univariate zero-shot forecaster."""

    @property
    def supported_quantile_levels(self) -> tuple[float, ...]:
        """Native quantile probabilities available from the loaded checkpoint."""
        ...

    @property
    def max_context_length(self) -> int:
        """Maximum accepted context length for the loaded checkpoint."""
        ...

    @property
    def max_horizon(self) -> int:
        """Maximum accepted horizon before native rollout/truncation would be needed."""
        ...

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
        """Forecast ``horizon`` steps after a one-dimensional floating context."""
        ...
