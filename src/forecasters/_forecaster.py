"""The small structural interface shared by concrete forecasters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

_FloatDType = TypeVar("_FloatDType", bound=np.floating[Any])


class Forecaster(Protocol):
    """A loaded, univariate zero-shot forecaster.

    Concrete implementations own a native SDK object. The protocol deliberately has no
    training, registry, capability hierarchy or generic keyword-forwarding surface.
    """

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
