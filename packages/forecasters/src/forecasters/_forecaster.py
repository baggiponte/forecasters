"""The small structural interface shared by concrete forecasters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt
    import torch

    from forecasters._typing import ScalarT


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
        context: npt.NDArray[ScalarT],
        *,
        horizon: int,
        quantile_levels: None = None,
    ) -> np.ndarray[tuple[int], np.dtype[ScalarT]]: ...

    @overload
    def predict(
        self,
        context: npt.NDArray[ScalarT],
        *,
        horizon: int,
        quantile_levels: Sequence[float],
    ) -> np.ndarray[tuple[int, int], np.dtype[ScalarT]]: ...

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
        context: npt.NDArray[np.generic] | torch.Tensor,
        *,
        horizon: int,
        quantile_levels: Sequence[float] | None = None,
    ) -> npt.NDArray[np.generic] | torch.Tensor:
        """Forecast ``horizon`` steps after a one-dimensional floating context."""
        ...
