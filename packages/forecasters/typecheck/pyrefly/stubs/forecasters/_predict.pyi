"""Pyrefly-only predict signatures; these stubs are not shipped in the wheel.

Q stays gradual because Sequence does not encode its length. Torch stubs track
shape but not dtype or device; runtime validation still covers those properties.
"""

from collections.abc import Sequence
from typing import Any, Protocol, overload

import numpy as np
import torch
from shape_extensions import Int, IntVar

class ShapedPredict(Protocol):
    # NumPy: the caller's floating dtype is preserved, and the context length does not
    # appear in the result. Horizon and quantile axes are symbolic.
    @overload
    def predict[C: IntVar, H: IntVar, D: np.floating[Any]](
        self,
        context: np.ndarray[[C], np.dtype[D]],
        *,
        horizon: Int[H],
        quantile_levels: None = None,
    ) -> np.ndarray[[H], np.dtype[D]]: ...
    @overload
    def predict[C: IntVar, H: IntVar, Q: IntVar, D: np.floating[Any]](
        self,
        context: np.ndarray[[C], np.dtype[D]],
        *,
        horizon: Int[H],
        quantile_levels: Sequence[float],
    ) -> np.ndarray[[H, Q], np.dtype[D]]: ...

    # Torch: the Pyrefly Torch stubs currently track shape but not dtype, so the backend
    # and axes are proven while the element dtype remains gradual.
    @overload
    def predict[C: IntVar, H: IntVar](
        self,
        context: torch.Tensor[[C]],
        *,
        horizon: Int[H],
        quantile_levels: None = None,
    ) -> torch.Tensor[[H]]: ...
    @overload
    def predict[C: IntVar, H: IntVar, Q: IntVar](
        self,
        context: torch.Tensor[[C]],
        *,
        horizon: Int[H],
        quantile_levels: Sequence[float],
    ) -> torch.Tensor[[H, Q]]: ...
