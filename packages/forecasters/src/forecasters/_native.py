"""Native quantile-axis normalization and selection.

Pyrefly checks the shape expressions that ty ignores below. Keep shape imports and
assertions under TYPE_CHECKING so NumPy-only users need neither Torch nor shape_extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing import assert_type

    import torch
    from shape_extensions import IntVar

    Q = IntVar("Q")
    H = IntVar("H")
else:
    Q = TypeVar("Q")
    H = TypeVar("H")


def chronos_quantiles(
    raw: torch.Tensor[[1, Q, H]],  # ty: ignore[invalid-type-form]
    *,
    median_index: int,
    indices: list[int] | None,
) -> torch.Tensor[[H]] | torch.Tensor[[H, int]]:  # ty: ignore[invalid-type-form]
    """Normalize Chronos ``(1, Q, H)`` and select the requested quantiles."""
    full = raw.detach()[0].transpose(0, 1)
    if TYPE_CHECKING:
        assert_type(full, torch.Tensor[[H, Q]])  # ty: ignore[not-subscriptable]
    return full[:, median_index] if indices is None else full[:, indices]


def timestral_quantiles(
    raw: torch.Tensor[[1, 1, H, Q]],  # ty: ignore[invalid-type-form]
    *,
    median_index: int,
    indices: list[int] | None,
) -> torch.Tensor[[H]] | torch.Tensor[[H, int]]:  # ty: ignore[invalid-type-form]
    """Normalize Timestral ``(1, 1, H, Q)`` and select the requested quantiles."""
    full = raw.detach()[0, 0]
    if TYPE_CHECKING:
        assert_type(full, torch.Tensor[[H, Q]])  # ty: ignore[not-subscriptable]
    return full[:, median_index] if indices is None else full[:, indices]
