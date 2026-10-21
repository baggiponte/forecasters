"""Native quantile-axis normalization and selection shared by the adapters.

Pyrefly checks the shape annotations in this module against the real executable body.
They use legacy ``IntVar`` symbols imported only under ``TYPE_CHECKING`` and postponed
annotations, so the runtime package has no shape-extensions dependency and importing
``forecasters`` still works with NumPy alone.

ty does not understand Pyrefly's shape syntax. The narrowly scoped ignores on the
annotated lines keep its checks meaningful for the rest of the module. These annotations
are not intended for runtime introspection; use the public ``predict`` signatures for that.
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
