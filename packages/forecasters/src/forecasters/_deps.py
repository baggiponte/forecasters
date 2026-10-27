"""Lazy optional-dependency imports with actionable installation guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

_TIMESTRAL_INSTALL = (
    "pip install 'timestral @ git+https://github.com/mistralai/timestral.git"
    "@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'"
)


def _install_hint(extra: str) -> str:
    return (
        f"Install the optional {extra!r} dependencies with "
        f"`pip install 'forecasters[{extra}]'` (or install the native SDK directly)."
    )


def import_torch() -> ModuleType:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyTorch is required for model inference. Install it directly with "
            "`pip install torch`, or install the Chronos-Bolt extra with "
            "`pip install 'forecasters[chronos]'`. Timestral is not published on PyPI; "
            f"install its official SDK with `{_TIMESTRAL_INSTALL}`.",
        ) from exc
    return torch


def import_chronos_pipeline() -> Any:
    try:
        from chronos import ChronosBoltPipeline
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return import_torch()
        raise ImportError(
            "Chronos-Bolt inference requires the official `chronos-forecasting` package. "
            + _install_hint("chronos"),
        ) from exc
    return ChronosBoltPipeline


def import_timestral_forecaster() -> Any:
    try:
        from timestral import TimestralForecaster
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return import_torch()
        raise ImportError(
            "Timestral inference requires the official `timestral` package from "
            "https://github.com/mistralai/timestral. Timestral is not published on PyPI; "
            f"install its official Git revision with `{_TIMESTRAL_INSTALL}`.",
        ) from exc
    return TimestralForecaster
