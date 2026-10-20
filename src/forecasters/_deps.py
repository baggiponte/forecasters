"""Lazy optional-dependency imports with actionable installation guidance."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _install_hint(extra: str) -> str:
    return (
        f"Install the optional {extra!r} dependencies with "
        f"`pip install 'forecasters[{extra}]'` (or install the native SDK directly)."
    )


def import_torch() -> ModuleType:
    """Import PyTorch only when an adapter actually needs it."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyTorch is required for model inference. Install it directly with "
            "`pip install torch`, or install a model extra with "
            "`pip install 'forecasters[chronos]'` or "
            "`pip install 'forecasters[timestral]'`."
        ) from exc
    return torch


def import_chronos_pipeline() -> Any:
    """Import the official Chronos-Bolt pipeline lazily."""
    try:
        from chronos import ChronosBoltPipeline
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return import_torch()
        raise ImportError(
            "Chronos-Bolt inference requires the official `chronos-forecasting` package. "
            + _install_hint("chronos")
        ) from exc
    return ChronosBoltPipeline


def import_timestral_forecaster() -> Any:
    """Import the official Timestral forecaster lazily."""
    try:
        from timestral import TimestralForecaster
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return import_torch()
        raise ImportError(
            "Timestral inference requires the official `timestral` package from "
            "https://github.com/mistralai/timestral. " + _install_hint("timestral")
        ) from exc
    return TimestralForecaster
