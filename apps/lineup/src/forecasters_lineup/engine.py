"""Shared local model catalog and native forecast helpers.

This module deliberately has no web-framework imports. The FastAPI demo and the
marimo AnyWidget both use it to validate requests, call the same native adapter
boundaries, and derive safe catalog metadata. Server scheduling and widget trait
transport remain in their respective hosts.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from forecasters import ChronosBolt, Forecaster, Timestral

MAX_CONTEXT_VALUES = 2_048
MAX_HORIZON = 1_024
MAX_MODELS_PER_REQUEST = 8
INTERVAL_LEVELS: tuple[float, float, float] = (0.1, 0.5, 0.9)


class IntervalPayload(TypedDict):
    """Native lower/upper quantiles for the demo's central 80% interval."""

    levels: list[float]
    lower: list[float]
    upper: list[float]


class RequestError(ValueError):
    """A client request that is invalid before any model is called."""


class ModelUnavailable(RuntimeError):
    """A selected model exists but cannot answer a forecast right now."""


@dataclass(frozen=True)
class DemoSettings:
    """Filesystem/loading settings for the real model factories."""

    cache_dir: Path | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Static catalog metadata plus a lazy native-adapter factory."""

    key: str
    name: str
    org: str
    variant: str
    color: str
    note: str
    url: str
    default_on: bool
    parameter_count: int
    factory: Callable[[], Forecaster]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def settings_from_env() -> DemoSettings:
    """Read only demo-specific overrides; Hugging Face otherwise uses its normal env."""
    cache_value = os.environ.get("FORECASTERS_DEMO_CACHE")
    cache_dir = Path(cache_value).expanduser() if cache_value else None
    return DemoSettings(
        cache_dir=cache_dir,
        local_files_only=_env_flag("FORECASTERS_DEMO_LOCAL_ONLY"),
    )


def default_model_specs(settings: DemoSettings | None = None) -> list[ModelSpec]:
    """Build the two-checkpoint catalog without importing either native SDK."""
    settings = settings or settings_from_env()

    def chronos() -> Forecaster:
        return ChronosBolt.from_pretrained(
            "amazon/chronos-bolt-tiny",
            cache_dir=settings.cache_dir,
            local_files_only=settings.local_files_only,
        )

    def timestral() -> Forecaster:
        return Timestral.from_pretrained(
            "mistralai/Timestral-5M",
            cache_dir=settings.cache_dir,
            local_files_only=settings.local_files_only,
        )

    return [
        ModelSpec(
            key="chronos_bolt",
            name="Chronos-Bolt",
            org="Amazon",
            variant="tiny",
            color="#e8590c",
            note="Chronos-Bolt tiny, native median on CPU.",
            url="https://github.com/amazon-science/chronos-forecasting",
            default_on=True,
            parameter_count=9_000_000,
            factory=chronos,
        ),
        ModelSpec(
            key="timestral",
            name="Timestral",
            org="Mistral AI",
            variant="5M",
            color="#0b7285",
            note="Timestral 5M, native median on CPU.",
            url="https://github.com/mistralai/timestral",
            default_on=True,
            parameter_count=5_000_000,
            factory=timestral,
        ),
    ]


def predict_native(
    adapter: Forecaster,
    context: np.ndarray,
    horizon: int,
    include_interval: bool,
) -> tuple[list[float], IntervalPayload | None]:
    """Call one native adapter and validate the shape/values used by both hosts."""
    if include_interval:
        # The adapter is the single authority on the loaded native grid; its ValueError
        # becomes a visible per-model error in each host.
        values = adapter.predict(
            context,
            horizon=horizon,
            quantile_levels=list(INTERVAL_LEVELS),
        )
        array = np.asarray(values)
        expected_shape = (horizon, len(INTERVAL_LEVELS))
        if array.shape != expected_shape:
            raise RuntimeError(
                f"adapter returned interval shape {array.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise RuntimeError(f"adapter returned non-floating interval dtype {array.dtype}")
        if not np.isfinite(array).all():
            raise RuntimeError("adapter returned a non-finite interval value")
        interval: IntervalPayload = {
            "levels": list(INTERVAL_LEVELS),
            "lower": [float(value) for value in array[:, 0]],
            "upper": [float(value) for value in array[:, 2]],
        }
        return [float(value) for value in array[:, 1]], interval

    values = adapter.predict(context, horizon=horizon)
    array = np.asarray(values)
    if array.shape != (horizon,):
        raise RuntimeError(f"adapter returned shape {array.shape}; expected ({horizon},)")
    if not np.issubdtype(array.dtype, np.floating):
        raise RuntimeError(f"adapter returned non-floating median dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError("adapter returned a non-finite median value")
    return [float(value) for value in array], None


def _parse_context(values: object) -> np.ndarray:
    if not isinstance(values, list):
        raise RequestError("y must be a one-dimensional array of finite numbers")
    if not values:
        raise RequestError("y must contain at least one value")
    if len(values) > MAX_CONTEXT_VALUES:
        raise RequestError(f"y has {len(values)} values; the demo limit is {MAX_CONTEXT_VALUES}")

    cleaned: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(f"y[{index}] must be a finite number")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise RequestError(f"y[{index}] must be a finite number") from None
        if not math.isfinite(number):
            raise RequestError(f"y[{index}] must be finite")
        cleaned.append(number)
    return np.asarray(cleaned, dtype=np.float64)


def _parse_horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("h must be a positive integer")
    if value <= 0:
        raise RequestError("h must be positive")
    if value > MAX_HORIZON:
        raise RequestError(f"h exceeds the demo limit of {MAX_HORIZON}")
    return value


def _parse_models(value: object, known_keys: Sequence[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RequestError("models must be a nonempty list of model keys")
    if len(value) > MAX_MODELS_PER_REQUEST:
        raise RequestError(f"select at most {MAX_MODELS_PER_REQUEST} models")
    selected: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RequestError("each model key must be a string")
        if item not in known_keys:
            raise RequestError(f"unknown model {item!r}")
        if item in selected:
            raise RequestError(f"model {item!r} was selected twice")
        selected.append(item)
    return selected


def _parse_include_interval(value: object) -> bool:
    if not isinstance(value, bool):
        raise RequestError("include_interval must be a boolean")
    return value


def parse_forecast_request(
    payload: object,
    known_keys: Iterable[str],
) -> tuple[np.ndarray, int, list[str], bool]:
    """Validate the same forecast boundary for HTTP, WebSocket and the widget."""
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    keys = list(known_keys)
    context = _parse_context(payload.get("y"))
    horizon = _parse_horizon(payload.get("h"))
    models = _parse_models(payload.get("models"), keys)
    include_interval = _parse_include_interval(payload.get("include_interval", False))
    return context, horizon, models, include_interval


def ready_model_info(
    specs: Iterable[ModelSpec],
    adapters: Mapping[str, Forecaster],
) -> list[dict[str, Any]]:
    """Return JSON-safe catalog metadata for adapters already loaded by a host."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        adapter = adapters.get(spec.key)
        if adapter is None:
            continue
        rows.append(
            {
                "key": spec.key,
                "name": spec.name,
                "org": spec.org,
                "variant": spec.variant,
                "color": spec.color,
                "note": spec.note,
                "url": spec.url,
                "default_on": spec.default_on,
                "parameter_count": spec.parameter_count,
                "status": "ready",
                "load_ms": 0.0,
                "error": None,
                "max_horizon": adapter.max_horizon,
                "max_context_length": adapter.max_context_length,
            }
        )
    return rows
