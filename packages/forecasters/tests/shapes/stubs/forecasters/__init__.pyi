"""Pyrefly-only public API overlay; keep signatures aligned with the runtime classes."""

from pathlib import Path
from typing import Any, Protocol

from forecasters._predict import ShapedPredict

class ChronosBolt(ShapedPredict):
    @property
    def supported_quantile_levels(self) -> tuple[float, ...]: ...
    @property
    def max_context_length(self) -> int: ...
    @property
    def max_horizon(self) -> int: ...
    def __init__(self, native: Any, *, model_name: str) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ...,
        *,
        device: str = ...,
        revision: str | None = ...,
        cache_dir: str | Path | None = ...,
        local_files_only: bool = ...,
    ) -> ChronosBolt: ...

class Timestral(ShapedPredict):
    @property
    def supported_quantile_levels(self) -> tuple[float, ...]: ...
    @property
    def max_context_length(self) -> int: ...
    @property
    def max_horizon(self) -> int: ...
    def __init__(self, native: Any, *, model_name: str) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ...,
        *,
        device: str = ...,
        revision: str | None = ...,
        cache_dir: str | Path | None = ...,
        local_files_only: bool = ...,
    ) -> Timestral: ...

class Forecaster(ShapedPredict, Protocol):
    @property
    def supported_quantile_levels(self) -> tuple[float, ...]: ...
    @property
    def max_context_length(self) -> int: ...
    @property
    def max_horizon(self) -> int: ...
