"""Purpose-built AnyWidget host for the shared Line-up renderer.

The widget owns JSON-safe model metadata, public AnyWidget request/reply custom
messages and a result snapshot trait for Python inspection. Native adapters stay
private Python attributes loaded once by the notebook; they never cross the
trait boundary.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Iterable, Mapping
from importlib import resources
from typing import Any

import anywidget
import traitlets

from forecasters import Forecaster
from forecasters_lineup.engine import (
    ModelSpec,
    RequestError,
    default_model_specs,
    parse_forecast_request,
    predict_native,
    ready_model_info,
)

LOGGER = logging.getLogger("forecasters.lineup.marimo")


def _shared_esm() -> str:
    """Compose the one shared renderer with a small AFM entry.

    anywidget evaluates ``_esm`` from a Blob URL, so a relative
    ``import "./lineup.js"`` cannot resolve against the package directory.
    Keeping the shared source as the single implementation and composing it in
    Python avoids copying the renderer or adding a bundler.
    """
    shared = (
        resources.files("forecasters_lineup")
        .joinpath("static", "lineup.js")
        .read_text(encoding="utf-8")
    )
    marker = "export function mountLineup"
    if marker not in shared:
        raise RuntimeError("shared Line-up renderer is missing its export")
    shared = shared.replace(marker, "function mountLineup", 1)
    entry = (
        resources.files(__package__)
        .joinpath("static", "widget-entry.js")
        .read_text(encoding="utf-8")
    )
    return f"{shared}\n{entry}"


def _widget_css() -> str:
    """Reuse the standalone stylesheet, with local font files inlined as data URLs."""
    css = (
        resources.files("forecasters_lineup")
        .joinpath("static", "lineup.css")
        .read_text(encoding="utf-8")
    )
    for filename in ("Archivo-100_900.woff2", "IBMPlexSans-var.woff2"):
        font = (
            resources.files("forecasters_lineup").joinpath("static", "fonts", filename).read_bytes()
        )
        encoded = base64.b64encode(font).decode("ascii")
        css = css.replace(
            f'url("/static/fonts/{filename}")',
            f'url("data:font/woff2;base64,{encoded}")',
        )
    return css


class LineupWidget(anywidget.AnyWidget):
    """Line-up canvas backed by native adapters loaded in the notebook kernel."""

    _esm = _shared_esm()
    _css = _widget_css()

    model_info: Any = traitlets.List([]).tag(sync=True)
    result = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        models: Mapping[str, Forecaster],
        *,
        specs: Iterable[ModelSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        self._adapters = dict(models)
        self._specs = list(specs) if specs is not None else default_model_specs()
        info = ready_model_info(self._specs, self._adapters)
        super().__init__(model_info=info, **kwargs)
        self.on_msg(self._on_frontend_message)

    def _emit(self, result: dict[str, Any]) -> None:
        self.result = result
        # A custom message is the supported public AnyWidget channel that reaches
        # every view immediately; marimo can defer trait-state broadcasts until a
        # later cell execution. The result trait remains available for Python.
        self.send({"type": "lineup_result", "result": result})

    def _on_frontend_message(
        self,
        widget: Any,
        content: Any,
        buffers: Any,
    ) -> None:
        del widget, buffers
        if not isinstance(content, dict) or content.get("type") != "lineup_request":
            return
        self._handle_request_payload(content.get("request"))

    def _handle_request_payload(self, payload: Any) -> None:
        request_id = (
            payload.get("request_id", payload.get("seq")) if isinstance(payload, dict) else None
        )
        try:
            context, horizon, keys, include_interval = parse_forecast_request(
                payload, self._adapters.keys()
            )
        except RequestError as exc:
            self._emit(
                {
                    "request_id": request_id,
                    "results": {},
                    "error": str(exc),
                }
            )
            return

        results: dict[str, dict[str, Any]] = {}
        for key in keys:
            adapter = self._adapters[key]
            started = time.perf_counter()
            try:
                if horizon > adapter.max_horizon:
                    raise RequestError(
                        f"{key} supports at most {adapter.max_horizon} steps; "
                        f"requested {horizon} (no truncation)"
                    )
                median, interval = predict_native(adapter, context, horizon, include_interval)
                item: dict[str, Any] = {
                    "median": median,
                    "ms": (time.perf_counter() - started) * 1000.0,
                }
                if interval is not None:
                    item["interval"] = interval
                results[key] = item
            except Exception as exc:
                LOGGER.exception("Forecast for %s failed", key)
                results[key] = {"error": f"{type(exc).__name__}: {exc}"}

        self._emit({"request_id": request_id, "results": results})
