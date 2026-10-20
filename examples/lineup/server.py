"""Minimal local demo server for the Line-up canvas.

The server owns only the browser transport required by ``examples/lineup/static``:
the model catalog and load status, a serialized CPU inference call, and a WebSocket
that coalesces superseded drawing requests. The forecast payload is the explicit
median vector; quantiles, bands and scoring are not part of this demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from forecasters import ChronosBolt, Timestral
from forecasters._forecaster import Forecaster

LOGGER = logging.getLogger("forecasters.lineup")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MAX_BODY_BYTES = 256 * 1024
MAX_CONTEXT_VALUES = 2_048
MAX_HORIZON = 1_024
MAX_MODELS_PER_REQUEST = 8
MAX_WEBSOCKET_TEXT = MAX_BODY_BYTES


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
    factory: Callable[[], Forecaster]


@dataclass
class ModelRuntime:
    spec: ModelSpec
    status: str = "idle"
    adapter: Forecaster | None = None
    error: str | None = None
    load_ms: float | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


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
            factory=timestral,
        ),
    ]


class ModelManager:
    """Owns model status, one lazy load per model, and serialized inference."""

    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        self._models = {spec.key: ModelRuntime(spec) for spec in specs}
        self._inference_lock = asyncio.Lock()

    def get(self, key: str) -> ModelRuntime | None:
        return self._models.get(key)

    def keys(self) -> list[str]:
        return list(self._models)

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for runtime in self._models.values():
            spec = runtime.spec
            adapter = runtime.adapter
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
                    "status": runtime.status,
                    "load_ms": runtime.load_ms,
                    "error": runtime.error,
                    "max_horizon": adapter.max_horizon if adapter is not None else None,
                    "max_context_length": (
                        adapter.max_context_length if adapter is not None else None
                    ),
                }
            )
        return rows

    def start_load(self, key: str) -> ModelRuntime:
        runtime = self._models.get(key)
        if runtime is None:
            raise KeyError(key)
        if runtime.status in {"loading", "ready"}:
            return runtime

        runtime.status = "loading"
        runtime.error = None
        runtime.load_ms = None
        runtime.adapter = None
        runtime.task = asyncio.create_task(self._load(key), name=f"load-{key}")
        return runtime

    async def _load(self, key: str) -> None:
        runtime = self._models[key]
        started = time.perf_counter()
        try:
            # Native loading is blocking, including any cache read, so keep it off-loop.
            adapter = await asyncio.to_thread(runtime.spec.factory)
            runtime.adapter = adapter
            runtime.status = "ready"
            runtime.error = None
        except Exception as exc:  # visible per-model failure; other models keep working
            runtime.adapter = None
            runtime.status = "error"
            runtime.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Loading %s failed", key)
        finally:
            runtime.load_ms = (time.perf_counter() - started) * 1000.0

    async def wait_until_settled(self, key: str) -> ModelRuntime:
        runtime = self._models[key]
        task = runtime.task
        if task is not None and not task.done():
            await task
        return runtime

    async def predict(self, key: str, context: np.ndarray, horizon: int) -> tuple[list[float], float]:
        runtime = self._models.get(key)
        if runtime is None:
            raise ModelUnavailable(f"unknown model {key!r}")
        if runtime.status != "ready" or runtime.adapter is None:
            detail = runtime.error or f"model is {runtime.status}"
            raise ModelUnavailable(f"{key} is not ready: {detail}")
        if horizon > runtime.adapter.max_horizon:
            raise ModelUnavailable(
                f"{key} supports at most {runtime.adapter.max_horizon} steps; "
                f"requested {horizon} (no truncation)"
            )

        started = time.perf_counter()
        async with self._inference_lock:
            # The caller may disappear (for example on a WebSocket disconnect), but the
            # native thread cannot be stopped. Keep the inference lock until that thread
            # has actually finished so a second request cannot overlap native compute.
            native = asyncio.create_task(
                asyncio.to_thread(_predict_median, runtime.adapter, context, horizon)
            )
            cancelled = False
            while True:
                try:
                    median = await asyncio.shield(native)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if native.done():
                        if not native.cancelled():
                            native.exception()
                        raise
                    continue
            if cancelled:
                raise asyncio.CancelledError
        elapsed = (time.perf_counter() - started) * 1000.0
        return median, elapsed

    async def shutdown(self) -> None:
        tasks = [runtime.task for runtime in self._models.values() if runtime.task]
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            if task.cancelled():
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def _predict_median(adapter: Forecaster, context: np.ndarray, horizon: int) -> list[float]:
    values = adapter.predict(context, horizon=horizon)
    array = np.asarray(values)
    if array.shape != (horizon,):
        raise RuntimeError(f"adapter returned shape {array.shape}; expected ({horizon},)")
    if not np.issubdtype(array.dtype, np.floating):
        raise RuntimeError(f"adapter returned non-floating median dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError("adapter returned a non-finite median value")
    return [float(value) for value in array]


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


def _parse_models(value: object, manager: ModelManager) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RequestError("models must be a nonempty list of model keys")
    if len(value) > MAX_MODELS_PER_REQUEST:
        raise RequestError(f"select at most {MAX_MODELS_PER_REQUEST} models")
    selected: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RequestError("each model key must be a string")
        if item not in manager.keys():
            raise RequestError(f"unknown model {item!r}")
        if item in selected:
            raise RequestError(f"model {item!r} was selected twice")
        selected.append(item)
    return selected


def parse_forecast_request(payload: object, manager: ModelManager) -> tuple[np.ndarray, int, list[str]]:
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    context = _parse_context(payload.get("y"))
    horizon = _parse_horizon(payload.get("h"))
    models = _parse_models(payload.get("models"), manager)
    return context, horizon, models


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, ModelUnavailable):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


async def _forecast_one(
    manager: ModelManager, key: str, context: np.ndarray, horizon: int
) -> dict[str, Any]:
    try:
        median, elapsed_ms = await manager.predict(key, context, horizon)
        return {"median": median, "ms": elapsed_ms}
    except Exception as exc:
        LOGGER.info("Forecast for %s failed: %s", key, exc)
        return {"error": _error_text(exc)}


def create_app(
    *,
    specs: Iterable[ModelSpec] | None = None,
    autoload: bool = True,
) -> FastAPI:
    """Build the demo app; tests inject fake specs and can disable startup loading."""
    manager = ModelManager(specs if specs is not None else default_model_specs())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if autoload:
            for runtime in manager._models.values():
                if runtime.spec.default_on:
                    manager.start_load(runtime.spec.key)
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title="forecasters Line-up demo", lifespan=lifespan)
    app.state.manager = manager

    @app.middleware("http")
    async def limit_request_body(request: Request, call_next: Callable[[Request], Any]):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_BODY_BYTES:
                    return JSONResponse(
                        {"detail": f"request body exceeds {MAX_BODY_BYTES} bytes"},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        return {"models": manager.snapshot()}

    @app.post("/api/models/{key}/load")
    async def load_model(key: str) -> JSONResponse:
        runtime = manager.get(key)
        if runtime is None:
            raise HTTPException(status_code=404, detail=f"unknown model {key!r}")
        runtime = manager.start_load(key)
        status_code = 202 if runtime.status == "loading" else 200
        return JSONResponse(
            {"key": key, "status": runtime.status}, status_code=status_code
        )

    @app.post("/api/forecast")
    async def forecast(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
        try:
            context, horizon, keys = parse_forecast_request(payload, manager)
        except RequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results: dict[str, Any] = {}
        for key in keys:
            results[key] = await _forecast_one(manager, key, context, horizon)
        return {"results": results}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        pending: tuple[int, np.ndarray, int, list[str]] | None = None
        wake = asyncio.Event()

        async def worker() -> None:
            nonlocal pending
            while True:
                await wake.wait()
                wake.clear()
                request = pending
                pending = None
                if request is None:
                    continue
                seq, context, horizon, keys = request
                for key in keys:
                    # A newer drawing supersedes the rest of an in-flight request.
                    if wake.is_set():
                        break
                    result = await _forecast_one(manager, key, context, horizon)
                    message = {"type": "forecast", "key": key, "seq": seq, **result}
                    if "error" in result:
                        message["type"] = "error"
                    try:
                        await ws.send_json(message)
                    except (WebSocketDisconnect, RuntimeError):
                        return

        worker_task = asyncio.create_task(worker())
        try:
            while True:
                text = await ws.receive_text()
                if len(text) > MAX_WEBSOCKET_TEXT:
                    await ws.send_json(
                        {"type": "error", "error": f"message exceeds {MAX_WEBSOCKET_TEXT} bytes"}
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    await ws.send_json({"type": "error", "error": f"invalid JSON: {exc}"})
                    continue

                seq_value = payload.get("seq") if isinstance(payload, dict) else None
                seq = seq_value if isinstance(seq_value, int) and not isinstance(seq_value, bool) else None
                if payload.get("type") != "forecast" if isinstance(payload, dict) else True:
                    await ws.send_json({"type": "error", "seq": seq, "error": "expected a forecast message"})
                    continue
                try:
                    context, horizon, keys = parse_forecast_request(payload, manager)
                except RequestError as exc:
                    await ws.send_json({"type": "error", "seq": seq, "error": str(exc)})
                    continue
                if seq is None or seq <= 0:
                    await ws.send_json({"type": "error", "error": "seq must be a positive integer"})
                    continue

                pending = (seq, context, horizon, keys)
                wake.set()
        except WebSocketDisconnect:
            pass
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    return app


app = create_app()
