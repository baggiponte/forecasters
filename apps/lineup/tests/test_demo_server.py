"""FastAPI demo transport tests using fake adapters.

These tests do not load a native checkpoint. They cover the HTTP/WebSocket contract,
lazy loading, retry, per-model isolation, median and interval transport, and request
coalescing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from forecasters_lineup.server import ModelManager, ModelSpec, create_app


@dataclass
class FakeAdapter:
    max_horizon: int = 8
    max_context_length: int = 64
    supported: tuple[float, ...] = (0.1, 0.5, 0.9)
    calls: list[tuple[np.ndarray, int]] = field(default_factory=list)
    quantile_requests: list[tuple[float, ...] | None] = field(default_factory=list)

    @property
    def supported_quantile_levels(self) -> tuple[float, ...]:
        return self.supported

    def predict(
        self,
        context: np.ndarray,
        *,
        horizon: int,
        quantile_levels: Any = None,
    ) -> np.ndarray:
        values = np.asarray(context, dtype=np.float64)
        self.calls.append((values.copy(), horizon))
        requested = None if quantile_levels is None else tuple(quantile_levels)
        self.quantile_requests.append(requested)
        if requested is not None:
            unavailable = [level for level in requested if level not in self.supported]
            if unavailable:
                raise ValueError(
                    f"fake does not support the requested quantile levels {unavailable!r}; "
                    f"supported native levels are {self.supported!r}"
                )
        median = values[-1] + np.arange(horizon, dtype=np.float64) * 0.01
        if requested is None:
            return median
        spread = np.arange(1, horizon + 1, dtype=np.float64)
        columns = []
        for level in requested:
            if level == 0.1:
                columns.append(median - spread * 0.01)
            elif level == 0.5:
                columns.append(median)
            elif level == 0.9:
                columns.append(median + spread * 0.02)
            else:
                raise AssertionError(f"unexpected quantile level {level!r}")
        return np.column_stack(columns)


class Factory:
    def __init__(self, adapter: FakeAdapter | None = None, failures: int = 0) -> None:
        self.adapter = adapter or FakeAdapter()
        self.failures = failures
        self.calls = 0

    def __call__(self) -> FakeAdapter:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("checkpoint unavailable")
        return self.adapter


def spec(
    key: str, factory: Factory, *, default_on: bool = False, parameter_count: int = 9_000_000
) -> ModelSpec:
    return ModelSpec(
        key=key,
        name=key.title(),
        org="test",
        variant="fake",
        color="#123456",
        note="fake adapter",
        url="https://example.invalid",
        default_on=default_on,
        parameter_count=parameter_count,
        factory=cast(Any, factory),
    )


def wait_status(client: TestClient, key: str, status: str, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get("/api/models")
        assert response.status_code == 200
        last = next(row for row in response.json()["models"] if row["key"] == key)
        if last["status"] == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"{key} did not reach {status}; last={last}")


def load_and_wait(client: TestClient, key: str, status: str = "ready") -> dict[str, Any]:
    response = client.post(f"/api/models/{key}/load")
    assert response.status_code in {200, 202}
    return wait_status(client, key, status)


def test_lazy_load_reuses_adapter_for_http_forecasts() -> None:
    factory = Factory()
    app = create_app(specs=[spec("fake", factory)], autoload=False)
    context = [0.125, 0.25, 0.123456789]

    with TestClient(app) as client:
        assert factory.calls == 0
        loaded = load_and_wait(client, "fake")
        assert loaded["parameter_count"] == 9_000_000
        assert factory.calls == 1

        first = client.post(
            "/api/forecast",
            json={"y": context, "h": 3, "models": ["fake"]},
        )
        second = client.post(
            "/api/forecast",
            json={"y": [0.5], "h": 2, "models": ["fake"]},
        )
        assert first.status_code == 200 and second.status_code == 200
        first_median = first.json()["results"]["fake"]["median"]
        assert first_median == pytest.approx([0.123456789, 0.133456789, 0.143456789])
        assert len(second.json()["results"]["fake"]["median"]) == 2

        adapter = factory.adapter
        assert factory.calls == 1
        assert len(adapter.calls) == 2
        np.testing.assert_array_equal(adapter.calls[0][0], np.asarray(context))


def test_http_and_websocket_interval_parity_and_median_only_off() -> None:
    factory = Factory()
    app = create_app(specs=[spec("fake", factory)], autoload=False)
    context = [0.1, 0.23456789]
    horizon = 4

    with TestClient(app) as client:
        load_and_wait(client, "fake")
        assert factory.adapter.quantile_requests == []

        response = client.post(
            "/api/forecast",
            json={
                "y": context,
                "h": horizon,
                "models": ["fake"],
                "include_interval": True,
            },
        )
        assert response.status_code == 200
        result = response.json()["results"]["fake"]
        median = np.asarray(result["median"])
        expected_median = context[-1] + np.arange(horizon) * 0.01
        spread = np.arange(1, horizon + 1)
        np.testing.assert_array_equal(median, expected_median)
        assert result["interval"]["levels"] == [0.1, 0.5, 0.9]
        np.testing.assert_array_equal(
            np.asarray(result["interval"]["lower"]), expected_median - spread * 0.01
        )
        np.testing.assert_array_equal(
            np.asarray(result["interval"]["upper"]), expected_median + spread * 0.02
        )
        assert factory.adapter.quantile_requests == [(0.1, 0.5, 0.9)]

        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "forecast",
                    "seq": 4,
                    "y": context,
                    "h": horizon,
                    "models": ["fake"],
                    "include_interval": True,
                }
            )
            message = ws.receive_json()
            assert message["type"] == "forecast"
            np.testing.assert_array_equal(np.asarray(message["median"]), median)
            assert message["interval"] == result["interval"]
        assert factory.adapter.quantile_requests == [(0.1, 0.5, 0.9)] * 2

        without_band = client.post(
            "/api/forecast",
            json={"y": context, "h": horizon, "models": ["fake"]},
        ).json()["results"]["fake"]
        assert "interval" not in without_band
        assert len(without_band["median"]) == horizon
        assert factory.adapter.quantile_requests[-1] is None


def test_interval_adapter_validation_error_reaches_the_model_error() -> None:
    factory = Factory(FakeAdapter(supported=(0.5, 0.9)))
    app = create_app(specs=[spec("fake", factory)], autoload=False)
    with TestClient(app) as client:
        load_and_wait(client, "fake")
        result = client.post(
            "/api/forecast",
            json={
                "y": [0.1],
                "h": 2,
                "models": ["fake"],
                "include_interval": True,
            },
        ).json()["results"]["fake"]
        assert "does not support the requested quantile levels [0.1]" in result["error"]
        assert "supported native levels are (0.5, 0.9)" in result["error"]
        assert factory.adapter.quantile_requests == [(0.1, 0.5, 0.9)]


def test_one_load_failure_is_isolated_and_retry_succeeds() -> None:
    bad_factory = Factory(failures=1)
    good_factory = Factory()
    app = create_app(specs=[spec("bad", bad_factory), spec("good", good_factory)], autoload=False)

    with TestClient(app) as client:
        load_and_wait(client, "bad", status="error")
        load_and_wait(client, "good")
        catalog = {row["key"]: row for row in client.get("/api/models").json()["models"]}
        assert "checkpoint unavailable" in catalog["bad"]["error"]
        assert catalog["good"]["status"] == "ready"

        response = client.post(
            "/api/forecast",
            json={"y": [0.2], "h": 2, "models": ["bad", "good"]},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert "error" in results["bad"]
        assert len(results["good"]["median"]) == 2

        load_and_wait(client, "bad")
        assert bad_factory.calls == 2
        assert good_factory.calls == 1
        retried = client.post(
            "/api/forecast",
            json={"y": [0.2], "h": 2, "models": ["bad", "good"]},
        ).json()["results"]
        assert "error" not in retried["bad"]
        assert len(retried["bad"]["median"]) == 2


def test_native_horizon_limit_is_reported_not_clipped() -> None:
    factory = Factory(FakeAdapter(max_horizon=3))
    app = create_app(specs=[spec("fake", factory)], autoload=False)
    with TestClient(app) as client:
        load_and_wait(client, "fake")
        response = client.post(
            "/api/forecast",
            json={"y": [0.1], "h": 4, "models": ["fake"]},
        )
        assert response.status_code == 200
        result = response.json()["results"]["fake"]
        assert "at most 3" in result["error"]
        assert factory.adapter.calls == []

        ok = client.post(
            "/api/forecast",
            json={"y": [0.1], "h": 3, "models": ["fake"]},
        ).json()["results"]["fake"]
        assert len(ok["median"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"y": "not-a-list", "h": 1, "models": ["fake"]},
        {"y": [], "h": 1, "models": ["fake"]},
        {"y": [0.1, True], "h": 1, "models": ["fake"]},
        {"y": [0.1, "0.2"], "h": 1, "models": ["fake"]},
        {"y": [[0.1]], "h": 1, "models": ["fake"]},
        {"y": [0.1], "h": True, "models": ["fake"]},
        {"y": [0.1], "h": 0, "models": ["fake"]},
        {"y": [0.1], "h": -1, "models": ["fake"]},
        {"y": [0.1], "h": 1.5, "models": ["fake"]},
        {"y": [0.1], "h": "1", "models": ["fake"]},
        {"y": [0.1], "h": 1_025, "models": ["fake"]},
        {"y": [0.1], "h": 1, "models": []},
        {"y": [0.1], "h": 1, "models": "fake"},
        {"y": [0.1], "h": 1, "models": ["unknown"]},
        {"y": [0.1], "h": 1, "models": ["fake", "fake"]},
        {"y": [0.1], "h": 1, "models": ["fake"], "include_interval": None},
        {"y": [0.1], "h": 1, "models": ["fake"], "include_interval": 1},
        {"y": [0.1], "h": 1, "models": ["fake"], "include_interval": "yes"},
        {"y": [0.1] * 2_049, "h": 1, "models": ["fake"]},
        {"y": [10**1_000], "h": 1, "models": ["fake"]},
    ],
)
def test_http_request_validation(payload: dict[str, Any]) -> None:
    app = create_app(specs=[spec("fake", Factory())], autoload=False)
    with TestClient(app) as client:
        response = client.post("/api/forecast", json=payload)
        assert response.status_code == 400
        assert "detail" in response.json()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_http_rejects_nonfinite_json_tokens(token: str) -> None:
    app = create_app(specs=[spec("fake", Factory())], autoload=False)
    body = f'{{"y":[0.1,{token}],"h":1,"models":["fake"]}}'.encode()
    with TestClient(app) as client:
        response = client.post(
            "/api/forecast", content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 400
        assert "finite" in response.json()["detail"]


def test_http_body_limit() -> None:
    app = create_app(specs=[spec("fake", Factory())], autoload=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/forecast",
            content=b"x" * (256 * 1024 + 1),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413


def test_websocket_forecast_errors_and_parity() -> None:
    factory = Factory()
    app = create_app(specs=[spec("fake", factory)], autoload=False)
    with TestClient(app) as client:
        load_and_wait(client, "fake")
        context = [0.1, 0.23456789]
        expected = factory.adapter.predict(np.asarray(context), horizon=4)

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "forecast", "seq": 7, "y": context, "h": 4, "models": ["fake"]})
            message = ws.receive_json()
            assert message["type"] == "forecast"
            assert message["key"] == "fake"
            assert message["seq"] == 7
            np.testing.assert_array_equal(np.asarray(message["median"]), expected)

            ws.send_json(
                {"type": "forecast", "seq": 8, "y": context, "h": 4, "models": ["unknown"]}
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["seq"] == 8
            assert "unknown model" in error["error"]

            ws.send_json({"type": "wrong", "seq": 9})
            assert ws.receive_json()["error"] == "expected a forecast message"


class BlockingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(
        self,
        context: np.ndarray,
        *,
        horizon: int,
        quantile_levels: Any = None,
    ) -> np.ndarray:
        self.calls.append((np.asarray(context, dtype=np.float64).copy(), horizon))
        self.quantile_requests.append(None if quantile_levels is None else tuple(quantile_levels))
        if len(self.calls) == 1:
            self.started.set()
            assert self.release.wait(timeout=5)
        return np.arange(horizon, dtype=np.float64)


def test_websocket_coalesces_to_latest_drawing() -> None:
    adapter = BlockingAdapter()
    app = create_app(specs=[spec("fake", Factory(adapter))], autoload=False)
    with TestClient(app) as client:
        load_and_wait(client, "fake")
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "forecast", "seq": 1, "y": [0.1], "h": 2, "models": ["fake"]})
            assert adapter.started.wait(timeout=2), "first prediction did not start"
            ws.send_json({"type": "forecast", "seq": 2, "y": [0.2], "h": 2, "models": ["fake"]})
            ws.send_json({"type": "forecast", "seq": 3, "y": [0.3], "h": 2, "models": ["fake"]})
            adapter.release.set()

            received = [ws.receive_json(), ws.receive_json()]
            assert [message["seq"] for message in received] == [1, 3]
            assert len(adapter.calls) == 2


def test_cancelled_caller_keeps_native_inference_serialized() -> None:
    adapter = BlockingAdapter()
    manager = ModelManager([spec("fake", Factory(adapter))])
    runtime = manager.get("fake")
    assert runtime is not None
    runtime.adapter = cast(Any, adapter)
    runtime.status = "ready"
    context = np.asarray([0.1], dtype=np.float64)

    async def scenario() -> None:
        first = asyncio.create_task(manager.predict("fake", context, 2))
        assert await asyncio.to_thread(adapter.started.wait, 2)
        first.cancel()
        await asyncio.sleep(0.05)
        second = asyncio.create_task(manager.predict("fake", context, 2))
        await asyncio.sleep(0.05)
        try:
            assert len(adapter.calls) == 1, "second call entered native inference early"
        finally:
            adapter.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        median, interval, _ = await second
        assert interval is None
        assert len(adapter.calls) == 2
        np.testing.assert_array_equal(np.asarray(median), np.arange(2, dtype=np.float64))

    asyncio.run(scenario())
