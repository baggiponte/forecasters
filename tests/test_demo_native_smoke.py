"""Opt-in HTTP/WebSocket smoke tests with the two cached real checkpoints.

Set ``FORECASTERS_NATIVE=1`` to run. The test uses ``.hf-cache`` with
``local_files_only=True`` and never downloads weights.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from examples.lineup.server import DemoSettings, create_app, default_model_specs

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("FORECASTERS_NATIVE") != "1",
        reason="set FORECASTERS_NATIVE=1 to run real-checkpoint demo smoke tests",
    ),
]

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".hf-cache"


def wait_ready(client: TestClient, key: str, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get("/api/models")
        assert response.status_code == 200
        last = next(row for row in response.json()["models"] if row["key"] == key)
        if last["status"] == "ready":
            return last
        if last["status"] == "error":
            pytest.fail(f"{key} failed to load: {last['error']}")
        time.sleep(0.1)
    pytest.fail(f"{key} did not become ready; last={last}")


def frozen_context() -> list[float]:
    values = 0.45 + 0.2 * np.sin(np.arange(48, dtype=np.float64) / 5.0)
    return [float(value) for value in values]


@pytest.mark.skipif(not CACHE_DIR.exists(), reason="downloaded checkpoint cache not found")
def test_real_checkpoints_over_http_and_websocket_reuse_loaded_adapters() -> None:
    settings = DemoSettings(cache_dir=CACHE_DIR, local_files_only=True)
    app = create_app(specs=default_model_specs(settings), autoload=False)
    context = frozen_context()
    horizon = 8

    with TestClient(app) as client:
        keys = ["chronos_bolt", "timestral"]
        for key in keys:
            response = client.post(f"/api/models/{key}/load")
            assert response.status_code in {200, 202}
        for key in keys:
            wait_ready(client, key)

        http_response = client.post(
            "/api/forecast",
            json={"y": context, "h": horizon, "models": keys},
        )
        assert http_response.status_code == 200
        http_results = http_response.json()["results"]

        manager = app.state.manager
        for key in keys:
            result = http_results[key]
            assert "error" not in result, result
            median = np.asarray(result["median"], dtype=np.float64)
            adapter = manager.get(key).adapter
            assert adapter is not None
            expected = adapter.predict(np.asarray(context, dtype=np.float64), horizon=horizon)
            np.testing.assert_array_equal(median, expected)

        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "forecast",
                    "seq": 1,
                    "y": context,
                    "h": horizon,
                    "models": keys,
                }
            )
            messages = [ws.receive_json(), ws.receive_json()]
            assert {message["key"] for message in messages} == set(keys)
            for message in messages:
                assert message["type"] == "forecast"
                median = np.asarray(message["median"], dtype=np.float64)
                np.testing.assert_array_equal(
                    median, np.asarray(http_results[message["key"]]["median"])
                )
