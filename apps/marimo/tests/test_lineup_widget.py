"""AnyWidget request/result mapping tests with fake native adapters.

These tests do not open marimo or load a checkpoint. They exercise the real
notebook boundary: strict request validation, adapter reuse, per-model error
isolation, native interval passthrough and JSON-safe catalog metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("anywidget")
from forecasters._forecaster import Forecaster
from forecasters_lineup.engine import ModelSpec
from forecasters_lineup_marimo.widget import LineupWidget


@dataclass
class FakeAdapter:
    max_horizon: int = 32
    max_context_length: int = 128
    supported: tuple[float, ...] = (0.1, 0.5, 0.9)
    fail: bool = False
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
        if self.fail:
            raise RuntimeError("native inference failed")
        if requested is not None:
            unavailable = [level for level in requested if level not in self.supported]
            if unavailable:
                raise ValueError(
                    f"does not support the requested quantile levels {unavailable!r}; "
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
            else:
                columns.append(median + spread * 0.02)
        return np.column_stack(columns)


def spec(key: str) -> ModelSpec:
    return ModelSpec(
        key=key,
        name=key.title(),
        org="test",
        variant="fake",
        color="#123456",
        note="fake adapter",
        url="https://example.invalid",
        default_on=True,
        parameter_count=1_000,
        factory=lambda: (_ for _ in ()).throw(AssertionError("factory must not run")),
    )


def widget_with(adapters: dict[str, FakeAdapter]) -> LineupWidget:
    return LineupWidget(
        cast(Mapping[str, Forecaster], adapters),
        specs=[spec(key) for key in adapters],
    )


def send_request(widget: LineupWidget, payload: dict[str, Any]) -> None:
    # Dispatch through the inherited ipywidgets path, exactly as the comm does.
    widget._handle_msg(
        {
            "content": {
                "data": {
                    "method": "custom",
                    "content": {"type": "lineup_request", "request": payload},
                }
            },
            "buffers": [],
        }
    )


def test_widget_maps_request_to_one_result_and_reuses_adapters() -> None:
    chronos = FakeAdapter()
    timestral = FakeAdapter()
    widget = widget_with({"chronos": chronos, "timestral": timestral})

    payload = {
        "seq": 7,
        "y": [0.1, 0.2],
        "h": 3,
        "models": ["chronos", "timestral"],
        "include_interval": False,
    }
    send_request(widget, payload)
    result = widget.result

    assert result["request_id"] == 7
    assert set(result["results"]) == {"chronos", "timestral"}
    for key in ("chronos", "timestral"):
        assert result["results"][key]["median"] == pytest.approx([0.2, 0.21, 0.22])
        assert "interval" not in result["results"][key]
    assert len(chronos.calls) == 1 and len(timestral.calls) == 1
    assert chronos.quantile_requests == [None]

    send_request(widget, {**payload, "seq": 8, "y": [0.5], "h": 2})
    assert widget.result["request_id"] == 8
    assert len(chronos.calls) == 2 and len(timestral.calls) == 2


def test_widget_keeps_other_model_alive_when_one_native_call_fails() -> None:
    bad = FakeAdapter(fail=True)
    good = FakeAdapter()
    widget = widget_with({"bad": bad, "good": good})

    send_request(
        widget,
        {
            "seq": 1,
            "y": [0.1],
            "h": 2,
            "models": ["bad", "good"],
            "include_interval": False,
        },
    )
    results = widget.result["results"]
    assert "native inference failed" in results["bad"]["error"]
    assert results["good"]["median"] == pytest.approx([0.1, 0.11])
    assert len(bad.calls) == 1 and len(good.calls) == 1


def test_widget_strict_validation_does_not_call_native_inference() -> None:
    adapter = FakeAdapter()
    widget = widget_with({"fake": adapter})

    for payload in (
        {"seq": 1, "y": [0.1, True], "h": 1, "models": ["fake"]},
        {"seq": 2, "y": [0.1], "h": 0, "models": ["fake"]},
        {"seq": 3, "y": [0.1], "h": 1, "models": ["unknown"]},
        {"seq": 4, "y": [0.1], "h": 1, "models": ["fake"], "include_interval": 1},
    ):
        send_request(widget, payload)
        assert widget.result["request_id"] == payload["seq"]
        assert widget.result["error"]
        assert widget.result["results"] == {}
    assert adapter.calls == []


def test_widget_reports_native_horizon_limit_without_calling_adapter() -> None:
    adapter = FakeAdapter(max_horizon=3)
    widget = widget_with({"fake": adapter})
    send_request(
        widget,
        {
            "seq": 1,
            "y": [0.1],
            "h": 4,
            "models": ["fake"],
            "include_interval": False,
        },
    )
    assert "at most 3" in widget.result["results"]["fake"]["error"]
    assert adapter.calls == []


def test_widget_passes_native_interval_and_surfaces_unsupported_levels() -> None:
    adapter = FakeAdapter()
    widget = widget_with({"fake": adapter})
    send_request(
        widget,
        {
            "seq": 1,
            "y": [0.1],
            "h": 2,
            "models": ["fake"],
            "include_interval": True,
        },
    )
    interval = widget.result["results"]["fake"]["interval"]
    assert interval["levels"] == [0.1, 0.5, 0.9]
    assert interval["lower"] == pytest.approx([0.09, 0.09])
    assert interval["upper"] == pytest.approx([0.12, 0.15])

    unsupported = FakeAdapter(supported=(0.5,))
    widget = widget_with({"fake": unsupported})
    send_request(
        widget,
        {
            "seq": 2,
            "y": [0.1],
            "h": 2,
            "models": ["fake"],
            "include_interval": True,
        },
    )
    result = widget.result["results"]["fake"]
    assert "does not support the requested quantile levels [0.1, 0.9]" in result["error"]
    assert "interval" not in result


def test_widget_keeps_opaque_request_ids_distinct_across_views() -> None:
    adapter = FakeAdapter()
    widget = widget_with({"fake": adapter})
    for request_id in ("view-a-1", "view-b-1"):
        send_request(
            widget,
            {
                "seq": 1,
                "request_id": request_id,
                "y": [0.1],
                "h": 2,
                "models": ["fake"],
                "include_interval": False,
            },
        )
        assert widget.result["request_id"] == request_id
    assert len(adapter.calls) == 2


def test_widget_model_info_is_json_safe_and_excludes_adapters() -> None:
    adapter = FakeAdapter()
    widget = widget_with({"fake": adapter})
    assert json.loads(json.dumps(widget.model_info)) == widget.model_info
    row = widget.model_info[0]
    assert row["status"] == "ready"
    assert row["max_horizon"] == adapter.max_horizon
    assert not any(isinstance(value, FakeAdapter) for value in row.values())
