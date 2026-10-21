from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from forecasters import ChronosBolt, Timestral

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.backend


class ChronosSDK:
    quantiles = (0.1, 0.5, 0.9)
    model_context_length = 32
    model_prediction_length = 8

    def __init__(self) -> None:
        self.output = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]])
        self.predict = Mock(side_effect=lambda *args, **kwargs: self.output)
        self.model = Mock()


class TimestralSDK:
    quantile_levels = (0.1, 0.5, 0.9)
    max_context = 40
    per_shot_horizon_patches = 4
    patch_size = 4

    def __init__(self) -> None:
        self.output = torch.tensor([[[[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]]])
        self.forecast = Mock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(quantiles=self.output)
        )


@pytest.fixture(params=[ChronosSDK, TimestralSDK], ids=["chronos-bolt", "timestral"])
def native(request: pytest.FixtureRequest) -> ChronosSDK | TimestralSDK:
    return request.param()


@pytest.fixture
def adapter(native: ChronosSDK | TimestralSDK) -> ChronosBolt | Timestral:
    if isinstance(native, ChronosSDK):
        return ChronosBolt(native, model_name="test/checkpoint")
    return Timestral(native, model_name="test/checkpoint")


@pytest.mark.parametrize("family", ["numpy", "torch"])
def test_selects_native_median_and_requested_columns(adapter, native, family: str) -> None:
    context = np.array([1.0, 2.0, 3.0], dtype=np.float64)[::-1]
    if family == "torch":
        context = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)

    median = adapter.predict(context, horizon=3)
    quantiles = adapter.predict(context, horizon=3, quantile_levels=[0.9, 0.1, 0.5])
    singleton = adapter.predict(context, horizon=3, quantile_levels=[0.5])
    expected = [[100.0, 1.0, 10.0], [200.0, 2.0, 20.0], [300.0, 3.0, 30.0]]
    if family == "numpy":
        assert median.dtype == np.float64
        np.testing.assert_array_equal(median, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(quantiles, expected)
        np.testing.assert_array_equal(singleton, [[10.0], [20.0], [30.0]])
    else:
        torch.testing.assert_close(median, torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
        torch.testing.assert_close(quantiles, torch.tensor(expected, dtype=torch.float64))
        torch.testing.assert_close(
            singleton, torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)
        )

    call = native.predict.call_args if isinstance(native, ChronosSDK) else native.forecast.call_args
    expected_dtype = torch.float32 if isinstance(native, ChronosSDK) else torch.float64
    torch.testing.assert_close(call.args[0], torch.tensor([3.0, 2.0, 1.0], dtype=expected_dtype))
    expected_kwargs = {"prediction_length": 3}
    if isinstance(native, ChronosSDK):
        expected_kwargs["limit_prediction_length"] = True
    assert call.kwargs == expected_kwargs


@pytest.mark.parametrize("failure", ["oversized-horizon", "layout-drift"])
def test_rejects_malformed_native_shape(adapter, native, failure: str) -> None:
    if isinstance(native, ChronosSDK):
        shape = (1, 3, 3) if failure == "oversized-horizon" else (1, 2, 3)
    else:
        shape = (1, 1, 3, 3) if failure == "oversized-horizon" else (1, 2, 3)
    native.output = torch.ones(shape)
    with pytest.raises(RuntimeError, match="test/checkpoint.*expected exactly"):
        adapter.predict(np.ones(3, dtype=np.float32), horizon=2)


def test_validates_only_selected_native_values(adapter, native) -> None:
    if isinstance(native, ChronosSDK):
        native.output[:, 0, :] = torch.nan
    else:
        native.output[:, :, :, 0] = torch.nan
    context = np.ones(3, dtype=np.float32)
    np.testing.assert_array_equal(adapter.predict(context, horizon=3), [10.0, 20.0, 30.0])
    with pytest.raises(RuntimeError, match="non-finite selected forecast"):
        adapter.predict(context, horizon=3, quantile_levels=[0.1])


def test_checkpoint_metadata_controls_request_limits(adapter, native) -> None:
    if isinstance(native, ChronosSDK):
        assert (adapter.max_context_length, adapter.max_horizon) == (32, 8)
        native.output = torch.ones(1, 3, 8)
        native_call = native.predict
    else:
        assert (adapter.max_context_length, adapter.max_horizon) == (40, 16)
        native.output = torch.ones(1, 1, 16, 3)
        native_call = native.forecast
    assert adapter.supported_quantile_levels == (0.1, 0.5, 0.9)
    adapter.predict(np.ones(adapter.max_context_length), horizon=adapter.max_horizon)
    native_call.reset_mock()

    with pytest.raises(ValueError, match="test/checkpoint"):
        adapter.predict(np.ones(adapter.max_context_length + 1), horizon=1)
    with pytest.raises(ValueError, match="test/checkpoint"):
        adapter.predict(np.ones(3), horizon=adapter.max_horizon + 1)
    with pytest.raises(ValueError, match="supported native levels"):
        adapter.predict(np.ones(3), horizon=3, quantile_levels=[0.33])
    native_call.assert_not_called()


def test_from_pretrained_routes_loader_arguments(
    adapter, native, monkeypatch, tmp_path: Path
) -> None:
    chronos = isinstance(native, ChronosSDK)
    module = "chronos_bolt" if chronos else "timestral"
    importer = "import_chronos_pipeline" if chronos else "import_timestral_forecaster"
    load = Mock(return_value=native)
    monkeypatch.setattr(
        f"forecasters.{module}.{importer}", lambda: SimpleNamespace(from_pretrained=load)
    )
    type(adapter).from_pretrained(
        "custom/checkpoint",
        device="cuda:1",
        revision="revision",
        cache_dir=tmp_path,
        local_files_only=True,
    )
    kwargs = {"revision": "revision", "cache_dir": str(tmp_path), "local_files_only": True}
    if chronos:
        native.model.to.assert_called_once_with("cuda:1")
    else:
        kwargs["device"] = "cuda:1"
    load.assert_called_once_with("custom/checkpoint", **kwargs)
