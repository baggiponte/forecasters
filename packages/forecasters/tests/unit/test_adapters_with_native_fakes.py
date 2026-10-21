"""Unit tests at the native SDK boundary, using Torch-backed fakes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # optional test backend; package import remains NumPy-only
    torch = None

from forecasters import ChronosBolt, Timestral

pytestmark = [
    pytest.mark.backend,
    pytest.mark.skipif(torch is None, reason="native-boundary fakes require optional PyTorch"),
]

CHRONOS_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TIMESTRAL_LEVELS = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)
CHRONOS_NAME = "fake/chronos-bolt-tiny"
TIMESTRAL_NAME = "fake/Timestral-5M"


def native_value(level_index: int, step: int, last_context: float) -> float:
    return float(level_index * 1000 + step + 7.0 * last_context)


@dataclass
class FakeChronosPipeline:
    quantiles: list[float] = field(default_factory=lambda: list(CHRONOS_LEVELS))
    model_context_length: int = 32
    model_prediction_length: int = 8
    calls: list[np.ndarray] = field(default_factory=list)
    batch_size: int = 1
    horizon_delta: int = 0
    extra_quantiles: int = 0
    nan_level_index: int | None = None
    override_value: tuple[int, float] | None = None

    def predict(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        assert limit_prediction_length is True
        self.calls.append(inputs.detach().cpu().to(torch.float32).numpy().copy())
        last = float(inputs.reshape(-1)[-1])
        qcount = len(self.quantiles) + self.extra_quantiles
        hcount = prediction_length + self.horizon_delta
        rows: list[list[float]] = []
        for index in range(qcount):
            row = [native_value(index, step, last) for step in range(hcount)]
            if self.nan_level_index == index:
                row = [math.nan] * hcount
            if self.override_value is not None and self.override_value[0] == index:
                row = [self.override_value[1]] * hcount
            rows.append(row)
        quantiles = torch.tensor(rows, dtype=torch.float32).reshape(1, qcount, hcount)
        return quantiles.repeat(self.batch_size, 1, 1)


@dataclass
class FakeTimestralForecaster:
    quantile_levels: tuple[float, ...] = TIMESTRAL_LEVELS
    max_context: int = 40
    per_shot_horizon_patches: int = 4
    patch_size: int = 4
    calls: list[np.ndarray] = field(default_factory=list)
    batch_size: int = 1
    variate_size: int = 1
    horizon_delta: int = 0
    extra_quantiles: int = 0
    nan_level_index: int | None = None
    override_value: tuple[int, float] | None = None

    def forecast(self, context: torch.Tensor, prediction_length: int) -> Any:
        self.calls.append(context.detach().cpu().to(torch.float32).numpy().copy())
        last = float(context[-1])
        qcount = len(self.quantile_levels) + self.extra_quantiles
        hcount = prediction_length + self.horizon_delta
        rows: list[list[float]] = []
        for index in range(qcount):
            row = [native_value(index, step, last) for step in range(hcount)]
            if self.nan_level_index == index:
                row = [math.nan] * hcount
            if self.override_value is not None and self.override_value[0] == index:
                row = [self.override_value[1]] * hcount
            rows.append(row)
        quantiles = torch.tensor(rows, dtype=torch.float32).T.reshape(1, 1, hcount, qcount)
        quantiles = quantiles.repeat(self.batch_size, self.variate_size, 1, 1)
        return SimpleNamespace(quantiles=quantiles, quantile_levels=self.quantile_levels)


@dataclass
class Harness:
    adapter: ChronosBolt | Timestral
    native: FakeChronosPipeline | FakeTimestralForecaster
    levels: tuple[float, ...]
    model_name: str


def make_chronos() -> Harness:
    native = FakeChronosPipeline()
    return Harness(
        ChronosBolt(native, model_name=CHRONOS_NAME),
        native,
        tuple(native.quantiles),
        CHRONOS_NAME,
    )


def make_timestral() -> Harness:
    native = FakeTimestralForecaster()
    return Harness(
        Timestral(native, model_name=TIMESTRAL_NAME),
        native,
        native.quantile_levels,
        TIMESTRAL_NAME,
    )


FACTORIES = [
    pytest.param(make_chronos, id="chronos-bolt"),
    pytest.param(make_timestral, id="timestral"),
]


def expected(
    harness: Harness, index: int, horizon: int, last_context: float, dtype: Any
) -> np.ndarray:
    return np.array(
        [native_value(index, step, last_context) for step in range(horizon)], dtype=dtype
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_adapter_exposes_loaded_checkpoint_metadata(factory: Any) -> None:
    harness = factory()
    assert harness.adapter.supported_quantile_levels == tuple(harness.levels)
    if isinstance(harness.native, FakeChronosPipeline):
        assert harness.adapter.max_context_length == harness.native.model_context_length
        assert harness.adapter.max_horizon == harness.native.model_prediction_length
    else:
        assert harness.adapter.max_context_length == harness.native.max_context
        assert harness.adapter.max_horizon == (
            harness.native.per_shot_horizon_patches * harness.native.patch_size
        )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_family_dtype_shapes_and_quantile_order(factory: Any, dtype: Any) -> None:
    harness = factory()
    context = np.linspace(-2.0, 3.0, 12, dtype=dtype)
    horizon = 6

    median = harness.adapter.predict(context, horizon=horizon)
    assert isinstance(median, np.ndarray)
    assert median.shape == (horizon,)
    assert median.dtype == context.dtype
    median_index = harness.levels.index(0.5)
    np.testing.assert_array_equal(
        median, expected(harness, median_index, horizon, float(context[-1]), context.dtype)
    )

    requested = [0.9, 0.1, 0.5]
    quantiles = harness.adapter.predict(context, horizon=horizon, quantile_levels=requested)
    assert isinstance(quantiles, np.ndarray)
    assert quantiles.shape == (horizon, 3)
    assert quantiles.dtype == context.dtype
    for column, level in enumerate(requested):
        np.testing.assert_array_equal(
            quantiles[:, column],
            expected(
                harness, harness.levels.index(level), horizon, float(context[-1]), context.dtype
            ),
        )


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_numpy_negative_stride_views_and_edge_dtypes(factory: Any, dtype: Any) -> None:
    harness = factory()
    base = np.linspace(-2.0, 3.0, 12, dtype=dtype)
    context = base[::-1]
    before = context.copy()
    assert context.strides[0] < 0

    out = harness.adapter.predict(context, horizon=4)
    assert out.dtype == context.dtype
    np.testing.assert_array_equal(context, before)
    median_index = harness.levels.index(0.5)
    np.testing.assert_array_equal(
        out, expected(harness, median_index, 4, float(context[-1]), context.dtype)
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_numpy_readonly_context_is_supported_without_mutation(factory: Any) -> None:
    harness = factory()
    context = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    context.setflags(write=False)
    before = context.copy()

    out = harness.adapter.predict(context, horizon=3)
    assert out.dtype == context.dtype
    assert not context.flags.writeable
    np.testing.assert_array_equal(context, before)


@pytest.mark.parametrize("factory", FACTORIES)
def test_default_median_matches_explicit_half_and_single_level_shape(factory: Any) -> None:
    harness = factory()
    context = np.arange(10, dtype=np.float64)
    default = harness.adapter.predict(context, horizon=5)
    explicit = harness.adapter.predict(context, horizon=5, quantile_levels=[0.5])
    assert explicit.shape == (5, 1)
    np.testing.assert_array_equal(explicit[:, 0], default)
    assert len(harness.native.calls) == 2


@pytest.mark.parametrize("factory", FACTORIES)
def test_torch_dtype_and_cpu_device_preservation(factory: Any) -> None:
    harness = factory()
    context = torch.linspace(0.0, 1.0, 10, dtype=torch.float64, device="cpu")
    out = harness.adapter.predict(context, horizon=4, quantile_levels=[0.5, 0.9])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 2)
    assert out.dtype == torch.float64
    assert out.device == context.device


@pytest.mark.parametrize("factory", FACTORIES)
def test_torch_bfloat16_dtype_preservation(factory: Any) -> None:
    harness = factory()
    context = torch.linspace(0.0, 1.0, 10, dtype=torch.bfloat16)
    out = harness.adapter.predict(context, horizon=3)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.bfloat16
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("factory", FACTORIES)
def test_torch_accelerator_device_preserved_when_available(factory: Any) -> None:
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    if device is None:
        pytest.skip("no accelerator available; CPU device preservation is tested separately")

    harness = factory()
    context = torch.linspace(0.0, 1.0, 10, dtype=torch.float32, device=device)
    out = harness.adapter.predict(context, horizon=3)
    assert isinstance(out, torch.Tensor)
    assert out.device == context.device


@pytest.mark.parametrize("factory", FACTORIES)
def test_native_horizon_shorter_than_requested_is_rejected(factory: Any) -> None:
    harness = factory()
    harness.native.horizon_delta = -1
    with pytest.raises(RuntimeError, match="native SDK returned shape") as excinfo:
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=5)
    assert harness.model_name in str(excinfo.value)
    assert "expected exactly" in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_native_level_grid_shorter_than_loaded_is_rejected(factory: Any) -> None:
    harness = factory()
    harness.native.extra_quantiles = -1
    with pytest.raises(RuntimeError, match="native SDK returned shape") as excinfo:
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)
    assert harness.model_name in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_extra_native_quantile_columns_are_rejected(factory: Any) -> None:
    harness = factory()
    harness.native.extra_quantiles = 1
    with pytest.raises(RuntimeError, match="expected exactly") as excinfo:
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)
    assert harness.model_name in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_extra_native_horizon_columns_are_rejected(factory: Any) -> None:
    harness = factory()
    harness.native.horizon_delta = 1
    with pytest.raises(RuntimeError, match="expected exactly") as excinfo:
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)
    assert harness.model_name in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_correct_shape_unrequested_quantile_nan_is_tolerated(factory: Any) -> None:
    harness = factory()
    harness.native.nan_level_index = harness.levels.index(0.1)
    context = np.linspace(0.0, 1.0, 8, dtype=np.float32)

    median = harness.adapter.predict(context, horizon=3)
    assert median.shape == (3,)
    assert np.isfinite(median).all()

    quantiles = harness.adapter.predict(context, horizon=3, quantile_levels=[0.5, 0.9])
    assert quantiles.shape == (3, 2)
    assert np.isfinite(quantiles).all()


def test_chronos_extra_batch_dimension_is_rejected() -> None:
    harness = make_chronos()
    harness.native.batch_size = 2
    with pytest.raises(RuntimeError, match=r"expected exactly \(1,"):
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)


@pytest.mark.parametrize("field_name", ["batch_size", "variate_size"])
def test_timestral_extra_batch_or_variate_dimension_is_rejected(field_name: str) -> None:
    harness = make_timestral()
    setattr(harness.native, field_name, 2)
    with pytest.raises(RuntimeError, match=r"expected exactly \(1, 1,"):
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)


@pytest.mark.parametrize("factory", FACTORIES)
def test_nonfinite_selected_native_values_raise_runtime_error(factory: Any) -> None:
    harness = factory()
    harness.native.nan_level_index = harness.levels.index(0.5)
    with pytest.raises(RuntimeError, match="non-finite selected forecast") as excinfo:
        harness.adapter.predict(np.ones(8, dtype=np.float32), horizon=3)
    assert harness.model_name in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("family", ["numpy", "torch"])
def test_float16_restoration_overflow_raises_runtime_error(factory: Any, family: str) -> None:
    harness = factory()
    harness.native.override_value = (harness.levels.index(0.5), 70000.0)
    if family == "numpy":
        context: Any = np.ones(8, dtype=np.float16)
    else:
        context = torch.ones(8, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="not representable") as excinfo:
        harness.adapter.predict(context, horizon=3)
    assert harness.model_name in str(excinfo.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_unavailable_quantile_prevents_native_call(factory: Any) -> None:
    harness = factory()
    with pytest.raises(ValueError) as excinfo:
        harness.adapter.predict(
            np.ones(8, dtype=np.float32), horizon=2, quantile_levels=[0.1, 0.33, 0.5]
        )
    message = str(excinfo.value)
    assert "0.33" in message
    assert "supported native levels" in message
    assert harness.model_name in message
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
def test_duplicate_equivalent_levels_prevent_native_call(factory: Any) -> None:
    harness = factory()
    with pytest.raises(ValueError, match="unique"):
        harness.adapter.predict(
            np.ones(8, dtype=np.float32),
            horizon=2,
            quantile_levels=[0.1, float(np.float32(0.1))],
        )
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"context": np.array([np.nan], dtype=np.float32)}, ValueError),
        ({"horizon": 0}, ValueError),
        ({"quantile_levels": [True]}, TypeError),
    ],
)
def test_invalid_requests_prevent_native_call(
    factory: Any, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    harness = factory()
    request = {"context": np.ones(8, dtype=np.float32), "horizon": 2, **kwargs}
    with pytest.raises(error):
        harness.adapter.predict(**request)
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
def test_invalid_torch_contexts_prevent_native_call(factory: Any) -> None:
    harness = factory()
    cases = [
        (torch.tensor([1, 2, 3], dtype=torch.int64), TypeError),
        (torch.tensor([True, False]), TypeError),
        (torch.tensor([1.0, float("nan")]), ValueError),
    ]
    for context, error in cases:
        with pytest.raises(error):
            harness.adapter.predict(context, horizon=2)
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
def test_empty_torch_context_is_rejected_before_native_call(factory: Any) -> None:
    harness = factory()
    with pytest.raises(ValueError, match="at least one observation"):
        harness.adapter.predict(torch.empty(0, dtype=torch.float32), horizon=2)
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
def test_context_and_horizon_limits_prevent_native_call(factory: Any) -> None:
    harness = factory()
    overlong = np.ones(harness.adapter.max_context_length + 1, dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        harness.adapter.predict(overlong, horizon=1)
    assert str(harness.adapter.max_context_length) in str(excinfo.value)
    assert harness.model_name in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        harness.adapter.predict(
            np.ones(8, dtype=np.float32), horizon=harness.adapter.max_horizon + 1
        )
    assert str(harness.adapter.max_horizon) in str(excinfo.value)
    assert harness.model_name in str(excinfo.value)
    assert harness.native.calls == []


@pytest.mark.parametrize("factory", FACTORIES)
def test_repeated_calls_do_not_leak_context(factory: Any) -> None:
    harness = factory()
    first = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    second = np.linspace(100.0, 101.0, 8, dtype=np.float64)

    first_a = harness.adapter.predict(first, horizon=3)
    _ = harness.adapter.predict(second, horizon=3)
    first_b = harness.adapter.predict(first, horizon=3)

    np.testing.assert_array_equal(first_a, first_b)
    assert len(harness.native.calls) == 3
    np.testing.assert_array_equal(harness.native.calls[0][-1], first[-1])
    np.testing.assert_array_equal(harness.native.calls[1][-1], second[-1])
    np.testing.assert_array_equal(harness.native.calls[2][-1], first[-1])


@pytest.mark.parametrize("adapter_class", [ChronosBolt, Timestral])
def test_from_pretrained_loads_and_reuses_native_model(
    adapter_class: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    is_chronos = adapter_class is ChronosBolt
    native = FakeChronosPipeline() if is_chronos else FakeTimestralForecaster()
    load_calls: list[tuple[str, dict[str, Any]]] = []
    devices: list[str] = []
    if is_chronos:
        native.model = SimpleNamespace(to=devices.append)

    class Loader:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> Any:
            load_calls.append((model_name, kwargs))
            return native

    module = "chronos_bolt" if is_chronos else "timestral"
    importer = "import_chronos_pipeline" if is_chronos else "import_timestral_forecaster"
    monkeypatch.setattr(f"forecasters.{module}.{importer}", lambda: Loader)
    adapter = adapter_class.from_pretrained(
        "custom/checkpoint",
        device="cuda:1",
        revision="checkpoint-revision",
        cache_dir=tmp_path,
        local_files_only=True,
    )
    expected_kwargs = {
        "revision": "checkpoint-revision",
        "cache_dir": str(tmp_path),
        "local_files_only": True,
    }
    if not is_chronos:
        expected_kwargs["device"] = "cuda:1"
    assert load_calls == [("custom/checkpoint", expected_kwargs)]
    if is_chronos:
        assert devices == ["cuda:1"]

    for last in (1.0, 2.0):
        result = adapter.predict(np.full(8, last, dtype=np.float32), horizon=2)
        index = adapter.supported_quantile_levels.index(0.5)
        np.testing.assert_array_equal(
            result, [native_value(index, step, last) for step in range(2)]
        )
    assert len(load_calls) == 1
    assert len(native.calls) == 2
