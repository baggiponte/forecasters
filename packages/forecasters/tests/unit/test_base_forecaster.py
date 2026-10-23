"""A third adapter gets the contract without copying either built-in adapter."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from forecasters import BaseForecaster

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.backend


class ExampleForecaster(BaseForecaster):
    def __init__(
        self,
        native,
        *,
        native_layout=("horizon", "quantile"),
        quantile_levels=(0.2, 0.5, 0.8),
        native_dtype=None,
    ) -> None:
        self.native = native
        super().__init__(
            model_name="example/checkpoint",
            supported_quantile_levels=quantile_levels,
            max_context_length=8,
            max_horizon=4,
            native_layout=native_layout,
            native_dtype=native_dtype,
        )

    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        return self.native(context, horizon=horizon)


@pytest.mark.parametrize("layout", ["canonical", "permuted"])
@pytest.mark.parametrize("family", ["numpy", "torch"])
def test_new_adapter_inherits_selection_and_representation(layout: str, family: str) -> None:
    values = torch.tensor(
        [[1.0, 11.0, 101.0], [2.0, 12.0, 102.0], [3.0, 13.0, 103.0], [4.0, 14.0, 104.0]]
    )
    if layout == "canonical":
        native = Mock(side_effect=lambda context, *, horizon: values[:horizon])
        model = ExampleForecaster(native)
    else:
        native = Mock(side_effect=lambda context, *, horizon: values[:horizon].T[:, None, :, None])
        model = ExampleForecaster(native, native_layout=("quantile", "batch", "horizon", "variate"))
    if family == "numpy":
        context = np.array([2.0, 5.0, 9.0], dtype=np.float64)[::-1]
        context.setflags(write=False)
    else:
        context = torch.tensor([9.0, 5.0, 2.0], dtype=torch.bfloat16)

    median = model.predict(context, horizon=2)
    quantiles = model.predict(context, horizon=2, quantile_levels=[0.8, 0.2])
    expected_quantiles = [[101.0, 1.0], [102.0, 2.0]]
    if family == "numpy":
        assert isinstance(median, np.ndarray) and isinstance(quantiles, np.ndarray)
        assert median.dtype == quantiles.dtype == np.float64
        np.testing.assert_array_equal(median, [11.0, 12.0])
        np.testing.assert_array_equal(quantiles, expected_quantiles)
        np.testing.assert_array_equal(context, [9.0, 5.0, 2.0])
        assert not context.flags.writeable
    else:
        torch.testing.assert_close(median, torch.tensor([11.0, 12.0]).bfloat16())
        torch.testing.assert_close(quantiles, torch.tensor(expected_quantiles).bfloat16())


@pytest.mark.parametrize(
    ("layout", "levels", "output", "expected"),
    [
        (("horizon", "quantile"), (0.5,), torch.tensor([[7.0]]), 7.0),
        (
            ("quantile", "batch", "horizon", "variate"),
            (0.2, 0.5, 0.8),
            torch.tensor([10.0, 20.0, 30.0]).reshape(3, 1, 1, 1),
            20.0,
        ),
    ],
)
def test_single_horizon_and_single_quantile_keep_public_shapes(layout, levels, output, expected):
    model = ExampleForecaster(
        Mock(return_value=output), native_layout=layout, quantile_levels=levels
    )
    context = np.ones(2, dtype=np.float32)
    median = model.predict(context, horizon=1)
    explicit = model.predict(context, horizon=1, quantile_levels=[0.5])
    assert median.shape == (1,)
    assert explicit.shape == (1, 1)
    np.testing.assert_array_equal(median, [expected])
    np.testing.assert_array_equal(explicit, [[expected]])


@pytest.mark.parametrize(
    ("context", "horizon", "levels", "error"),
    [
        ([1.0, 2.0], 1, None, TypeError),
        (np.ones(2), True, None, TypeError),
        (np.ones(9), 1, None, ValueError),
        (np.ones(2), 5, None, ValueError),
        (np.ones(2), 1, [0.33], ValueError),
        (np.ones(2), 1, [0.2, float(np.float32(0.2))], ValueError),
    ],
)
def test_new_adapter_rejects_invalid_requests_before_sdk(context, horizon, levels, error) -> None:
    native = Mock()
    model = ExampleForecaster(native)
    with pytest.raises(error):
        model.predict(context, horizon=horizon, quantile_levels=levels)
    native.assert_not_called()


@pytest.mark.parametrize("output", [np.ones((4, 3)), torch.ones(3, 4), torch.ones(1, 4, 3)])
def test_new_adapter_cannot_silently_reshape_malformed_output(output) -> None:
    model = ExampleForecaster(Mock(return_value=output))
    with pytest.raises(RuntimeError, match="example/checkpoint.*expected"):
        model.predict(np.ones(2), horizon=4)


def test_declared_singleton_axes_must_really_be_singletons() -> None:
    model = ExampleForecaster(
        Mock(return_value=torch.ones(3, 2, 4, 1)),
        native_layout=("quantile", "batch", "horizon", "variate"),
    )
    with pytest.raises(RuntimeError, match="expected exactly"):
        model.predict(np.ones(2), horizon=4)


@pytest.mark.parametrize(
    "layout", [("horizon",), ("horizon", "quantile", "quantile"), ("horizon", "quantile", "typo")]
)
def test_invalid_layout_is_an_authoring_error(layout) -> None:
    with pytest.raises(ValueError, match="native_layout"):
        ExampleForecaster(Mock(), native_layout=layout)


def test_native_dtype_cannot_silently_truncate_the_context() -> None:
    with pytest.raises(ValueError, match="native_dtype must be a floating Torch dtype"):
        ExampleForecaster(Mock(), native_dtype=torch.int64)


def test_native_bfloat16_quantiles_restore_numpy_representation() -> None:
    model = ExampleForecaster(
        Mock(return_value=torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.bfloat16)),
        native_dtype=torch.bfloat16,
    )
    context = np.ones(2, dtype=np.float64)
    result = model.predict(context, horizon=1, quantile_levels=[0.8, 0.2])
    assert isinstance(result, np.ndarray) and result.dtype == context.dtype
    np.testing.assert_array_equal(result, [[3.5, 1.5]])
