from __future__ import annotations

import numpy as np
import pytest

from forecasters._outputs import numpy_native_context, require_native_tensor, restore_output
from forecasters._validation import InputSpec

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.backend


def test_negative_stride_context_can_cross_the_torch_bridge() -> None:
    context = np.array([1.0, 2.0, 3.0])[::-1]
    native = torch.as_tensor(numpy_native_context(context))
    torch.testing.assert_close(native, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))


def test_readonly_context_gets_a_writable_copy() -> None:
    context = np.ones(3)
    context.setflags(write=False)
    native = numpy_native_context(context)
    assert native.flags.writeable
    assert not np.shares_memory(native, context)


@pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8, reason="longdouble is float64 here")
def test_extended_precision_context_is_narrowed_for_torch() -> None:
    native = numpy_native_context(np.array([1.0, 2.0], dtype=np.longdouble))
    assert native.dtype == np.float64
    np.testing.assert_array_equal(native, [1.0, 2.0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_restores_numpy_dtype(dtype) -> None:
    spec = InputSpec(family="numpy", dtype=np.dtype(dtype), device=None)
    restored = restore_output(torch.tensor([1.5, 2.5]), spec, model_name="test/checkpoint")
    assert restored.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(restored, [1.5, 2.5])


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_restores_torch_dtype(dtype) -> None:
    spec = InputSpec(family="torch", dtype=dtype, device=torch.device("cpu"))
    restored = restore_output(torch.tensor([1.5, 2.5]), spec, model_name="test/checkpoint")
    torch.testing.assert_close(restored, torch.tensor([1.5, 2.5], dtype=dtype))


def test_restores_accelerator_device_when_available() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("no accelerator available")
    context = torch.ones(1, device=device)
    spec = InputSpec(family="torch", dtype=context.dtype, device=context.device)
    restored = restore_output(torch.tensor([1.0]), spec, model_name="test/checkpoint")
    torch.testing.assert_close(restored, context)


def test_rejects_nonfinite_selected_values() -> None:
    spec = InputSpec(family="numpy", dtype=np.dtype(np.float64), device=None)
    with pytest.raises(RuntimeError, match="test/checkpoint.*non-finite selected"):
        restore_output(torch.tensor([torch.nan]), spec, model_name="test/checkpoint")


@pytest.mark.parametrize(
    "spec",
    [
        InputSpec(family="numpy", dtype=np.dtype(np.float16), device=None),
        InputSpec(family="torch", dtype=torch.float16, device=torch.device("cpu")),
    ],
    ids=["numpy", "torch"],
)
def test_rejects_overflow_in_destination_dtype(spec: InputSpec) -> None:
    with pytest.raises(RuntimeError, match="not representable"):
        restore_output(torch.tensor([70000.0]), spec, model_name="test/checkpoint")


def test_native_output_must_be_a_tensor() -> None:
    with pytest.raises(RuntimeError, match="test/checkpoint.*expected a torch.Tensor"):
        require_native_tensor(np.ones(3), model_name="test/checkpoint")
