from __future__ import annotations

import pytest

from forecasters._validation import validate_context

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.backend


@pytest.mark.parametrize(
    ("context", "error"),
    [
        (torch.tensor([1, 2], dtype=torch.int64), TypeError),
        (torch.tensor([True, False]), TypeError),
        (torch.tensor([1.0, float("nan")]), ValueError),
        (torch.empty(0), ValueError),
        (torch.ones(2, 2), ValueError),
    ],
)
def test_rejects_invalid_tensor_values_and_shapes(context, error) -> None:
    with pytest.raises(error):
        validate_context(context)
