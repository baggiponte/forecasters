from __future__ import annotations

import pytest

from forecasters._validation import validate_context

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.backend


@pytest.mark.parametrize(
    ("context", "error"),
    [
        (torch.empty(0), ValueError),
        (torch.ones(2, 2), ValueError),
    ],
)
def test_rejects_invalid_tensor_shapes(context, error) -> None:
    with pytest.raises(error):
        validate_context(context)
