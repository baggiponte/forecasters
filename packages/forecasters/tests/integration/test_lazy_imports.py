"""Package import must not eagerly require either native SDK or PyTorch."""

from __future__ import annotations

import subprocess
import sys


def test_import_forecasters_is_lazy() -> None:
    code = (
        "import sys, forecasters; "
        "assert forecasters.ChronosBolt and forecasters.Timestral and forecasters.Forecaster; "
        "forbidden = {'torch', 'chronos', 'timestral', 'transformers'}; "
        "loaded = forbidden & set(sys.modules); "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
