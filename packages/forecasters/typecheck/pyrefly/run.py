"""Run the Pyrefly tensor-shape fixtures.

Native helper: Pyrefly must check the real ``src/forecasters/_native.py`` implementation
with no errors, proving the native quantile-axis normalization and selection shapes.

Positive fixture: Pyrefly must report no errors, so every ``assert_type`` proves the
inferred public shape.

Negative fixture: Pyrefly must report exactly the errors declared by ``# EXPECT:``
comments. A stale expectation (an error that disappears or moves) fails the check.

Run with the isolated dependency group so the shape-aware stub packages are installed:

    uv run --isolated --locked --no-default-groups --group=pyrefly \
        python packages/forecasters/typecheck/pyrefly/run.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
PUBLIC_CONFIG = ROOT / "pyrefly.toml"
NATIVE_CONFIG = ROOT / "pyrefly-native.toml"
NATIVE_SOURCE = ROOT.parents[1] / "src" / "forecasters" / "_native.py"
EXPECT_PREFIX = "# EXPECT:"
INSTALL_HINT = (
    "pyrefly is not installed. Run this script in the isolated checker environment:\n"
    "  uv run --isolated --locked --no-default-groups --group=pyrefly "
    "python packages/forecasters/typecheck/pyrefly/run.py"
)


def _run(
    source: Path, config: Path
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    executable = shutil.which("pyrefly")
    if executable is None:
        raise SystemExit(INSTALL_HINT)

    result = subprocess.run(
        [
            executable,
            "check",
            "--config",
            str(config),
            "--output-format",
            "json",
            str(source),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    start = result.stdout.find("{")
    if start == -1:
        raise SystemExit(
            f"Pyrefly produced no JSON report for {source.name}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout[start:])
    return result, list(payload.get("errors", []))


def _diagnostics(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def check_native() -> None:
    result, errors = _run(NATIVE_SOURCE, NATIVE_CONFIG)
    if result.returncode != 0 or errors:
        raise SystemExit(
            f"Native helper shape check failed ({len(errors)} diagnostics).\n"
            + _diagnostics(result)
        )
    print(f"native: {NATIVE_SOURCE.name} normalization/selection shapes verified")


def check_positive() -> None:
    fixture = FIXTURES / "positive.py"
    result, errors = _run(fixture, PUBLIC_CONFIG)
    if result.returncode != 0 or errors:
        raise SystemExit(
            f"Positive shape fixtures failed ({len(errors)} diagnostics).\n" + _diagnostics(result)
        )
    print(f"positive: {fixture.name} inferred all asserted shapes")


def _expected_errors(fixture: Path) -> dict[int, str]:
    """Map the line that must fail to the expected diagnostic kind.

    ``# EXPECT: kind`` applies to the same line when written inline, or to the next line
    when it is a standalone comment.
    """
    expected: dict[int, str] = {}
    for number, line in enumerate(fixture.read_text().splitlines(), start=1):
        if EXPECT_PREFIX not in line:
            continue
        kind = line.split(EXPECT_PREFIX, 1)[1].strip()
        code_before_comment = line.split(EXPECT_PREFIX, 1)[0].strip()
        target = number if code_before_comment else number + 1
        if target in expected:
            raise SystemExit(f"{fixture.name}: duplicate expectation for line {target}")
        expected[target] = kind
    if not expected:
        raise SystemExit(f"{fixture.name} declares no EXPECT comments")
    return expected


def check_negative() -> None:
    fixture = FIXTURES / "negative.py"
    result, errors = _run(fixture, PUBLIC_CONFIG)
    expected = _expected_errors(fixture)

    actual: dict[int, list[str]] = {}
    for error in errors:
        if error.get("severity") != "error":
            continue
        actual.setdefault(int(error["line"]), []).append(str(error["name"]))

    problems: list[str] = []
    for line, kind in expected.items():
        names = actual.get(line, [])
        if names != [kind]:
            problems.append(f"line {line}: expected exactly [{kind!r}], got {names!r}")

    unexpected = {line: names for line, names in actual.items() if line not in expected}
    for line, names in sorted(unexpected.items()):
        problems.append(f"line {line}: unexpected diagnostics {names!r}")

    if result.returncode == 0:
        problems.append("Pyrefly exited successfully although the negative fixture has errors")
    if problems:
        raise SystemExit(
            "Negative shape fixtures did not match their expectations:\n  "
            + "\n  ".join(problems)
            + "\n"
            + _diagnostics(result)
        )

    print(f"negative: {fixture.name} produced {len(expected)} expected static errors")


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit(
            "Pyrefly shape fixtures require Python 3.12+ because pyrefly-numpy-stubs "
            "requires it; the library itself still supports Python 3.11+."
        )
    check_native()
    check_positive()
    check_negative()
