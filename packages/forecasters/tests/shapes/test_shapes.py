"""Check the public tensor-shape contracts with Pyrefly.

Positive fixtures must infer every asserted shape with no diagnostics. Negative fixtures
must emit exactly the diagnostics marked by EXPECT comments; otherwise a lost shape check
could silently pass CI.

Pyrefly lives in the isolated `pyrefly` dependency group, so these tests skip everywhere
else. Run them with:

    uv run --isolated --locked --no-default-groups --group=pyrefly pytest packages/forecasters/tests/shapes
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SHAPES = Path(__file__).resolve().parent
CONFIG = SHAPES / "pyrefly.toml"
POSITIVE = SHAPES / "fixtures" / "positive.py"
NEGATIVE = SHAPES / "fixtures" / "negative.py"
EXPECT_PREFIX = "# EXPECT:"

pytestmark = [
    pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="pyrefly-numpy-stubs requires Python 3.12+; the library itself supports 3.11",
    ),
    pytest.mark.skipif(
        shutil.which("pyrefly") is None,
        reason="pyrefly is installed only in the isolated `pyrefly` dependency group",
    ),
]


def _run(source: Path) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    result = subprocess.run(
        [
            shutil.which("pyrefly"),
            "check",
            "--config",
            str(CONFIG),
            "--output-format",
            "json",
            str(source),
        ],
        cwd=SHAPES,
        check=False,
        capture_output=True,
        text=True,
    )
    start = result.stdout.find("{")
    if start == -1:
        pytest.fail(
            f"Pyrefly produced no JSON report for {source.name}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout[start:])
    return result, list(payload.get("errors", []))


def _diagnostics(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


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
            pytest.fail(f"{fixture.name}: duplicate expectation for line {target}")
        expected[target] = kind
    if not expected:
        pytest.fail(f"{fixture.name} declares no EXPECT comments")
    return expected


def test_positive_fixture_infers_all_asserted_shapes() -> None:
    result, errors = _run(POSITIVE)

    assert result.returncode == 0 and not errors, (
        f"Positive shape fixtures failed ({len(errors)} diagnostics).\n" + _diagnostics(result)
    )


def test_negative_fixture_reports_exactly_the_expected_errors() -> None:
    result, errors = _run(NEGATIVE)
    expected = _expected_errors(NEGATIVE)

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

    assert not problems, (
        "Negative shape fixtures did not match their expectations:\n  "
        + "\n  ".join(problems)
        + "\n"
        + _diagnostics(result)
    )
