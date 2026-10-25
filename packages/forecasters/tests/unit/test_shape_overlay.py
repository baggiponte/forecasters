"""Keep the hand-written Pyrefly overlay's predict signatures in step with the runtime.

The overlay under ``tests/shapes/stubs`` mirrors ``predict`` by hand, so nothing else stops
it drifting. These tests parse both files with ``ast``: they need neither NumPy, Torch nor
Pyrefly, so they run in the default test job and fail when a parameter is renamed, added,
or moved between positional and keyword-only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2]
RUNTIME_BASE = PACKAGE / "src" / "forecasters" / "_base.py"
RUNTIME_PROTOCOL = PACKAGE / "src" / "forecasters" / "_forecaster.py"
OVERLAY = PACKAGE / "tests" / "shapes" / "stubs" / "forecasters" / "_predict.pyi"

Signature = tuple[tuple[str, bool], ...]


def _predict_definitions(path: Path, class_name: str) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    (cls,) = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    return [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "predict"
    ]


def _is_overload(function: ast.FunctionDef) -> bool:
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "overload")
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
        for decorator in function.decorator_list
    )


def _signature(function: ast.FunctionDef) -> Signature:
    """Parameter names after ``self``, each flagged when keyword-only."""
    positional = [(arg.arg, False) for arg in function.args.args[1:]]
    keyword_only = [(arg.arg, True) for arg in function.args.kwonlyargs]
    return tuple(positional + keyword_only)


def _runtime_signature(path: Path, class_name: str) -> Signature:
    (implementation,) = [
        function
        for function in _predict_definitions(path, class_name)
        if not _is_overload(function)
    ]
    return _signature(implementation)


def test_overlay_declares_overloads() -> None:
    overloads = [function for function in _predict_definitions(OVERLAY, "ShapedPredict")]
    assert len(overloads) >= 2, "the overlay must declare at least the median and quantile forms"
    assert all(_is_overload(function) for function in overloads)


@pytest.mark.parametrize(
    ("runtime", "class_name"),
    [
        pytest.param(RUNTIME_BASE, "BaseForecaster", id="BaseForecaster"),
        pytest.param(RUNTIME_PROTOCOL, "Forecaster", id="Forecaster"),
    ],
)
def test_overlay_overloads_match_runtime_parameters(runtime: Path, class_name: str) -> None:
    expected = _runtime_signature(runtime, class_name)

    for overload in _predict_definitions(OVERLAY, "ShapedPredict"):
        assert _signature(overload) == expected, (
            f"{overload.lineno}: overlay predict parameters differ from {class_name}.predict "
            f"in {runtime.name}; update the overlay in tests/shapes/stubs"
        )
