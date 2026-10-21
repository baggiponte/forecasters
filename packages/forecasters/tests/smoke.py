"""Run against installed distributions in environments without optional SDKs."""

import sys
from importlib.resources import files
from importlib.util import find_spec

import forecasters

assert forecasters.ChronosBolt and forecasters.Timestral and forecasters.Forecaster
assert files("forecasters").joinpath("py.typed").is_file()
for name in ("torch", "chronos", "timestral", "transformers"):
    assert name not in sys.modules, f"{name} was imported eagerly"
    assert find_spec(name) is None, f"{name} was installed as a required dependency"
