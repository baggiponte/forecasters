"""Thin adapters over native SDKs for pretrained time-series foundation models.

Importing :mod:`forecasters` does not import PyTorch or either native SDK. Load a concrete
adapter with ``ChronosBolt.from_pretrained(...)`` or ``Timestral.from_pretrained(...)``.
"""

from forecasters._forecaster import Forecaster
from forecasters.chronos_bolt import ChronosBolt
from forecasters.timestral import Timestral

__version__ = "0.1.0"

__all__ = ["ChronosBolt", "Forecaster", "Timestral", "__version__"]
