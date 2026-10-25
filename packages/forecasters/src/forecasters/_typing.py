"""Private type variables shared by the public prediction overloads."""

from typing import TypeVar

import numpy as np

ScalarT = TypeVar("ScalarT", bound=np.generic)
