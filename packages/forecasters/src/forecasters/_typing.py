"""Private type variables shared by the public prediction overloads."""

from typing import TypeVar

import numpy as np

# Numeric and boolean NumPy scalars only: excludes str_, bytes_, object_, datetime64 and void.
# timedelta64 subclasses signedinteger, so it is still accepted; the SDK rejects it at runtime.
ScalarT = TypeVar("ScalarT", bound=np.number | np.bool_)
