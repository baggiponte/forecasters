# forecasters

A minimal Python interface over native SDKs for pretrained time-series foundation models.

The package exposes two concrete adapters, `ChronosBolt` and `Timestral`, plus one small
structural `Forecaster` protocol. It performs univariate, zero-shot, native-quantile
inference; it does not train, batch, generate samples or manage model deployments.

## Installation

```bash
python -m pip install forecasters
```

Native SDKs are optional and separate from the core NumPy-only installation:

```bash
# Chronos-Bolt through the official chronos-forecasting package
python -m pip install 'forecasters[chronos]'

# Timestral is not published on PyPI; install the official SDK from this repository commit
python -m pip install 'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
```

## Implemented API

```python
import numpy as np
from forecasters import ChronosBolt, Timestral

model = ChronosBolt.from_pretrained()       # amazon/chronos-bolt-tiny
# model = Timestral.from_pretrained()       # mistralai/Timestral-5M

context = np.sin(np.arange(64, dtype=np.float32) / 5.0)
median = model.predict(context, horizon=8)                    # (8,)
quantiles = model.predict(
    context, horizon=8, quantile_levels=[0.9, 0.1, 0.5]
)                                                              # (8, 3), request order
```

- `from_pretrained(...)` loads the checkpoint explicitly through the official SDK and
  retains the loaded object for repeated predictions. Both adapters accept typed
  `model_name`, `device`, `revision`, `cache_dir` and `local_files_only` parameters, with
  defaults `amazon/chronos-bolt-tiny` and `mistralai/Timestral-5M`.
- `predict(context, *, horizon, quantile_levels=None)` accepts only a one-dimensional,
  nonempty, finite NumPy array or PyTorch tensor with a floating dtype. NumPy views
  (including negative-stride and read-only views), float16, Torch bfloat16 and NumPy
  `longdouble` are supported. Python sequences, integers, booleans, complex arrays and other
  array libraries are rejected rather than coerced.
- NumPy input returns NumPy values with the input floating dtype. Torch input returns a
  tensor on the input device with the input floating dtype. Native compute device is
  configured separately by `from_pretrained(device=...)`.
- The default is the native 0.5 quantile with shape `(H,)`. Requested levels return
  `(H, Q)` in caller order, including `(H, 1)` for one level; no median is inserted.
- Quantiles are selected from the loaded checkpoint's native levels without interpolation
  or nearby-level substitution. `supported_quantile_levels`, `max_context_length` and
  `max_horizon` expose the loaded checkpoint's simple typed metadata.
- Context and horizon values beyond the native limits raise before inference. The adapters
  never truncate context or request SDK rollout beyond the native horizon.
- Native results must have exactly Chronos `(1, Q, H)` or Timestral `(1, 1, H, Q)` shape
  for the loaded quantile grid and requested horizon. Short, extra or unexpected dimensions
  raise before selection. Selected values are checked for finiteness both natively and after
  restoration; NaNs in known but unrequested quantile columns of a correct-shape result are
  ignored. Invalid SDK shapes, selected NaNs and float16 overflow raise `RuntimeError`
  instead of returning silently damaged forecasts.

`numpy` and the small `array-api-compat` package are the core dependencies. The latter is
used for standard family recognition, floating-dtype/finiteness checks and device access; it
does not make native model runtime portable. `import forecasters` does not import PyTorch or
either native SDK. NumPy-only consumers can install the package without PyTorch.

The Timestral SDK is installed from its official Git repository because it has no PyPI
distribution. The package intentionally does not advertise a `timestral` extra that cannot
be resolved from PyPI. The workspace root defines the Timestral Git source for development.

## Typing

The package ships `py.typed`. `predict` is overloaded so mypy and ty preserve the input
family: a NumPy context returns `NDArray` with the input floating dtype, and a Torch
context returns `Tensor`. The shape is `(H,)` for the default native median and `(H, Q)`
for requested levels.

The source repository also contains an experimental Pyrefly overlay that adds static
tensor-shape inference for the public API. With the shape-aware NumPy/Torch stubs,
Pyrefly proves a one-dimensional context of length `C` becomes `(H,)` or `(H, Q)` and can
reject a wrong horizon, rank, NumPy dtype or backend at the call site. A separate Pyrefly
config checks the real `_native.py` helper that both adapters use to normalize the Chronos
and Timestral native quantile axes. This syntax is understood by Pyrefly only. Postponed
annotations defer the annotations themselves, but `assert_type`'s second argument is an
ordinary expression and is evaluated at runtime; this example quotes it and is static-only:

```python
from __future__ import annotations

from typing import assert_type

import numpy as np

from forecasters import ChronosBolt

context: np.ndarray[[64], np.dtype[np.float32]]
model = ChronosBolt.from_pretrained()
assert_type(model.predict(context, horizon=8), "np.ndarray[[8], np.dtype[np.float32]]")
assert_type(
    model.predict(context, horizon=8, quantile_levels=[0.1, 0.5, 0.9]),
    "np.ndarray[[8, int], np.dtype[np.float32]]",
)
```

The quantile axis is gradual (`int`) because sequence lengths are not tracked statically;
changing the requested levels cannot change the proven rank or horizon. Torch dtypes are
not tracked by the current Pyrefly Torch shape stubs. These stubs are a checker-only
development fixture, are not included in the wheel and are not imported at runtime.
Runtime validation remains authoritative for context and horizon limits, native SDK
result shapes, finiteness and dtype/device restoration. See
[`typecheck/pyrefly/README.md`](typecheck/pyrefly/README.md) for commands and limitations.

## Verified behavior

Real-checkpoint parity, native-boundary fake tests, NumPy-only validation and lazy-import
checks live in `packages/forecasters/tests` in the source repository. The real-checkpoint
tests are opt-in with `FORECASTERS_NATIVE=1` and use cached checkpoints.

Not implemented in this milestone: training/fine-tuning, sample-based forecasts, batching,
multivariate inputs, covariates, timestamps, missing-value policies, calibration, model
selection, scikit-learn integration, deployment/multi-user serving, CLI and benchmarking
infrastructure. The Chronos-Bolt SDK can roll beyond 64 steps and Timestral beyond one
1,024-step block, but the library deliberately rejects those requests instead of using
rollout.

## Design

- [Architecture decisions](../../docs/adr/README.md)
- [Domain glossary](../../GLOSSARY.md)
- [Historical design contract](../../CONTRACT.md)

A license file is still needed before the first public release.
