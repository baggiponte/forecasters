# Pyrefly tensor-shape checks (experimental)

These checks cover two things:

1. the real native quantile-axis helper in `src/forecasters/_native.py`, checked by
   Pyrefly against its executable body;
2. the public `predict` shape contract through the shape-aware NumPy and Torch stubs:

```text
context: (context_length,)  ->  default median: (horizon,)
                            ->  requested levels: (horizon, quantiles)
```

The order is the runtime order documented by the package: horizon first, quantiles second.

## Run

```bash
uv run --isolated --locked --no-default-groups --group=pyrefly \
  python packages/forecasters/typecheck/pyrefly/run.py
```

The group is pinned in `uv.lock`. It must run isolated because
`pyrefly-numpy-stubs`/`pyrefly-torch-stubs` overlay the real libraries' type
information and conflict with ty. Do not add this group to the default development
environment.

## Native implementation proof

`pyrefly-native.toml` checks the real `packages/forecasters/src/forecasters/_native.py`.
Both adapters call that helper at runtime. Its body:

- normalizes Chronos `(1, Q, H)` to `(H, Q)`;
- normalizes Timestral `(1, 1, H, Q)` to `(H, Q)`;
- selects the median or the requested quantile rows.

The annotations use legacy `IntVar` names imported only under `TYPE_CHECKING` and
postponed annotations. A `TYPE_CHECKING`-only `assert_type` states the normalized shape,
and Pyrefly checks it against the actual `detach`, indexing and `transpose` operations. No
implementation is duplicated in a fixture or hidden behind a stub. The runtime package
does not depend on `shape_extensions`.

## Public contract fixtures

`fixtures/positive.py` must pass with zero diagnostics. Its `assert_type` calls verify:

- a one-dimensional NumPy context returns rank 1 for the default median and rank 2 when
  `quantile_levels` is supplied;
- the first result axis is the literal `horizon` when it is passed as an integer literal;
- NumPy float32, float64 and longdouble dtypes are preserved, including through a
  shape-erased `NDArray[np.float64]` annotation;
- the `Forecaster` protocol carries the same shape overloads as the concrete adapters;
- Torch input returns a Torch tensor rather than a NumPy array;
- ordinary NumPy/Torch shape operations (indexing and transpose) continue to infer shapes
  from the result.

`fixtures/negative.py` must fail with exactly the errors named by its `EXPECT` comments.
`run.py` checks both the presence and the location of every expected error, and rejects
unexpected diagnostics, so a stale expectation cannot silently pass. It exercises wrong
horizon, wrong rank, wrong NumPy dtype, non-floating (integer/boolean) contexts, a 2-D
context, a Python sequence context, a non-integer horizon, a missing horizon, non-numeric
quantile levels, and a NumPy/Torch backend mismatch.

## Limitations

- The quantile axis is gradual. Python sequences do not carry their length at the type
  level, so Pyrefly displays it as `int` and cannot prove that one requested level gives
  `(horizon, 1)`. The overlay does not invent an exact count.
- Torch shape stubs track shape but not element dtype or device, so Torch dtype/device
  preservation is runtime-tested, not proven statically.
- A non-literal `horizon: int` produces a gradual first axis (`(int, ...)`). Literal
  horizons are exact.
- `pyrefly-numpy-stubs` requires Python 3.12+, so this check runs on 3.12 even though the
  library supports Python 3.11+. The normal ty check still targets 3.11 semantics.
- The stubs are a checker-only overlay for the documented public API. They are not shipped
  in the wheel, are never imported at runtime, and are not a proof of the implementation.
  The separate native-helper check is the implementation proof; runtime validation remains
  authoritative for context and horizon limits, native SDK result shapes, finiteness and
  dtype/device restoration.
- The public fixtures are static-only and never executed. Postponed annotations defer
  annotations, but `assert_type`'s second argument is an ordinary expression that would be
  evaluated if the fixture were imported. Run them only through Pyrefly.

## Maintenance

Keep the public overlay signatures aligned with `packages/forecasters/src/forecasters`. The
positive and negative fixtures are the public-contract regression guards. The native helper
is checked directly, so a change to the Chronos/Timestral axis order or selection must be
reflected in its annotations or the Pyrefly check fails.
