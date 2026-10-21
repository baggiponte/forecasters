# forecasters — minimal contract

> Historical design proposal, retained for comparison. The 2026-10-20 product direction
> reopens this contract's scope, point statistic, interface and mandatory worker design.
> See [the proposed ADRs](docs/adr/README.md). The freeze and product overrides below
> describe the previous proposal; they are not approval of the new direction or implemented behavior.

**Version:** 0.3 (post second external review)  
**Status:** sign-off candidate. Public API frozen after this revision; implementation follows.  
**Product direction:** a sklearn-flavoured `fit` / `predict` / `predict_proba` surface over native
foundation-model SDKs. It is *not* a full sklearn estimator and it trains nothing.

Deliberate product overrides are recorded in §10.

---

## 1. Scope

`forecasters` is the thinnest useful compatibility layer over pretrained time-series foundation models.

It owns:

- adapter registration and lifecycle;
- a versioned JSON-lines worker protocol and one process per model (dependency isolation);
- input validation, capability handling, and a small error taxonomy;
- a common `Forecast` value object carrying the model's **native** point and quantiles.

It does **not** own: training/fine-tuning, scaling, imputation, multivariate/exogenous plumbing, interpolation
between quantile levels, quantile-crossing repair, evaluation, plotting, serving, batching, or GPU scheduling.

**Zero-shot only.** `fit` captures the context and loads the model. No weights change.

---

## 2. Public API

### 2.1 Module

```python
def load(
    adapter: str,
    *,
    python: str | os.PathLike | None = None,  # worker interpreter; default sys.executable
    model: str | None = None,                  # override adapter default id/path
    device: str | None = None,                 # "cpu", "cuda", "mps", ...
    seed: int | None = None,                   # sampling models
    timeout: float | None = None,              # per predict; None = wait
    load_timeout: float | None = None,         # worker init/load; None = wait
    env: Mapping[str, str] | None = None,      # extra worker environment
    **adapter_kwargs,                          # revision, dtype, ...
) -> Forecaster: ...

def available() -> list[AdapterInfo]: ...
def register(name: str, target: str) -> None:  # "pkg.module:Class"; entry points preferred
```

- `load` is cheap: it validates name/arguments and builds the object. The worker starts on `fit`.
- `adapter_kwargs` must be JSON-serialisable (for example `dtype="bfloat16"`, not a torch dtype). They are
  delivered to the worker in the `init` message, never on the command line.
- `available()` returns adapter metadata derived from class attributes (default model, package, default
  capabilities). It does **not** claim the native package is installed in the worker interpreter.
- `register` takes an import string, never a class object. Built-in names are reserved. Registering a duplicate
  name raises `ValueError`. Third-party adapters are discovered from the `forecasters.adapters` entry-point
  group; duplicate entry-point names raise `ValueError`.

```python
@dataclass(frozen=True)
class AdapterInfo:
    name: str
    default_model: str
    package: str
    capabilities: Capabilities
```

### 2.2 `Forecaster`

```python
class Forecaster:
    adapter: str
    model: str
    capabilities: Capabilities   # default until fit; authoritative handshake values after

    def fit(self, y: ArrayLike) -> "Forecaster": ...
    def predict(self, h: int) -> np.ndarray: ...                 # (h,) float64, native point
    def predict_quantiles(self, h: int) -> Forecast: ...        # canonical quantile API
    def predict_proba(self, h: int) -> Forecast: ...            # documented alias of predict_quantiles
    def close(self) -> None: ...
    def __enter__(self) -> "Forecaster": ...
    def __exit__(self, *exc) -> None: ...
```

**`fit` order (normative):**

1. Validate `y` locally: 1-D, non-empty, finite, numeric; copy it.
2. If the worker is not running, start it and send `init`; wait for `ready` bounded by `load_timeout`.
3. Replace `self.capabilities` with the handshake values. Apply `min_context`/`max_context` and keep the newest
   `max_context` values.
4. Commit the context and increment its version.

A **load failure is terminal** (the object closes). A **validation failure after a successful load is
non-terminal**; the previous context remains. A failed first validation leaves the object unfitted.

`fit` again replaces the context and does **not** reload the model. `fit` and predict calls are serialised by a
per-object lock: use **one object per thread**; sharing one object between threads can interleave a refit and a
predict even though individual calls are atomic.

`predict(h)` returns the native point forecast (mean or median, declared by `capabilities.point`).
`predict_quantiles(h)` is canonical. `predict_proba(h)` is the same method under the product-requested name:
same signature, same result, one implementation and one test suite. `predict` and the quantile method share a
one-slot cache keyed by `(context_version, h)` so calling both does not run the model twice.

`h` must be a real `int`; `bool`, arrays and sequences are rejected with an `InvalidInput` message that mentions
sklearn incompatibility.

`close()` is idempotent. After close, calls raise `ClosedForecaster`; threads waiting on the lock when another
call closes the object also receive `ClosedForecaster`. A worker crash, timeout, protocol error or load failure
is terminal for the object; create a new `Forecaster`.

Not picklable, not fork-safe. `Forecast` is picklable.

### 2.3 `Forecast`

Immutable, levels-major, native numbers only.

```python
@dataclass(frozen=True)
class Forecast:
    adapter: str
    point: np.ndarray                  # (h,), read-only; native point
    levels: tuple[float, ...] | None   # native levels; None for a point-only model
    values: np.ndarray | None          # (len(levels), h), read-only; None for point-only
    meta: Mapping[str, Any]            # MappingProxyType
```

- `levels` and `values` are both present or both `None`; a point-only forecast has neither.
- `levels` are strictly increasing and unique. The worker validates this; a duplicate/invalid level is a
  protocol violation and raises `WorkerError(phase="protocol")` (terminal) rather than being repaired.
- `point` and `values` are `float64`, finite, read-only. `values` is not sorted; native quantile crossings are
  preserved and flagged in `meta["quantile_crossing"] = True`.
- `quantile(level)` returns the row for an **exact native** level (`abs(level - native) <= 1e-9`); otherwise
  `LevelNotAvailable` with the native levels. No interpolation. On a point-only forecast it raises
  `UnsupportedCapability`.
- `interval(coverage)` returns `(lower, upper)` only when both tails are native; otherwise
  `LevelNotAvailable` with the native levels.
- `meta` is required and uses a read-only mapping. Required keys: `model`, `sdk_version`, `point_kind`,
  `context_used`, `ms`. Optional keys: `revision`, `device`, `quantile_crossing`, `forecasters_version`.
  `point_kind` is `"mean"` or `"median"`. `ms` is the per-response native inference time.
- No `.median`, `.horizon`, `.ms` field or `.to_dict()` in 0.1; use `point`, `values.shape[1]`, and `meta`.

### 2.4 `Capabilities`

```python
@dataclass(frozen=True)
class Capabilities:
    quantiles: bool                 # predict_quantiles/predict_proba available
    point: Literal["mean", "median"]
    min_context: int = 1
    max_context: int | None = None
    max_horizon: int | None = None
```

- The class attribute on an adapter describes its **default model** and is what `available()` reports.
  The **worker handshake is authoritative**: a `model=` override may change context/horizon limits or even the
  point kind.
- `Forecaster.capabilities` starts as the class default and is replaced by the handshake after `fit`. The parent
  rejects the handshake only on hard invariant violations: adapter name mismatch, malformed types,
  `point` outside `{"mean", "median"}`, negative limits, or `min_context > max_context`. It does not compare
  limit values with the class defaults.
- `h < 1`, non-int → `InvalidInput`; `h > max_horizon` → `HorizonTooLong`.
- `len(y) < min_context` → `ContextTooShort`.
- `len(y) > max_context` → the newest `max_context` values are used; `meta["context_used"]` records the count.
  This is the one intentional input transformation.
- No `multivariate`, `exogenous`, `missing_values` or `samples` flags until those features ship.

---

## 3. Adapter contract

`Adapter` is an **abstract base class** (`abc.ABC`). Built-in and third-party adapters subclass it. The module
must be importable in **both** the parent and the worker; it must not import the native SDK at module import
time (only inside `load()`), so the parent can read class attributes without torch installed.

```python
class Adapter(ABC):
    ADAPTER_API: ClassVar[int] = 1
    name: ClassVar[str]
    default_model: ClassVar[str]
    package: ClassVar[str]              # install hint for MissingDependency
    capabilities: ClassVar[Capabilities]

    def __init__(
        self,
        *,
        model: str,
        device: str | None = None,
        seed: int | None = None,
        **kwargs,                       # JSON-serialisable adapter_kwargs
    ) -> None: ...

    @abstractmethod
    def load(self) -> None:
        """Import the native SDK, download/load weights, warm up. The only networked phase."""

    @abstractmethod
    def forecast(
        self, y: np.ndarray, h: int
    ) -> tuple[np.ndarray, tuple[float, ...] | None, np.ndarray | None]:
        """Return (native point (h,), native levels, native values (L, h)).

        Point-only models return levels=None, values=None and set capabilities.quantiles=False.
        No sorting, interpolation, clipping, scaling or fitting.
        """
```

Rules:

1. `forecast` receives a 1-D `float64` copy of the newest `max_context` values. It must not mutate or re-scale.
2. Call the native SDK's forecast method. Reshape/transpose and select native point/quantile columns; do not
   reimplement architecture, tokenisation, scaling, or quantile heads.
3. Return native numbers. The worker validates shapes/finiteness/levels and builds the `Forecast`; it does not
   repair numbers.
4. A point-only model returns a point and `levels=None`/`values=None`; `predict` works,
   `predict_quantiles` raises `UnsupportedCapability`.
5. `load()` is the only place that may use the network (weights). `forecast()` must be offline.
6. Missing native dependency → `MissingDependency(f"install {package}, model={default_model}")`.
7. Adapter exceptions become `WorkerError(phase="forecast")` with a short reason; the remote traceback is
   attached for debugging. A forecast-phase adapter error is **non-terminal**; a load-phase error is terminal.

The worker receives the adapter import string on argv and the options in an `init` message. The adapter
distribution must be importable in the worker interpreter.

---

## 4. Process and protocol

- One worker per `Forecaster`, started by `fit` with:
  `python -m forecasters.worker --adapter <module:Class>`.
- The worker interpreter must have `forecasters` importable (`pip install --no-deps forecasters`) and the native
  SDK installed. The library does **not** inject its own `site-packages` into `PYTHONPATH`; that would leak the
  parent's torch/numpy into the worker and defeat isolation. The worker must support the core's minimum numpy
  version, `numpy>=1.26`.
- **Options travel over stdin**, never argv, so tokens/paths do not appear in `ps` and non-JSON values are
  rejected early.
- Worker stdout is reserved for the protocol. The worker duplicates fd 1 before importing the SDK, redirects
  `sys.stdout`/fd 1 to stderr, and writes protocol lines to the duplicated fd. This protects against tqdm, HF
  warnings and `print`.
- The parent drains worker stderr on a reader thread into a bounded ring buffer (64 KiB) and includes the tail in
  `WorkerError`; it never leaves a `stderr=PIPE` unread.
- POSIX only in 0.1. Windows behaviour is unsupported and untested. The timeout implementation uses a reader
  thread/selector; blocking `readline()` alone is not sufficient.

### Handshake

```json
-> {"v": 1, "op": "init", "options": {"model": "amazon/chronos-bolt-tiny", "device": "cpu", "seed": 42}}
<- {"v": 1, "op": "ready", "adapter": "chronos_bolt", "adapter_api": 1,
    "forecasters_version": "0.1.0", "sdk_version": "2.3.2",
    "capabilities": {"quantiles": true, "point": "median", "min_context": 1,
                     "max_context": 2048, "max_horizon": 64},
    "load_ms": 187.4, "pid": 4321}
<- {"v": 1, "op": "error", "phase": "load", "kind": "missing_dependency",
    "error": "install chronos-forecasting, model=amazon/chronos-bolt-tiny", "traceback": "..."}
```

Pre-ready failures use the `op=error` envelope. The parent checks the adapter name, `v`, and `adapter_api`
exactly. `forecasters_version` may differ across the parent/worker installs; the policy is: same major version,
protocol `v` and `adapter_api` must match exactly, otherwise `WorkerError(phase="protocol")`.

### Requests

```json
-> {"v": 1, "id": 7, "op": "forecast", "y": [...], "h": 24}
<- {"v": 1, "id": 7, "ok": true, "point": [...], "levels": [...], "values": [[...]],
    "meta": {"ms": 12.3}}
<- {"v": 1, "id": 7, "ok": false, "kind": "adapter_error", "error": "...", "traceback": "..."}
```

- The parent requires every response `id` to match its request. A mismatched id, malformed line, failed JSON
  parse or unknown `kind` is a protocol error and closes the object.
- `levels`/`values` may be omitted or `null` for a point-only model.
- `kind` values and mappings:
  - `missing_dependency` → `MissingDependency`
  - `load_error` → `WorkerError(phase="load")`
  - `adapter_error` → `WorkerError(phase="forecast")`
  - `protocol_error` → `WorkerError(phase="protocol")`
  - `crash` → `WorkerError(phase="crash")` — generated by the parent from process/read failure; never sent by
    the worker
  - `timeout` → `Timeout` — generated by the parent; never sent by the worker
- `WorkerError.phase` is one of `"load"`, `"forecast"`, `"protocol"`, `"crash"`.
- Timeouts: `timeout` bounds each predict; on timeout the worker is killed and the object closes.
  `load_timeout` bounds `init`/`ready`; on expiry the worker is killed and the object closes.
- Cleanup: `close()` is idempotent and escalates EOF → SIGTERM → SIGKILL. `atexit` and `weakref.finalize`
  terminate orphaned workers; a worker also exits on stdin EOF. `KeyboardInterrupt` during predict kills the
  worker and re-raises.
- The worker inherits `os.environ` (HF_HOME, HF_TOKEN, CUDA_VISIBLE_DEVICES) plus `env=` overrides.
- `seed` is passed to the adapter; adapters document which outputs are deterministic.
- No automatic restart: a crash, load failure, timeout or protocol error is terminal for that `Forecaster`.
- Payloads are arrays of finite floats; context is capped by `max_context`, horizon by `max_horizon`, so JSON
  size is bounded. JSON float64 round-trips are sufficient for 0.1; no binary encoding, streaming, batching or
  per-call native kwargs. Non-finite values are rejected before serialisation and after decoding.

### Failure semantics

| Failure | Terminal for the object? |
| --- | --- |
| load failure, missing dependency, handshake/version mismatch | yes |
| worker crash, protocol error, predict timeout | yes |
| adapter exception during a forecast | no; object remains usable, context unchanged |
| invalid input, short context, horizon too long, level not available | no; previous state intact |
| `close()` | yes |

Threads waiting on the per-object lock when another call closes the object receive `ClosedForecaster`.

---

## 5. Errors

```
ForecasterError
├── UnknownAdapter(KeyError)
├── InvalidInput(ValueError)
│   ├── ContextTooShort
│   ├── HorizonTooLong
│   └── LevelNotAvailable
├── UnsupportedCapability(ValueError)
├── NotFittedError(RuntimeError)
├── ClosedForecaster(RuntimeError)
├── MissingDependency(ImportError)
├── WorkerError(RuntimeError)          # .phase in {"load","forecast","protocol","crash"}
└── Timeout(TimeoutError)
```

No silent fallbacks: an unsupported quantile request, horizon, or capability raises. `WorkerError` is the one
"worker failed" class; `phase` distinguishes load, forecast, protocol and crash.

---

## 6. Inputs

- `y`: 1-D array-like; `np.asarray(y, dtype=np.float64)`; copied; finite; non-empty; `len >= min_context`.
- A 2-D array raises `InvalidInput` (v0.1 is univariate only). Exactly one exception is used for all shape
  problems.
- `h`: real integer (`bool`, arrays and sequences rejected); `1 <= h <= max_horizon`.
- No timestamps, frequency, covariates (`X`), or missing values in 0.1. They are explicit roadmap items, not
  hidden arguments.
- Levels are native only: no interpolation, no clamping, no 0–100 units, no duplicate levels.

---

## 7. Versioning

The public Python API follows semantic versioning. The adapter contract carries `ADAPTER_API = 1`; the worker
reports it in the handshake. Protocol `v` and `adapter_api` must match exactly; `forecasters_version` must share
the major version. Changing the `Adapter` ABC, `Forecast`, `Capabilities` or public method signatures is an
API-breaking change.

---

## 8. Verification

Built-in adapters ship a conformance test:

```python
from forecasters.testing import check_adapter

check_adapter(
    adapter_cls=ChronosBoltAdapter,
    y=y,
    h=24,
    native_fn=lambda y, h: native_quantiles_and_point(...),
    atol=1e-5,
)
```

`check_adapter` asserts the adapter's point/levels/values match the native SDK within tolerance, with no sorting
or interpolation. It also checks shapes, finiteness, native-level handling, point-only behaviour, and capability
flags. Passing it is a release requirement for built-in adapters and the recommended badge for third-party
adapters.

The protocol/worker contract is covered by scenarios including stdout noise, malformed
replies, version/capability mismatch, crash, timeout, closed object, context truncation, level-out-of-range,
short reply, duplicate levels, and concurrent calls.

---

## 9. Examples

```python
from forecasters import load

with load("chronos_bolt", python=".venvs/chronos/bin/python") as f:
    f.fit(y)                            # stores context, loads the model once
    point = f.predict(24)               # native point
    q = f.predict_quantiles(24)         # native nine deciles (canonical name)
    q90 = q.quantile(0.9)               # exact native level
    lo, hi = q.interval(0.8)            # requires 0.1 and 0.9 to be native

    # Product-requested alias; identical result and implementation:
    same = f.predict_proba(24)

# Hybrid environment: one interpreter per family
bolt = load("chronos_bolt", python=".venvs/chronos/bin/python")
toto = load("toto2", python=".venvs/toto2/bin/python")
```

A custom adapter:

```python
from forecasters import Adapter, Capabilities

class MyAdapter(Adapter):
    ADAPTER_API = 1
    name = "my_model"
    default_model = "local/checkpoint"
    package = "my-model-sdk"
    capabilities = Capabilities(quantiles=True, point="median", max_horizon=128)

    def __init__(self, *, model, device=None, seed=None, **kwargs):
        ...

    def load(self):
        ...  # import the SDK here

    def forecast(self, y, h):
        ...  # return (point, levels, values)
```

---

## 10. Decisions and review reconciliation

### Deliberately kept despite reviews

1. **`fit` / `predict` / `predict_proba`.** The product direction is a familiar sklearn-flavoured surface.
   `fit` is context capture + model load; it trains nothing. The contract states loudly that this is not a full
   sklearn estimator. The prior-art survey recommends a stateless `forecast(y, h)`; that is a valid alternative,
   but the product requirement wins for 0.1.
2. **`predict_proba` exists.** `predict_quantiles` is canonical; `predict_proba` is a documented alias with the
   same implementation and tests. This preserves the requested name while giving the ecosystem a precise name.

### Adopted from the reviews

3. **Native numbers only.** No interpolation, no row sorting, no median coercion; duplicate/non-increasing levels
   are rejected. The only exception is context truncation, recorded in `meta["context_used"]`.
4. **Explicit native point.** `Capabilities.point` says mean or median; the handshake is authoritative.
5. **Minimal surface:** no samples, no multivariate/exogenous/missing-value flags, no `levels` argument, no
   `Superseded`, no `to_dict`, no `.median`, no `ms` field.
6. **Small error tree** with stdlib bases; `WorkerError.phase` collapses load/forecast/protocol/crash.
7. **String-based registration**, ABC adapters, entry points, parent+worker importability, no import-time SDK.
8. **Worker isolation:** `forecasters` installed in the worker venv, no parent `PYTHONPATH` injection.
9. **Init message, handshake, capability authority, id checking, stderr drain, cleanup, timeouts, seed, env,
   metadata, conformance suite.**
10. **Crash/timeout/protocol errors are terminal**; a forecast-phase adapter error is not.
11. **POSIX-only for 0.1**, documented.

### Resolved open decisions

- `timeout` and `load_timeout` default to `None` (wait).
- Entry-point group: `forecasters.adapters`; duplicate names raise `ValueError`; built-ins are reserved.
- `meta` keys: required `model`, `sdk_version`, `point_kind`, `context_used`, `ms`; optional `revision`,
  `device`, `quantile_crossing`, `forecasters_version`.
- Point-only representation: `levels=None`, `values=None`; `predict` works, quantile methods raise.
- `Forecast.interval` includes the native levels in its error message.
- Alias decision: both `predict_quantiles` (canonical) and `predict_proba` (alias) ship in 0.1.
