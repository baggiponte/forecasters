# Review: `forecasters` CONTRACT.md (draft 0.1)

## Verdict

The draft has the right instincts: native SDK calls, process isolation, raising errors instead of falling back silently, and a short non-goals list. It still breaks its own goals in a few places:

- **It changes numbers.** Interpolating quantiles, sorting each horizon row, and always using the median as the point forecast all mean the output can differ from the native SDK. That conflicts with "get the same numbers as the model's native SDK".
- **It contradicts itself.** At least ten clauses disagree with each other (listed in §1).
- **It plans too far ahead.** It includes capability flags for future features, a 15-class error tree, a `Superseded` mechanism, and `samples`.
- **Some necessary clauses are missing.** The worker protocol channel, `PYTHONPATH` isolation, timeouts, orphaned processes, and seeding/determinism are not covered.

The fixes below mostly remove things rather than add them.

---

## 1. Contradictions (must fix)

| # | Clauses | Problem | Fix |
|---|---|---|---|
| C1 | §1 "same numbers as native SDK" vs §2.3 "worker sorts each horizon row" | Sorting changes native output whenever quantiles cross (this happens with Chronos-Bolt and TimesFM on short contexts). Calling it a "numerical guard, not a modelling step" is still modelling. | Don't sort. Validate shape and finiteness only. If you want, put `meta["quantile_crossing"]=True`. |
| C2 | §1 "same numbers" vs §2.2 `levels=` interpolation and `Forecast.quantile` linear interpolation | Linear interpolation between quantiles assumes a distribution shape. That is modelling logic. | Allow only native levels (see Q3/Q4). |
| C3 | §2.2 "point forecast is the median" vs "if point-only, `predict` calls the native point path" vs §4 rule 3 "point-only returns `levels=(0.5,)`" | A point-only model's native output is usually a mean, so labelling it 0.5 is false. Also, the `Adapter` only has `forecast()`, so there is no "native point path" to call. | Make the adapter return an explicit `point` array. See Q5. |
| C4 | `predict(h, *, levels=...)` | A `levels` argument on a method that returns one point array means nothing. | Remove it. |
| C5 | §4 protocol sends `"levels"` in the request; `Adapter.forecast(y, h)` takes no levels | The worker would have nothing to do with them. | Remove `levels` from the protocol. Level selection is a slice in the parent process. |
| C6 | Module `load()` vs method `Forecaster.load()`; §2.1 says "worker starts on `fit`/`load`" | Two meanings for one name, and the sentence is ambiguous. | Rename the method to `start()`, or drop it (Q1). |
| C7 | §2.1 "`load` validates adapter availability… does not start a process" | You can't check whether the SDK is importable in another interpreter without starting that interpreter. | `load()` checks only the name and the arguments. `MissingDependency` is raised when the worker starts. |
| C8 | §5 "crash → `WorkerError` until `close()` and a new `load()`" vs §2.2 "after `close()`, calls raise `ClosedForecaster`" | The recovery path described is forbidden by the close rule. | A crash is terminal for that object. Build a new `Forecaster`. |
| C9 | §5 "newest request wins, stale raises `Superseded`" vs §2.2 per-object lock | Under the lock there can never be two pending requests. That makes `Superseded` dead code with its own error semantics. "Newest wins" is also the wrong policy for a forecasting library. | Delete `Superseded`. Allow one request in flight, enforced by the lock. |
| C10 | §3 "use last `max_context` values" vs §6 "no silent fallbacks" | Truncating context is a silent change to the input. | Keep truncation, since every native SDK does it anyway, but make it visible: `meta["context_used"]=n`. Document it as the one intentional exception. |
| C11 | §6 `Timeout` exists, but no API takes a timeout | The error can never be raised. | Add `timeout` to `load()` (§4). |
| C12 | `Forecast.key` vs `Forecaster.model`/`name` vs errors carrying `model` "(key)" | Three names for two concepts. | Use `adapter` (registry key) and `model` (HF id/path) everywhere. |
| C13 | `meta: Mapping = {}` on a frozen dataclass | `dataclass` raises `ValueError` for a mutable default. `frozen=True` also doesn't make an ndarray immutable. | Use `field(default_factory=dict)` (or `MappingProxyType`), and set `values.flags.writeable = False`. |

---

## 2. Answers to the 10 open questions

**1. `fit` semantics / stateless forecast.**
`fit` should validate and store the context, and start the worker if it isn't running. Users expect the cost and the failures (`MissingDependency`, `ModelLoadError`) to show up at `fit`, not on the first predict. Refitting must not reload the model; this needs to be a clause, not an implied behaviour.
Do not add a public stateless `forecast(y, h)`. `f.fit(y).predict_quantiles(h)` already is the stateless call, because `fit` is cheap after the first one. Two ways to do the same thing is not "thinnest".
Be honest in the docs: this API is sklearn-*flavoured*. `predict(h)` is not sklearn's `predict(X)`, and `clone`, `get_params` and pipelines won't work. Say so, so nobody files sklearn-compatibility bugs.

**2. Name and return type.**
Return a `Forecast` object, because levels must travel with the values. A raw `(h, n)` array with levels in a separate place is how bugs happen.
Name it `predict_quantiles`. In sklearn, `predict_proba` means class probabilities, so reusing it here misleads people. Don't add an alias: two names doubles the docs and tests and never goes away. (sktime already uses `predict_quantiles` and `predict_interval`, which is a useful precedent.)

**3. Native vs common levels.**
Use native levels only. A common grid forces interpolation or extrapolation, which is modelling (C2). Comparing models is the Line-up app's job: it can take the intersection of native levels, or interpolate itself and own that choice.

**4. Out-of-range intervals.**
Raise, and go further than the draft: also raise for in-range levels that aren't native. `levels=` must be a subset of the native levels, matched with a fixed tolerance (e.g. `abs ≤ 1e-9`). `interval(c)` works only if both `(1-c)/2` and `(1+c)/2` are native. That makes interpolation and clamping the same question with the same answer: the library does neither.

**5. Point forecast semantics.**
Use whatever the native SDK calls its point forecast, and declare which it is. TimesFM's point output is the mean, Chronos-Bolt's is the 0.5 quantile, and Moirai's is the median of its samples. Forcing "median" contradicts "same numbers" for TimesFM.
Add `Forecast.point` (shape `(h,)`) and `Capabilities.point: Literal["mean", "median"]`. `predict(h)` returns `Forecast.point`.

**6. Process per model.**
Keep one worker per `Forecaster`. A pool or registry means scheduling, caching and sharing GPU memory, all of which are already non-goals. Line-up can hold a `dict[str, Forecaster]`; that is about 10 lines of app code.

**7. Registration and discovery.**
This needs real design work. `register(name, cls)` is broken across processes: a class defined in the parent's `__main__` can't be imported in the worker.

- The registry maps `name → "module:qualname"` strings, never class objects.
- Built-in adapters go in a static table. Third-party adapters use the `forecasters.adapters` entry-point group, discovered **in the parent environment**.
- An adapter module must import its SDK lazily (inside `load()`), so the parent can read the class attributes without torch installed.
- The worker gets the import string through argv. The adapter's *distribution* must be importable in the worker venv, either installed there or passed via `extra_paths=` (see §4). State this explicitly.
- `register(name, "pkg.mod:Cls")` can stay for local development, as long as it takes a string.

**8. Capabilities.**
Booleans are enough, and fewer are better. Remove `multivariate`, `exog` and `missing_values`: a flag that is always `False` is not a capability, it's a roadmap. Remove `samples` (Q10). Keep `quantiles`, `point`, `min_context`, `max_context` and `max_horizon`. Add fields when a feature actually ships. That is additive and doesn't break anything.

**9. Concurrency.**
Keep the lock. A single stdin/stdout pipe has to be serialised anyway, so the lock is the minimum correct implementation, not an extra feature. Document it as: "thread-safe, serialised; use separate objects for parallelism."

**10. Samples.**
Defer. Sending `(n_samples, h)` as JSON is expensive, it's unused in v0.1, and adding it later as `Forecast.samples` plus a capability flag doesn't break anything.

---

## 3. Over-engineering against the "thinnest layer" goal

- **Interpolation, row sorting, and median-as-point:** these are modelling. Remove them (C1, C2, Q5).
- **15 error classes:** collapse to about 8, and inherit from stdlib types so `except ValueError` works:
  ```
  ForecasterError
  ├── UnknownAdapter(KeyError)
  ├── InvalidInput(ValueError)           # shape, finiteness, h, context length, levels; message says which
  ├── UnsupportedCapability(ValueError)  # e.g. predict_quantiles on a point-only model
  ├── NotFitted(RuntimeError)
  ├── ClosedForecaster(RuntimeError)
  ├── MissingDependency(ImportError)
  ├── WorkerError(RuntimeError)          # load failure, adapter exception, crash; .remote_traceback
  └── Timeout(TimeoutError)
  ```
  If you want, keep `HorizonTooLong` and `LevelNotAvailable` as subclasses of `InvalidInput`, but nobody catches `ContextTooShort` separately from `InvalidInput`. `ModelLoadError`, `AdapterError` and `WorkerError` all mean "the worker failed", so they become one class with a `phase` attribute.
- **`Forecast.ms`, `.to_dict()`, `.horizon`, `.median`:** `ms` goes into `meta`. `to_dict` is a serialisation policy (and it drops samples and meta); leave it to apps. `.horizon` is just `values.shape[1]`. `.median` hides the mean/median issue.
- **`revision` as a top-level argument:** it's HF-specific and some adapters load local checkpoints. Pass it through `**kwargs`. Keep `device`, since it's universal.
- **`available()` returning `package`:** fine, but don't promise that it checks installation (C7).
- **`Superseded`:** delete it (C9).

---

## 4. Missing contract clauses

1. **Protocol channel.** Native SDKs print to stdout: tqdm bars, HF warnings, `print`. That corrupts JSON on stdout. The worker must move the protocol to a duplicated file descriptor and point `sys.stdout` at stderr before importing the SDK (or use a dedicated pipe fd). This is a must.
2. **`PYTHONPATH` isolation.** "Prepend the library's package root" is dangerous: if that root is the host's `site-packages`, the worker silently imports the host's torch and numpy, which defeats isolation. Prepend a directory containing **only** the `forecasters` package, or ship the worker as a zipapp or a single file. Also require the core to work with any numpy ≥ X in the worker (state the minimum version).
3. **Handshake.** Define `hello → {"v", "adapter", "sdk_version", "capabilities", "load_ms"}`. If the protocol version doesn't match, raise `WorkerError`. If the worker-reported capabilities differ from the class attributes, raise as well.
4. **Timeouts.** `load(..., timeout_load=None, timeout_predict=None)`. Weight downloads can take minutes, so the load timeout defaults to `None`. On timeout, kill the worker and mark the object closed.
5. **Orphans and cleanup.** The worker exits when stdin reaches EOF, which happens when the parent dies. The parent registers `atexit` and `weakref.finalize` to terminate the worker. `close()` is idempotent and escalates: EOF, then SIGTERM, then SIGKILL after N seconds. Define what `KeyboardInterrupt` during a predict does: kill the worker, close the object, and re-raise.
6. **Environment.** The worker inherits `os.environ` (HF_HOME, HF_TOKEN, CUDA_VISIBLE_DEVICES). Add an optional `env=` override. State that the network is allowed during `load`/`fit` only.
7. **Determinism.** Sampling models (Chronos-T5, Toto, Moirai) can't give "the same numbers" without a seed. Add `seed: int | None` to `load()`, passed to the adapter, and require adapters to document which outputs are deterministic.
8. **Reproducibility metadata.** `Forecast.meta` must include `sdk_version`, `model`, `revision` (if known), `device` and `context_used`.
9. **Wire precision.** JSON `float64` round-trips through `repr`, but NaN and Inf aren't valid JSON. Inputs are already required to be finite; outputs must be too, or raise `WorkerError`. State the payload limits, or that long contexts are slow over JSON.
10. **Conformance suite.** "Same numbers as native" is only enforceable with a shipped test harness: `forecasters.testing.check_adapter(cls, y, h, native_fn, atol)`. Make passing it a requirement for built-in adapters.
11. **Versioning.** Semver applies to the Python API and the `Adapter` ABC. The protocol version is internal, but the worker and parent must come from the same library version, which is guaranteed by the shipped-package approach in (2).
12. **Per-call native kwargs.** Some SDKs take per-call options, such as TimesFM's `freq`. Decide now: v0.1 allows them only at load time via `**kwargs`. No per-call passthrough.

---

## 5. Smallest revised public API

```python
# forecasters/__init__.py
def load(
    adapter: str,
    *,
    python: str | os.PathLike | None = None,   # worker interpreter; default sys.executable
    model: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    timeout: float | None = None,              # per-predict; load is unbounded
    env: Mapping[str, str] | None = None,
    **adapter_kwargs,                          # revision, dtype, freq, ...
) -> Forecaster: ...                           # cheap: name + arg validation only

def available() -> list[AdapterInfo]: ...      # name, default_model, package, capabilities
def register(name: str, target: str) -> None: ...   # "pkg.mod:Class"; entry points preferred

class Forecaster:
    adapter: str
    model: str
    capabilities: Capabilities

    def fit(self, y: ArrayLike) -> Self: ...              # validate+copy; start worker once
    def predict(self, h: int) -> np.ndarray: ...          # == predict_quantiles-free native point, (h,) float64
    def predict_quantiles(self, h: int, *, levels: Sequence[float] | None = None) -> Forecast: ...
    def close(self) -> None: ...                          # idempotent
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc) -> None: ...

@dataclass(frozen=True)
class Forecast:
    adapter: str
    point: np.ndarray                 # (h,), read-only; capabilities.point says mean|median
    levels: tuple[float, ...]         # native subset, strictly increasing
    values: np.ndarray                # (len(levels), h), read-only, NOT re-sorted
    meta: Mapping[str, Any] = field(default_factory=dict)   # sdk_version, context_used, ms, ...

    def quantile(self, level: float) -> np.ndarray: ...     # exact native match or InvalidInput
    def interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray]: ...  # both tails native or raise

@dataclass(frozen=True)
class Capabilities:
    quantiles: bool
    point: Literal["mean", "median"]
    min_context: int = 1
    max_context: int | None = None
    max_horizon: int | None = None

class Adapter:                         # imported in parent WITHOUT the SDK; SDK imported in load()
    name: ClassVar[str]
    default_model: ClassVar[str]
    package: ClassVar[str]
    capabilities: ClassVar[Capabilities]
    def __init__(self, *, model: str, device: str | None, seed: int | None, **kwargs): ...
    def load(self) -> None: ...
    def forecast(self, y: np.ndarray, h: int) -> tuple[np.ndarray, tuple[float, ...], np.ndarray]:
        """Return (point (h,), native levels, quantiles (n, h)). No sorting, no interpolation."""
```

Note on `predict`: it returns `Forecast.point` and works on point-only models.

**Changes from the draft:**
- **Removed:** `Forecaster.load`, `predict(levels=)`, the `predict_proba` name, interpolation, row sorting, `samples`, `ms` as a field, `to_dict`, `.median`, `.horizon`, the three future capability flags, `Superseded`, `revision` as a top-level argument, and about 7 error classes.
- **Added:** `Forecast.point`, `Capabilities.point`, and the `seed`, `timeout` and `env` arguments.
- **Clauses added:** the protocol fd, `PYTHONPATH` isolation, the handshake, orphan cleanup, the conformance suite, and string-based registration.

**Net:** the public API shrinks and every number it returns is the SDK's own. The only exception is context truncation, which is visible in `meta["context_used"]`.
