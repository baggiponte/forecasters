# Interface options: the smallest honest typed surface

> User decision recorded after this comparison: direct, typed, minimal `predict(context, horizon=...)`
> was accepted on 2026-10-20 in [ADR 0002](../docs/adr/0002-minimal-predict-interface.md).
> Alternatives below are retained as research, not pending questions about the chosen callable surface.

**Status:** research note. Revised 2026-10-20 after the product direction was reopened; not a contract change
and not implementation.  
**Supersedes:** the first revision of this note, which treated CONTRACT v0.3 and its frozen API as binding.  
**Scope:** public Python forecasting interface only. Worker protocol, process lifecycle, and adapter discovery are
implementation details unless they leak into the typed surface.  
**Inputs:** `README.md`, `CONTRACT.md` (historical v0.3), proposed ADRs `docs/adr/0001`–`0004`,
`design/prior-art.md`, `design/model-matrix.md`, and the official scikit-learn and sktime documentation/source
fetched with `curl` on 2026-10-20 (links in §11).

---

## 0. Current recommendation

Follow the median-first direction of the proposed ADRs:

1. **One public call:** `predict(context, *, horizon=...) -> NDArray[np.float64]`, length `horizon`, element 0
   the next observation, values the predictive **median** (0.5 quantile) in the context's original units.
2. **Context per call.** No `fit`, no fitted/unfitted state, no context version, no cache, no per-object lock.
3. **No public probabilistic surface in the first milestone.** No quantile result object, no `Capabilities`
   dataclass, no prediction-interval helper; `predict` returns a bare NumPy vector.
4. **Native SDKs in-process.** Each adapter owns its SDK object and calls native inference; no mandatory
   worker, JSON-lines protocol, supervisor, or handshake (ADR 0004).
5. **Minimal abstractions.** Concrete adapters plus at most one small structural type; no registry, plugin
   discovery, or capability/tag hierarchy for two models (ADR 0002).
6. **Median source is explicit.** A native 0.5 quantile where the SDK has one; otherwise a documented,
   seeded sample estimate (ADR 0001). Quantile grids, intervals, requested levels and sample output are
   deferred to the study in ADR 0003.

Why this is a defensible narrowing rather than a regression:

- **A median-only contract is a subset, not a misstatement.** It exposes one statistic and makes no claim
  about quantiles, so dropping a native nine-decile grid is not false capability reporting. The earlier
  finding that median-only "fails conformance" applies only if the historical CONTRACT v0.3 — which requires
  native levels/values from models that have them — is retained. That contract is now a historical proposal.
- **Every in-scope model can supply a 0.5 quantile.** `design/model-matrix.md` records that Chronos-Bolt's
  returned "mean" is already the 0.5 column, that most candidates expose a median column, and that
  sample-first models (Chronos v1, Moirai 1.1, Toto 1, Lag-Llama, Sundial) expose samples from which a median
  is estimated. The milestone does not need a model whose native output lacks a median.
- **The product compares median curves.** Preserving arbitrary native point semantics (mean for TimesFM 2.5,
  sample mean for Chronos v1) has no consumer in the first milestone; it can be reopened when a comparison
  view actually needs conditional means.
- **Thinness is a product requirement here.** State machine, capability reporting, level negotiation and
  worker isolation were all justified by a larger framework that does not exist.

Why the previous revision of this note recommended otherwise: it read CONTRACT v0.3's freeze and §10 product
overrides as current constraints, so it optimized for preserving that surface and for native-point fidelity.
The user has reopened those decisions; the ADRs, not the freeze, now set direction. What survives from that
analysis is the *evidence* below (§3, §4) — sklearn and sktime compatibility claims, and the cost of adding
estimator machinery — which is independent of which direction is chosen.

---

## 1. The question

### Additional direct prior art: FoundationForecast

[TimeCopilot/FoundationForecast](https://github.com/TimeCopilot/foundationforecast) already unifies many of the
same model families. Its example uses `FoundationForecast(models=[...]).forecast(df, h=24, level=[90])`,
with `unique_id`, `ds` and `y` columns; it also covers cross-validation, anomalies and exogenous inputs.
This is a real alternative to building another unified forecasting interface, not merely a model catalog.

The [current project manifest](https://github.com/TimeCopilot/foundationforecast/blob/main/pyproject.toml)
declares many model runtimes as core dependencies, including torch, GluonTS and several `timecopilot-*`
distributions. For the requested first milestone, direct 1-D arrays, optional per-model SDK dependencies and
upstream native SDKs are concrete differences that justify exploring Forecasters. Wrapping FoundationForecast
inside another generic facade would add indirection without resolving those differences. This is a scope and
dependency assessment, not a benchmark or a claim that its implementation is inferior.

The shared interface only earns its maintenance cost if two adapters remove SDK-specific shape/statistic
handling from the same canvas consumer. If that benefit does not materialize, prefer direct native calls or
reuse an existing library rather than expanding Forecasters into another framework.

What is the smallest strongly typed Python interface that exposes zero-shot time-series foundation-model
inference without pretending to be an estimator framework?

Four candidates were compared; the second revision chooses the first:

1. **Stateless call (recommended):** `predict(context, *, horizon) -> NDArray[np.float64]`.
2. **sklearn-flavoured object:** `fit(y) -> self`, `predict(horizon) -> np.ndarray`.
3. **sktime-shaped object:** `fit(y, fh) -> self`, `predict(fh) -> series`.
4. **CONTRACT v0.3 surface (historical):** `fit(y)`, `predict(h)`, `predict_quantiles(h) -> Forecast`,
   `predict_proba` alias, `Capabilities`, worker-backed `Forecaster`.

The durable conclusion is that options 2 and 3 are *familiar names*, not compatibility, and that option 4 is
a product façade with a state machine and a probabilistic obligation. Option 1 is now the recommendation
rather than a conditional alternative.

---

## 2. Options at a glance

| Option | State | Horizon | Probabilistic result | Fits sklearn tools? | Fits sktime tools? | Main strength | Main trap |
|---|---|---|---|---|---|---|---|
| **1. Stateless** `predict(y, *, horizon)` | No data state; model handle only | Per call | None in first milestone | No | No | Honest zero-shot semantics; no refit, cache, lock, or `NotFittedError` state | Context passed on every call; no familiar `fit` workflow |
| **2. sklearn-style** `fit(y).predict(h)` | Captured context | Per call | Separate method | No | No | Familiar method names; `fit` returns self | Implies `get_params`, `clone`, `fit(X,y)`, `predict(X)`, `score`; none are true |
| **3. sktime-style** `fit(y, fh).predict(fh)` | Captured context + `fh` | `fh`, possibly absolute | Quantiles/intervals/distribution | No | No | Matches forecasting vocabulary and `fh` | Requires `X`, `ForecastingHorizon`, tags, `update`, `score`, `clone`; not a `BaseForecaster` |
| **4. CONTRACT v0.3** | Captured context + worker | Per call `h: int` | `Forecast` with native levels | No | No | Explicit capabilities; native numbers; pre-designed | `predict_proba` collides outside this library; state/cache/lock/worker rules are extra; quantile obligation blocks a median milestone |

Current status: option 1 is the ADR 0002 proposal; options 2 and 3 remain rejected for the same reasons as
before; option 4 is retained as a historical comparison, useful if a larger model zoo or a probabilistic
surface is ever reopened.

The important distinction remains between **syntax familiarity** and **protocol compatibility**. Option 1
makes the weaker claim: one function, no estimator vocabulary at all.

---

## 3. What the official APIs actually require

### 3.1 scikit-learn

Verified from the official `BaseEstimator`, `clone`, `check_is_fitted`, `Tags`, `cross_val_score`,
`TimeSeriesSplit`, and common-pitfalls pages.

- `BaseEstimator` supplies parameter handling: `get_params`/`set_params`, `clone`, repr, serialization hooks.
  All constructor parameters must be explicit `__init__` keyword arguments; no `*args`/`**kwargs`.
- `clone(estimator)` constructs a **new unfitted estimator with the same parameters**. It deep-copies parameter
  values but not fitted data. The implementation calls `estimator.get_params(deep=False)`, constructs
  `klass(**params)`, and raises `TypeError` if the object has no `get_params`.
- `check_is_fitted` looks for attributes with a trailing underscore, or `__sklearn_is_fitted__`; a stateless
  estimator can declare `requires_fit=False` in its tags.
- `cross_val_score(estimator, X, y, cv=...)` requires an estimator implementing `fit`; it passes
  `X_train, y_train` to `fit`, then predicts on `X_test`, then calls the estimator's default `score` or a scorer.
  The default `cv` is 5-fold `(Stratified)KFold` for ordinary estimators.
- For time-ordered data, `TimeSeriesSplit` exists precisely because ordinary CV "would lead to training on future
  data and evaluating on past data". It produces expanding train prefixes and later test blocks.
- The common-pitfalls guide states the general rule: "never call fit on the test data", because leakage produces
  optimistic estimates.

**Consequences for `forecasters`:**

- Neither the recommended `predict(context, *, horizon)` nor the historical `Forecaster` is a scikit-learn
  estimator. `fit(y)`, if ever added, is not `fit(X, y)`; `predict(h)` is not `predict(X)`; no `get_params`,
  `set_params`, `clone`, `score`, `__sklearn_tags__`, or `__sklearn_is_fitted__` exists.
- `clone(forecaster)` raises `TypeError` at the `get_params` check, not merely because a native handle cannot be
  copied.
- `cross_val_score(forecaster, y, ...)` cannot work: it will pass a training array as `X` to `fit(y)`, then pass
  test values as the horizon to `predict(h)`, and it expects a `score` method that does not exist.
- Even if `get_params`/`clone`/`score` were added piecemeal, the X/y sample-matrix assumption and the absence of
  a time-series splitter would remain. That is the wrong compatibility target.
- Reusing one object across folds is unsafe whenever context is captured: `fit` mutates the captured context and
  the native model is shared. A correct backtest needs an independent object per fold, or a dedicated
  time-series backtest API. The stateless form sidesteps this: each call receives exactly the prefix and horizon.
- If real scikit-learn integration is ever required, it needs a concrete consumer and an adapter with genuine
  estimator semantics, not `BaseEstimator` inheritance on a forecasting object (ADR 0002 review decision).

### 3.2 sktime

Verified from the `BaseForecaster` API reference, the official `_base.py` docstrings, the tags reference, and the
probabilistic-forecasting guide.

- Public signatures include `fit(y, X=None, fh=None)`, `predict(fh=None, X=None)`,
  `predict_quantiles(fh=None, X=None, alpha=None)`, `predict_interval(fh=None, X=None, coverage=0.90)`,
  `predict_proba(fh=None, X=None, marginal=True)`, `predict_var`, `update`, `score`, and `reset`.
- `fh` is a `ForecastingHorizon`, not a bare integer: it can be relative or absolute (for example a pandas
  index). It may be supplied at `fit` or `predict`, governed by the `requires-fh-in-fit` tag (default `True`).
- `X` is exogenous data. `y` may be a `Series`, `Panel`, or `Hierarchical` container, not just a 1-D array.
- `fit` is required unless the `fit_is_empty` tag is `True`. The object has state (`is_fitted`, `cutoff`, `fh`),
  `clone`, `get_params`/`set_params`, `get_fitted_params`, `score` (MAPE), `update`, and `reset`.
- `capability:pred_int` defaults to `False`. When true, all four probabilistic methods are available; interval
  and quantile forecasts are treated as equivalent information, and variance/distribution derivations may rely on
  normality assumptions. The docs warn that derived distribution forecasts may not be consistent with the
  intervals or quantiles.
- `score` is MAPE on `predict(fh, X)` against `y`; it is not a general evaluation framework.

**Consequences for `forecasters`:**

- The recommended interface is not a `BaseForecaster`, and deliberately so: no `X`, `fh` objects, `update`,
  `reset`, tags, `score`, or `clone`.
- sktime pipelines, registry filtering, tuning, and backtesting will not accept it.
- A later `Capabilities` dataclass would be conceptually like a tag bundle, but it would not be sktime's tag
  protocol and could not be discovered by `all_estimators` or `get_tags`. That is another reason not to build
  one for a two-model milestone.

### 3.3 The genuinely shared vocabulary

Only the vocabulary transfers:

- quantile levels are probabilities in `[0, 1]`;
- a median is the 0.5 predictive quantile;
- prediction intervals are derived from quantiles of the future observation, not from parameter uncertainty.

That vocabulary is enough to document a median return value; it is not enough to claim protocol compatibility
with either library.

---

## 4. Familiarity versus compatibility

| Expectation | scikit-learn | sktime | CONTRACT v0.3 `Forecaster` (historical) | Verdict |
|---|---|---|---|---|
| `fit(X, y)` | Required | No; `fit(y, X, fh)` | No; `fit(y)` | Incompatible |
| `predict(X)` | Required | No; `predict(fh, X)` | No; `predict(h: int)` | Incompatible |
| `get_params` / `set_params` | From `BaseEstimator` | From skbase | Absent | Incompatible |
| `clone` | Deep-copies params, unfitted | Supported | Absent | Incompatible |
| `score` | Optional but expected | MAPE by default | Absent | Incompatible |
| CV / backtesting | `cross_val_score`, `TimeSeriesSplit` | `evaluate`/backtesting | No helper; external only | Incompatible; leakage risk |
| Capability declaration | `__sklearn_tags__` | `get_tags` / `capability:pred_int` | `Capabilities` dataclass | Familiar, not compatible |
| `predict_proba` | Class probabilities | Distribution object | Quantile `Forecast` alias | Semantic collision |
| Fitted-state check | trailing `_`, `requires_fit` | `check_is_fitted` | custom `NotFittedError` | Familiar, not compatible |
| Construction | Cheap, parameter-only | Cloneable estimator | `load` may create a worker; `fit` loads weights | Not estimator-shaped |

**Rule:** call nothing here "sklearn-compatible" or "sktime-compatible". The recommended surface avoids the
issue by not borrowing `fit`, `predict_proba`, or capability vocabulary at all. If a `fit`/`predict` façade is
ever added as sugar, its docstrings must state explicitly what it does *not* provide.

---

## 5. State, `fit`, and the stateless alternative

### 5.1 Stateless `predict(context, *, horizon)`

A loaded model object still exists; only the *data* is stateless:

```python
class ChronosBolt:
    def predict(self, context: ArrayLike, *, horizon: int) -> FloatArray: ...
    def close(self) -> None: ...
```

Advantages:

- No `fit`, no `NotFittedError`, no context version, no refit semantics, no one-slot cache, no per-object data
  lock, no stale-state failure mode.
- Trivially safe under concurrent calls (assuming the native model handle is safe or externally locked).
- Matches the zero-shot nature of every planned adapter.
- Makes backtesting natural: each call receives exactly the training prefix and horizon; there is no hidden
  state to clone.
- Smaller worker payload semantics if isolation is ever added: every call is self-contained.

Costs:

- The caller passes context on every call; a UI must retain it.
- No familiar `fit` workflow, and context validation/normalisation happens per call (cheap relative to
  inference).
- Model loading still needs a handle, so "stateless" means stateless with respect to data, not a free function
  that reloads weights each time.

### 5.2 Stateful `fit(y)` / `predict(h)`

Advantages:

- Familiar workflow and concise UI code.
- One place to validate/normalise/cap context; context is copied once.
- Model loading can stay lazy until `fit`.
- Enables the one-slot `(context_version, h)` cache shared by point and quantile calls.

Costs:

- `fit` is not training; the name invites the wrong mental model.
- Adds fitted/unfitted state, `NotFittedError`, context replacement, cache invalidation, per-object locking, and
  a "one object per thread" rule.
- Makes external CV/clone workflows harder, not easier.
- A second `fit` silently replaces the context; users may expect incremental training or `partial_fit`.

### 5.3 Decision

Choose the stateless form as the only public entry point (ADR 0002). The previous revision kept `fit`/`predict`
because the contract froze it and because a "familiar workflow" was treated as a product requirement; that
requirement is now explicitly relaxed. Adding a `fit` façade later is cheap if a real consumer appears; removing
one after it is public and frozen is not. Do not ship both as peer entry points.

---

## 6. Point and quantile semantics

### 6.1 Median-only public contract versus native point

A **median-only** contract means the one returned vector is the 0.5 quantile. A **native point** contract
returns whatever the model calls its point forecast and declares whether it is a mean or median.

| Policy | Pros | Cons |
|---|---|---|
| Median-only public contract (recommended) | One comparable statistic; matches the product comparison; most models expose 0.5 natively or via samples; smallest surface | A native mean is not exposed; a mean-only output cannot recover a median; sample-based models require a documented estimate |
| Native point + `point_kind` | No hidden transformation; works for mean, median, and point-only models | Cross-model comparisons are not apples-to-apples unless the consumer reads `point_kind`; adds a metadata field to a vector-returning API |
| Two entry points (`predict_median`, `predict_mean`) | Explicit semantics | Two surfaces to maintain; the second has no consumer |

The historical contract chose native point + `Capabilities.point`; the earlier revision of this note endorsed
that because it avoided transforming numbers. Under the median-first direction, "no transformation" is
replaced by "one declared statistic". Inspect all native outputs, not just the SDK's point forecast:
TimesFM 2.5 has a mean/point output **and** a quantile output containing 0.5, so it does not need sampling to
provide a median. Chronos v1 and Toto 1 instead supply forecast samples and can estimate the median from them.
A truly mean-only output establishes no median and must be excluded from this contract.

Two clarifications that the earlier revision got wrong:

- **No capability misrepresentation.** A median-only surface does not claim to expose "everything the model
  can do". It exposes a subset and makes no quantile claim, so a model with a nine- or 21-level head is not
  being misreported. Misreporting only arises under a contract that promises native quantiles when they exist
  — i.e. historical v0.3, or a later revision that keeps `Capabilities.quantiles`.
- **Naming.** The return value should be documented as the predictive median, not "the model's forecast" or
  "point forecast", because for Chronos-Bolt the SDK's own `mean` label is the 0.5 column and for TimesFM 2.5
  the point head is a mean.

### 6.2 Sample-derived medians

Several models are sample-first: Chronos v1, Moirai 1.1, Toto 1, Lag-Llama, and Sundial. ADR 0001 permits a
sample-based adapter to estimate the median from native forecast samples provided it documents the sampling
configuration and the run-to-run variation; the earlier revision forbade this, which would have excluded most
sample-first models from a median milestone for no product reason.

- An SDK's own documented empirical-quantile method (for example Chronos v1's `predict_quantiles`, which uses
  `torch.quantile`) is the preferred path: one API call, upstream semantics.
- Computing `np.median(samples)` in the adapter is a **derived** statistic: defensible, but it must be
  documented with sample count, seed, and observed variation, and it is noisier than a learned 0.5 head.
- Sample medians can be non-monotone across horizons and do not coincide with a sample mean. Do not smooth,
  sort, or reconcile them against anything else.

### 6.3 Quantiles, prediction intervals, confidence intervals (deferred)

These are different objects and none is in the median milestone (ADR 0003):

- A **future quantile** at level `q` is the `q`-quantile of the predictive distribution of
  `y_{T+1..T+h}`.
- A **prediction interval** with coverage `c` is a pair of future quantiles, usually `(1-c)/2` and `(1+c)/2`.
  sktime's guide notes that a point forecast "does not need to be median forecast aka 0.5-quantile forecast".
- A **confidence interval** is an interval for a parameter or the predictive *mean*; it reflects estimation
  uncertainty, not the dispersion of future observations, and is not obtainable from a native quantile grid.

Constraints to carry into the later quantile design, not to implement now:

- Levels are probabilities, and requested levels must be distinguished from native levels.
- Exact native-level selection with an explicit error for unavailable levels; no silent interpolation or tail
  extrapolation. Under this rule a nine-decile model satisfies `interval(0.8)` (0.1/0.9) but not
  `interval(0.9)` (no 0.05/0.95).
- Quantile crossings are native model behaviour: preserve and flag them, never repair them to make a nicer
  interval.
- Array orientation, sampling controls, and the cost of a second prediction call must be specified before any
  public probabilistic type is frozen.

---

## 7. Result shape

For the median milestone the answer is a bare array: `NDArray[np.float64]` of shape `(horizon,)`. There is no
second block to pair it with, so the value-object argument that dominated the earlier revision does not apply
yet.

Options considered, kept for the later probabilistic surface:

1. Return `np.ndarray` for points and a separate quantile array/tuple. **(Recommended now.)**
2. Return one immutable `Forecast` value object with point, optional native `levels`/`values`, and metadata.
3. Introduce `PointForecast`, `QuantileForecast`, `ProbabilisticForecast`, or a generic `Result[T]`.

Option 1 is sufficient and smallest for a single statistic. Option 3 adds exactly the wrapper hierarchy the
library does not need. If quantiles return later, option 2 is the right size — one type, optional quantile
block, provenance in one place, `quantile`/`interval` as convenience methods — and the historical `Forecast`
design is a reasonable starting point for that study. Do not introduce an adapter-specific result class at any
point.

---

## 8. Adapter typing and avoidable duplication

Under the thin direction (ADR 0002) the boundary is one callable, so the recommendation changes from the
earlier revision:

- **Concrete adapters plus at most one small structural type.** A `Protocol` (or a plain base class with a
  single `predict` method) is enough to let a test double stand in. Do not add both a `Protocol` and an ABC
  with the same methods.
- **The earlier "keep the ABC, define no Protocol" recommendation was v0.3-shaped.** It existed to enforce
  adapter class attributes (`name`, `package`, `capabilities`, `ADAPTER_API`) and an API version that the
  median milestone does not have. Revisit an ABC only when third-party adapters or discovery actually appear.
- **No capability dataclass or tag hierarchy yet.** Format limits (accepted context length, maximum horizon,
  device/dtype) are constructor or adapter attributes validated at call time, or simply documented per
  concrete adapter. They exist to fail loudly, not to be discovered.
- **No registry, plugin discovery, supervisor, or `available()`/`register()` in 0.1.** Two imported adapter
  classes with explicit names are enough; a catalog belongs to the example server.
- **No generic SDK keyword forwarding.** Native loading configuration is explicit per adapter
  (`model`, `device`, `dtype`, `revision`), so failures point at a defined parameter instead of a passthrough
  dict.
- **One result type, one validation path.** Private shared validation (1-D, finite, non-empty, positive integer
  horizon, adapter limits) is a function, not a framework.

---

## 9. Recommended minimal surface

```python
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike          # runtime accepts only 1-D numeric, finite, non-empty, within adapter limit


class ChronosBolt:
    """Zero-shot median forecasts from Chronos-Bolt, via its native SDK."""

    def __init__(
        self,
        model: str = "amazon/chronos-bolt-small",
        *,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None: ...

    def predict(self, context: ArrayLike, *, horizon: int) -> FloatArray: ...
    def close(self) -> None: ...


class TimesFM:  # illustrative second adapter
    def predict(self, context: ArrayLike, *, horizon: int) -> FloatArray: ...
    def close(self) -> None: ...
```

This block is documentation only; it is not an implementation and should not be pasted into the package.

Notes:

- The return value is the predictive median, element 0 the next observation, in the context's units.
- `ArrayLike` is broad for callers, but the runtime contract is narrower: 1-D, numeric, finite, non-empty, and
  within the adapter's accepted context length.
- `horizon` is a real positive integer, not `bool`, an array, or a sequence; this is a runtime check because
  Python's type system cannot express it. Unsupported horizons (above the adapter's native cap, or requiring
  a rollout the milestone does not support) raise explicitly rather than truncating.
- Constructor parameters carry loading choices; there is no generic `**adapter_kwargs` passthrough.
- `close()` (or a context manager) releases the native model; it is lifecycle, not forecasting semantics.

### 9.1 Historical surface, not recommended

CONTRACT v0.3 proposed `Forecaster.fit(y)`, `predict(h)`, `predict_quantiles(h) -> Forecast`, a `predict_proba`
alias, a `Capabilities` dataclass, and a worker-backed `load()`. It is retained for comparison only. It costs a
state machine, a probabilistic obligation that the first milestone does not need, an alias name that collides
with class probabilities elsewhere, and a process protocol that no demonstrated dependency conflict requires
(ADR 0004). If a later milestone needs quantiles, isolated environments, or estimator-like sugar, revisit these
pieces individually rather than restoring the whole surface.

---

## 10. Unanswered decisions

1. **Second adapter validation.** ADR 0005 proposes Timestral 5M; verify installation compatibility and CPU
   inference before committing to the pair.
2. **Median provenance.** Select native 0.5 for quantile heads (including TimesFM 2.5), document sample estimates
   for sample-first models, and exclude point-only outputs that do not establish a median.
3. **Sample-median documentation.** Minimum sample count, seed handling, and what variation is recorded for
   sample-first adapters.
4. **Horizon policy per adapter.** Cap and raise, or allow the SDK's own rollout beyond native horizon?
   `model-matrix` shows both behaviours exist natively (Chronos-Bolt rolls at 64, TimesFM recompiles).
5. **Context policy.** Silent truncation to the adapter's maximum context (native everywhere) or an explicit
   error? Truncation changes the answer silently, so it needs a stated rule.
6. **Probabilistic study scope (ADR 0003).** Which SDKs' quantile grids, sample APIs and point heads to
   compare, and what evidence is required before a quantile type is proposed.
7. **Transport schema.** Whether the example server's JSON should carry capability/limit metadata now that the
   library does not (see `design/repository-audit.md` §4).
8. **Evaluation.** Confirm that scoring/backtesting stays out of the core; if added later, it needs expanding
   windows and a horizon-aware split, not `cross_val_score` or `KFold`.
9. **Typing details.** `numpy.typing.ArrayLike` versus `Sequence[float]`; `Self`-style fluent APIs (none now);
   whether `close()` or a context manager is the documented lifecycle.

---

## 11. Sources

All URLs were fetched with `curl` on 2026-10-20.

**scikit-learn**

- `BaseEstimator` (parameters, explicit `__init__` contract, clone/get_params context):
  <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>
- `clone` (deep copy of parameters, unfitted result, `__sklearn_clone__`, `safe=False` fallback):
  <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>
- `clone` source (`get_params(deep=False)` and the `TypeError` when absent):
  <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py>
- `check_is_fitted` (trailing underscore, `__sklearn_is_fitted__`, stateless `requires_fit`):
  <https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html>
- `Tags` (`requires_fit` default `True`):
  <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html>
- `cross_val_score` (estimator implementing `fit`, `X`/`y`, CV splitting, default score):
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>
- `TimeSeriesSplit` (time-ordered splits; ordinary CV trains on future data):
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- Common pitfalls, data leakage ("never call fit on the test data"):
  <https://scikit-learn.org/stable/common_pitfalls.html>
- Developer guide, estimator tags and `check_is_fitted`:
  <https://scikit-learn.org/stable/developers/develop.html>

**sktime**

- `BaseForecaster` API reference (method list, tags, `fit`, `predict`, probabilistic methods):
  <https://www.sktime.net/docs/api-reference/sktimeforecastingbasebaseforecaster/>
- `BaseForecaster` source and docstrings (`fit(y, X=None, fh=None)`, `predict(fh=None, X=None)`,
  `predict_quantiles(..., alpha=None)`, `predict_interval(..., coverage=0.90)`,
  `predict_proba(..., marginal=True)`, `score` MAPE, `capability:pred_int`, `requires-fh-in-fit`,
  `fit_is_empty`):
  <https://github.com/sktime/sktime/blob/main/sktime/forecasting/base/_base.py>
- Tags reference (`capability:pred_int`, `fit_is_empty`, `requires-fh-in-fit`):
  <https://www.sktime.net/docs/api-reference/tags/>
- `capability:pred_int` (all four probabilistic methods; interval/quantile equivalence; normality caveats):
  <https://www.sktime.net/docs/api-reference/sktimeregistry-tagscapability--pred-int/>
- Probabilistic forecasting guide (point forecast as conditional expectation; point need not be median;
  interval forecasts versus confidence intervals on the predictive mean; `predict_interval`/`predict_quantiles`/
  `predict_var`/`predict_proba`):
  <https://www.sktime.net/docs/examples/probabilistic-forecasting/>

**Project sources**

- `README.md` — current median-first milestone, no compatibility claims.
- `CONTRACT.md` (v0.3) — historical surface, native-numbers rule, `Capabilities`, `Forecast`, product overrides;
  superseded as a direction but useful as a comparison.
- `docs/adr/0001`–`0004` — proposed median-only scope, `predict(context, *, horizon)` interface, deferred
  probabilistic surface, serving outside the core.
- `GLOSSARY.md` — median, quantile level, prediction interval, native forecast.
- `design/prior-art.md` — stateless `forecast(key, y, h)` recommendation; sklearn/sktime/Darts/GluonTS/
  NeuralForecast lessons.
- `design/model-matrix.md` — native point kinds, quantile grids, sample-based models, per-adapter limits.
