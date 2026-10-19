# Edge cases and invariants — the minimal `forecasters` contract

**Status:** design input; no implementation. Target API:

```python
f = Forecaster(adapter="chronos_bolt", python=".venvs/chronos/bin/python")
f.fit(y)                                            # -> self
point = f.predict(h)                                # np.ndarray, shape (h,)
proba = f.predict_proba(h, levels=[0.1, 0.5, 0.9])  # quantile object
```

`levels=` is assumed on `predict_proba` because non-native levels must be handleable (D2). Rows have stable
IDs that §7 turns into tests. Out of scope: full multivariate/exog schemas, non-quantile distributions.

## 0. Assumptions and conflicts

- **Name collision.** `runtime.py` already defines `Forecaster` as a *pool* over `Registry` with
  `.forecast(key, y, h)`; the single-model `Forecaster` cannot share the name. `types.Forecast` is levels-major
  `(levels, horizon)` — reuse it or rename (D1).
- Zero-shot: `fit(y)` validates and stores context only; model load is lazy (D3).
- Capabilities are static class attributes, answerable without a checkpoint; `native_levels` may be `None`
  until load (C-06).
- The worker is a subprocess speaking the JSON-lines protocol (`PROTOCOL_VERSION == 1`).
- Errors subclass `ForecasterError`; the sketch has `WorkerError`, `Timeout`, `ModelUnavailable`, `Superseded`.
  Proposed additions in D10.

## 1. Input edge cases

| ID | Case | Required behaviour |
|---|---|---|
| I-01 | 1-D `y` | Accepted; output `(h,)`. |
| I-02 | `(n, 1)` | Squeezed; output `(h,)`. |
| I-03 | `(n, k)`, `k > 1`, univariate adapter | `UnsupportedCapability`; never flatten or take column 0. |
| I-04 | empty (`[]`, `(0,)`) | `ValueError` in `fit`, before any worker exists. |
| I-05 | `len < min_context` (including length 1) | Accepted at API level; `min_context <= 1` forecasts, else `ContextTooShort` (message names both values). |
| I-06 | constant series | Accepted; finite points; zero-width bands allowed (O-05); no NaN from normalisation. |
| I-07 | `NaN` / `±inf` | `ValueError` by default, even for missing-value adapters (D9). |
| I-08 | int/float32/float64/list/tuple/`Series` | Accepted; float32 transport, float64 results. |
| I-09 | object, string, complex, bool | `TypeError`; no silent parsing (`["1","2"]` must not cast). |
| I-10 | non-contiguous or read-only array | Accepted; copied; caller array never mutated. |
| I-11 | `h` is `0`, negative, `2.0`, or `True` | `ValueError`/`TypeError`; `True` is not `1`. |
| I-12 | `h > max_horizon` | `HorizonTooLong` naming the native max; no truncation, no chunked rollout. |
| I-13 | `len(y) > max_context` | Keep newest `max_context`, warn once, `meta["context_truncated"]=True`; silent dropping forbidden (D4). |
| I-14 | missing timestamps/frequency | No frequency parameter, no inference from a pandas index; adapters needing `freq=` raise `MissingFrequency`. |
| I-15 | duplicate/unordered timestamps | Contract is positional; index documented as ignored. |

## 2. Output edge cases

| ID | Case | Required behaviour |
|---|---|---|
| O-01 | requested levels not native | Per-step linear interpolation in level space; `native_levels` retained; exactly the requested sorted levels returned. |
| O-02 | level outside native range (`0.05`; Moirai `[0.1, 0.9]`) | `LevelOutOfRange`; never clamp or extrapolate. Default-level trap (D2). |
| O-03 | levels empty/duplicated/unsorted/`<=0`/`>=1` | `ValueError` at the boundary, before dispatch. |
| O-04 | crossing (`q(0.1) > q(0.9)`) | Per-step sort repair, `meta["crossing_repaired"]=True`; strict mode may raise (D7). |
| O-05 | zero-width interval | Legal (constant input); equal arrays, not an error. |
| O-06 | `0.5` not among levels | `median` interpolates; outside native range → `LevelOutOfRange` or `None` (D6). |
| O-07 | `predict` vs `predict_proba` | `predict` = `0.5` quantile when available, else native point (usually mean); `meta["point_kind"]` records which. |
| O-08 | shape/dtype | `float64`, C-contiguous, `(h,)` (or `(h,k)` if multivariate), finite. Never `float32`/masked. |
| O-09 | rounding | None; no display rounding, no clipping to non-negative unless declared `non_negative=True`. |
| O-10 | adapter returns `< h` steps or wrong axis order | `AdapterError`; never pad or truncate silently. `> h` may be sliced (C-05). |
| O-11 | sampled adapter | Explicit `seed` (default `0`) inside the worker; same seed/context → identical output; parent RNG untouched. |
| O-12 | quantile object | Frozen, levels-major `(len(levels), h)`, `to_dict()` JSON-safe; arrays are copies. |

## 3. Lifecycle

| ID | Case | Required behaviour |
|---|---|---|
| L-01 | predict before `fit` | `NotFittedError`; worker not spawned. |
| L-02 | repeated `fit` | Last context wins atomically; no reload (one load, INV-06); equals a fresh instance on the same data. |
| L-03 | `fit` fails after a success | Previous context remains usable (transactional); document (D3). |
| L-04 | concurrent `predict`/`predict_proba` | Serialised per instance; correct shapes for all. Latest-wins `Superseded` unacceptable here (D8). |
| L-05 | `predict_proba` on point-only adapter | `UnsupportedCapability` naming the adapter; never fabricate quantiles. |
| L-06 | load failure (interpreter/import/download) | `ModelUnavailable` with path and `__cause__`; sticky, no download loop. |
| L-07 | worker crash mid-request | In-flight call raises `WorkerError`; next call does exactly one restart; no silent retry. |
| L-08 | timeout | `Timeout`; worker killed; next call restarts and succeeds; no zombie. |
| L-09 | `close()`/context exit/GC | Idempotent; later calls raise `ForecasterClosed`; no live children, no `__del__` exception. |
| L-10 | `pickle.dumps(f)` | `TypeError` (live worker not serialisable); a `Forecast` pickles (D10). |
| L-11 | `os.fork()` after fit | Child detects PID mismatch, raises `ForkedProcessError`; parent unaffected. |

## 4. Capability mismatches

| ID | Case | Required behaviour |
|---|---|---|
| C-01 | no intervals | `supports_intervals == False`; `predict` works; `predict_proba` raises `UnsupportedCapability`. |
| C-02 | no multivariate/exog | `(n,k>1)` input or `exog=` raises `UnsupportedCapability`; never ignored. |
| C-03 | missing-value support | Flag exposed, default strict (I-07); opt-in only, no wrapper-side imputation. |
| C-04 | fixed native context | Pad only if `pads=True`, else `ContextTooShort`; longer is trimmed (I-13). |
| C-05 | fixed native horizon (Moirai: 192) | Forecast native horizon, slice to `h`; reject `h > 192`. |
| C-06 | capabilities before load | `f.capabilities` answers statically; `None` is never treated as `True`. |

## 5. Process and protocol

| ID | Case | Required behaviour |
|---|---|---|
| P-01 | non-JSON line / partial line then EOF | Skip noise; EOF without the matching `id` is `WorkerError`, not an indefinite wait. |
| P-02 | valid JSON, invalid schema | `ProtocolError(WorkerError)`; kill worker; never wait for timeout. |
| P-03 | version mismatch (`"protocol": 2`) | `ProtocolError` naming both versions; no retry loop. |
| P-04 | late reply for a superseded `id` | Discarded by strict `id` match; never satisfies another request. |
| P-05 | oversized request (`h=100001`, 10^7 values, 10^6 levels) | `PayloadTooLarge`/`ValueError` before serialising; cap reply frame too. |
| P-06 | two threads restart after a crash | Exactly one `Popen` and load (single-flight under lock). |
| P-07 | adapter prints to stdout | Protocol on a duplicated fd, stdout→stderr (as `worker.py` does); buffers bounded. |
| P-08 | stderr noise | Bounded deque (~40 lines), attached to errors only. |
| P-09 | shutdown / duplicate reply | Graceful `{"op":"shutdown"}`, reap; first reply per `id` wins. |
| P-10 | `NaN`/`Infinity` tokens in a reply | `ProtocolError`; `allow_nan=False` on emit, strict on parse. |

## 6. Invariants

- **INV-01** Before `fit`, outputs raise `NotFittedError`; no worker exists.
- **INV-02** Validation precedes dispatch: bad `y`, `h`, or `levels` never contacts a worker.
- **INV-03** `predict(h)` returns `float64`, C-contiguous, `(h,)`; `(h,k)` only with multivariate support.
- **INV-04** No `NaN`/`inf` leaves the wrapper.
- **INV-05** Quantiles are non-decreasing across sorted levels at every step.
- **INV-06** One model load per `Forecaster`; `meta["worker_pid"]` stable unless a documented restart (expose it).
- **INV-07** `predict_proba(h, levels=L).values.shape == (len(L), h)`; `L` sorted, strictly in `(0, 1)`.
- **INV-08** Levels are never extrapolated, clamped, or silently dropped.
- **INV-09** Every reduction — truncation, slicing, crossing repair — is observable in `meta` or a warning.
- **INV-10** Errors are `ForecasterError` subclasses with actionable messages; tracebacks go to stderr or an attribute.
- **INV-11** `close()` is idempotent, leaves no child; later calls raise `ForecasterClosed`.
- **INV-12** Stale replies never reach a newer caller.
- **INV-13** Deterministic adapter + same context/seed → identical output.

## 7. Contract tests

| # | Test | Setup | Expected |
|---|---|---|---|
| CT-01 | `empty_fit` | `fit([])` | `ValueError`; no worker (I-04) |
| CT-02 | `shape_contract` | `(64,1)` then `(64,2)` | `(4,)`, then `UnsupportedCapability` (I-02/03) |
| CT-03 | `nonfinite_rejected` | `fit([1., nan, 3.])` | `ValueError`; no dispatch (I-07) |
| CT-04 | `dtype_rejects_strings` | `fit(["1","2","3"])` | `TypeError` (I-09) |
| CT-05 | `invalid_horizon` | `predict(0)`, `(-1)`, `(True)` | `ValueError`/`TypeError` (I-11) |
| CT-06 | `horizon_beyond_native` | `max_horizon=8`; `predict(9)` | `HorizonTooLong` containing `8` (I-12) |
| CT-07 | `context_truncated_observable` | `max_context=32`; 100 points | last 32 used; flag + warning (I-13) |
| CT-08 | `short_context` | `min_context=16`; 4 points | `ContextTooShort` (I-05) |
| CT-09 | `constant_zero_width` | `fit(np.ones(32))` | finite; `interval(0.9)` lower == upper (O-05) |
| CT-10 | `level_interpolation` | native `[0.1,0.5,0.9]`, request `[0.2,0.5,0.8]` | manual interp values (O-01) |
| CT-11 | `level_outside_native` | request `[0.05]` | `LevelOutOfRange` (O-02) |
| CT-12 | `crossing_repaired` | adapter returns `[3,1,2]` | sorted per step; flag (O-04) |
| CT-13 | `short_reply_rejected` | 3 rows for `h=5` | `AdapterError`; never padded (O-10) |
| CT-14 | `predict_before_fit` | fresh instance; `predict(2)` | `NotFittedError`; no process (L-01) |
| CT-15 | `refit_no_reload` | `fit(A)`, `fit(B)`, `predict` | equals fresh-on-B; same pid (L-02) |
| CT-16 | `concurrent_predict` | 8 threads x 20 calls | all `(h,)`; no `Superseded`; one pid (L-04) |
| CT-17 | `proba_unsupported` | point-only adapter | `UnsupportedCapability` (L-05) |
| CT-18 | `load_failure_sticky` | adapter raises `ImportError`; call twice | `ModelUnavailable` twice; one boot attempt (L-06) |
| CT-19 | `crash_then_restart` | `os._exit(1)` once | `WorkerError`; next call new pid (L-07) |
| CT-20 | `timeout_kills` | sleep 2 s, `timeout=0.1` | `Timeout`; next call succeeds (L-08) |
| CT-21 | `close_idempotent` | close twice; then predict | `ForecasterClosed` after (L-09) |
| CT-22 | `serialization_split` | pickle `f`; pickle `Forecast` | `TypeError`; `Forecast` round-trips (L-10) |
| CT-23 | `stdout_noise` | adapter prints 1e5 bytes | forecast succeeds (P-07) |
| CT-24 | `malformed_line_skipped` | `garbage` then valid reply | correct result (P-01) |
| CT-25 | `version_mismatch` | ready says `{"protocol": 99}` | `ProtocolError` naming 99/1 (P-03) |
| CT-26 | `stale_reply` | answer id 2 then id 1 | each caller gets its own data (P-04) |
| CT-27 | `oversized_request` | `predict(100_001)`, `len(y)=1e7` | `PayloadTooLarge`/`ValueError` (P-05) |
| CT-28 | `fork_refused` | `os.fork()`; child predicts | `ForkedProcessError`; parent unaffected (L-11) |

## 8. Decisions for the Lead

- **D1 — naming collision.** Keep `Forecaster` for the new single-model API; rename the existing pool to
  `ModelPool`/`Zoo`; reuse `Forecast` as the proba type.
- **D2 — default levels.** `COMMON_LEVELS` (`0.05`/`0.95`) exceeds Moirai's native range. Default to
  adapter-declared levels; accept requests only inside `[min(native), max(native)]`; else `LevelOutOfRange`.
- **D3 — fit semantics.** Lazy load; `fit` stores validated context only; failed `fit` leaves the previous
  context intact.
- **D4 — context overflow.** Newest-`max_context` slice + warning + flag. Rejected: error (hostile for long
  series) and silent trim (unobservable).
- **D5 — horizon overflow.** `HorizonTooLong`. Rejected: shorter array (breaks `predict(h)`) and chunked
  rollout (modelling logic).
- **D6 — point forecast.** Median when available, else native point, tagged `point_kind`; decide whether
  `predict == predict_proba(...).median` must hold.
- **D7 — crossing.** Sort-repair + flag; `strict_quantiles=True` raises. Decide default.
- **D8 — concurrency.** Serialise per instance; latest-wins only for interactive/UI mode; never leak
  `Superseded` from `predict`.
- **D9 — missing values.** Strict reject in v1; capability flag reserved; no imputation.
- **D10 — serialization/errors.** Refuse pickling (`TypeError`) and forking (PID check) with actionable
  messages; `Forecast` pickles. Flat `ForecasterError` subclasses: `NotFittedError`, `ForecasterClosed`,
  `UnsupportedCapability`, `ContextTooShort`, `HorizonTooLong`, `LevelOutOfRange`, `PayloadTooLarge`,
  `ForkedProcessError`, plus `ProtocolError(WorkerError)`.
