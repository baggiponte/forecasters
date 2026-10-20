# Implementation validation — first library milestone

Date: 2026-10-20. This is implementation evidence, not a performance benchmark or a model
recommendation. Accepted ADRs 0001, 0002 and 0004–0009 were treated as the design authority;
`CONTRACT.md` and the historical design research were not treated as binding implementation
instructions. This file also records the supervisor review round 1 fixes.

## Implemented surface

- `src/forecasters/__init__.py`: exports `Forecaster`, `ChronosBolt`, `Timestral`; importing
  the package is lazy with respect to PyTorch and both native SDKs.
- `src/forecasters/_forecaster.py`: one small structural `Forecaster` protocol with typed
  NumPy/Torch overloads (`numpy.typing.NDArray` plus a floating-dtype `TypeVar`), native
  quantile levels, context limit, horizon limit and `predict`.
- `src/forecasters/chronos_bolt.py`: concrete `ChronosBolt` adapter over the official
  `chronos-forecasting` / `ChronosBoltPipeline`.
- `src/forecasters/timestral.py`: concrete `Timestral` adapter over the official Mistral AI
  `timestral` package.
- `src/forecasters/_validation.py`, `_outputs.py`, `_deps.py`: private validation, native
  input normalization, output validation/restoration and lazy-import helpers.
- `tests/test_adapters_with_native_fakes.py`: native-boundary tests; requires the optional
  PyTorch test backend and is skipped per test when PyTorch is absent.
- `tests/test_validation_numpy_only.py`: NumPy-only validation and regression tests, no
  Torch or native SDK required.
- `tests/test_native_parity.py`: real-checkpoint parity tests, gated by
  `FORECASTERS_NATIVE=1`.
- `tests/test_lazy_imports.py`: verifies package import does not eagerly load the optional
  stack.
- `tests/typing_consumer.py`: mypy-only consumer checks; not collected by pytest.
- `pyproject.toml`: NumPy and `array-api-compat` are core dependencies; `chronos`,
  `timestral` and `dev` extras are separate. `py.typed` is preserved.

No ADR, historical research document, server or frontend file was edited.

## Validation environment and exact revisions

Measured on the repository's full `.venv` (uv, Python 3.12.15), macOS 26.3.2 on Apple arm64
CPU. A separate core-only `.venvs/core-dev` environment installed the project's `dev` extra
without Torch or native SDKs.

| Component | Version / revision |
| --- | --- |
| NumPy | 2.5.4 |
| `array-api-compat` | 1.15.0 |
| PyTorch | 2.14.1 |
| Transformers | 5.20.2 |
| `chronos-forecasting` | 2.3.2 (PyPI) |
| `timestral` | 1.0.0 from `https://github.com/mistralai/timestral.git` commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` |
| `amazon/chronos-bolt-tiny` snapshot | `a0e552de83495b5c28c14c71c374f3e33280b340` |
| `mistralai/Timestral-5M` snapshot | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| pytest / mypy | 9.1.1 / 2.4.0 |

The checkpoints were downloaded once into the ignored `.hf-cache/`; validation runs used
`local_files_only=True`. No credentials were read.

## Array API Compatibility evaluation

The user pointed to the Python Array API standard. NumPy 2.5.4 exposes
`__array_namespace__`; PyTorch 2.14.1 does not. `array-api-compat` was evaluated and adopted
as a small core dependency for the interoperability points where it simplifies the code:

- `is_numpy_array` / `is_torch_array` recognize the two allowed families without importing
  PyTorch eagerly in validation or output checks. Arbitrary objects exposing
  `__array_namespace__` are still rejected: the accepted context scope remains strictly
  NumPy arrays and Torch tensors.
- `array_namespace`, `device`, `xp.isdtype`, `xp.isfinite` and `xp.all` provide the Torch
  floating-dtype and finiteness checks and the post-restoration finiteness check.
- It does **not** make model runtime portable, remove native Torch conversion, or remove
  dtype/device transfer decisions. SDK conversion remains explicit; native compute device is
  configured separately; results are restored to the caller's family, dtype and device.
- `xp.asarray`/`xp.astype` alone did not solve NumPy negative-stride/read-only-to-Torch
  conversion, so NumPy native representation is normalized explicitly with
  `np.ascontiguousarray` and a copy only when the view is not writable. The caller's array is
  never mutated.
- NumPy `longdouble` is accepted. Where it is wider than float64, it is converted to float64
  for native inference with an explicit overflow check; the output is restored to
  `longdouble`. This repository's macOS environment aliases `longdouble` to float64, so the
  wider Linux path is implemented but not exercised on that host.

## Review round 1 regression fixes

1. **NumPy views and edge dtypes.** Negative-stride views and read-only arrays now work
   without mutation. Tested with NumPy float16/float32/float64/longdouble and Torch
   float64/bfloat16; caller dtype and Torch device are preserved. Native input normalization
   is separate from the returned representation.
2. **Native shape checks.** Before column selection, Chronos must return exactly
   `(1, loaded_levels, H)` and Timestral must return exactly `(1, 1, H, loaded_levels)`.
   Short horizons, short/extra quantile grids, extra batch/variate dimensions and extra
   horizon columns all raise `RuntimeError`; no unidentified columns are mapped or dropped.
   Round 2 tightened this from the earlier "at least" tail policy because both official SDKs
   return the exact shapes and extra unidentified heads cannot be mapped safely.
3. **Output finiteness/representability.** Only the selected values are checked. Selected
   native non-finite values raise `RuntimeError`; the restored dtype/device is checked again,
   so finite float32 values such as 70000 that overflow to float16 `inf` raise instead of
   being returned. NaNs in known but unrequested quantile columns are tolerated when the
   native result has the correct exact shape. Round 2 also fixed empty Torch context
   detection: it now uses the first dimension rather than the `torch.size()` method.
4. **Actionable dependency and checkpoint errors.** `import_torch` now names valid options:
   `pip install torch`, `pip install 'forecasters[chronos]'` and
   `pip install 'forecasters[timestral]'` (no invalid synthetic extra). The exact checkpoint
   id passed to `from_pretrained` is retained and used in validation and native-output errors.
5. **Typing and layering.** Public overloads use `NDArray[_FloatDType]` and `torch.Tensor`;
   no public Python-sequence/context `ArrayLike` or broad keyword surface. The duplicate
   Torch import passthroughs were removed. Runtime protocol checking was removed because the
   protocol is structural and not needed at runtime; `_from_native` remains the internal
   construction path used by both `from_pretrained` and boundary tests.
6. **NumPy-only collection.** Core-only dev collection now succeeds. Boundary tests skip
   explicitly without PyTorch; NumPy-only tests still run. Skip counts are reported separately
   and are not presented as coverage.

## Native SDK findings verified from installed source/interfaces

- **Chronos-Bolt**: `ChronosBoltPipeline` exposes `quantiles` from the loaded config and
  `model_context_length` / `model_prediction_length`. Its `predict(...)` returns
  `(batch, native_quantiles, horizon)`. The adapter calls that method once with
  `limit_prediction_length=True` and selects the median or requested native columns.
  `ChronosBoltPipeline.predict_quantiles` is deliberately not used because that SDK method
  can interpolate/extrapolate unavailable levels. Tiny checkpoint: 9 native levels
  `0.1..0.9`, context limit 2,048, native horizon 64.
- **Timestral**: `TimestralForecaster` exposes `quantile_levels`, `max_context`,
  `per_shot_horizon_patches` and `patch_size`. Its `forecast(...)` returns a `Forecast` with
  quantiles `(batch, variates, horizon, native_quantiles)`. The adapter calls `forecast` once
  and selects columns from that result. Native SDK defaults were retained for symmetric
  averaging and non-negative clipping, and parity was checked against the same defaults.
  Timestral-5M: 21 native levels `0.01, 0.05, ..., 0.99`, context limit 32,768, native
  one-block horizon 1,024 (`32 patches × 32 steps`).
- Both SDKs internally compute float32 and may move compute to their configured device. The
  adapters cast the selected result back to the caller's floating dtype and, for Torch input,
  back to the caller's device. No autograd support is promised.

## Commands and observed results

```text
$ .venv/bin/python -m pytest -o addopts='' -q
127 passed, 2 skipped             # native-gated parity tests skipped by default

$ HF_HUB_OFFLINE=1 FORECASTERS_NATIVE=1 .venv/bin/python -m pytest -o addopts='' -q
129 passed                        # includes both real-checkpoint parity tests

$ .venvs/core-dev/bin/python -m pytest -o addopts='' -q
24 passed, 105 skipped            # no Torch/native SDK; skips are explicit, not coverage

$ .venv/bin/mypy src tests/typing_consumer.py
Success: no issues found in 8 source files

$ uv build
Successfully built dist/forecasters-0.1.0.tar.gz
Successfully built dist/forecasters-0.1.0-py3-none-any.whl
```

The static consumer check verifies `NDArray[np.float64]` input infers NumPy float64 output,
`NDArray[np.float32]` infers NumPy float32 output, `torch.Tensor` infers `torch.Tensor`, and
a Python-list context is rejected by the overloads (the expected mypy `call-overload` error
is explicitly ignored; an unused ignore would fail the check).

Real parity used the frozen context
`2.5 + sin(arange(48)/5) + linspace(0, 0.4, 48)` as float64, horizon 8. For each checkpoint,
the adapter median and requested `[0.9, 0.1, 0.5]` columns were compared with exact
`assert_array_equal` against the corresponding columns of one direct official-SDK result;
default median was also compared with an explicit `[0.5]` request. Both tests passed after the
new shape and finiteness checks.

A separate minimal environment was created with only the built wheel and its core
dependencies. `import forecasters` succeeded and did not load `torch`, `chronos`, `timestral`
or `transformers`. Calling `ChronosBolt.from_pretrained()` or `Timestral.from_pretrained()`
without its SDK produced an actionable `ImportError` naming the adapter extra and
installation command.

## Measured load and warm timings (observations only)

Each adapter was measured in its own process, with checkpoint cache warm, 64-step float32 CPU
context, horizon 8 and requested levels `[0.1, 0.5, 0.9]`. Warm values are seven predictions
after one initial prediction.

| Adapter | Load (s) | First predict (s) | Warm median (s) | Warm min–max (s) |
| --- | ---: | ---: | ---: | ---: |
| ChronosBolt / chronos-bolt-tiny | 1.818 | 0.189 | 0.00244 | 0.00238–0.00297 |
| Timestral / Timestral-5M | 0.406 | 0.044 | 0.00371 | 0.00362–0.00470 |

These numbers include host- and process-specific import/cache effects. They establish that the
tiny checkpoints load and run on this CPU; they are not latency guarantees.

## Limits and open issues

- This milestone is univariate, unbatched, zero-shot native-quantile inference only. Training,
  samples, covariates, timestamps, missing-value policy, calibration, serving, CLI and
  benchmarking are out of scope.
- Chronos-Bolt and Timestral SDKs can roll forecasts beyond their native per-call horizons.
  The library rejects those requests by design rather than silently enabling rollout.
- Native shapes are exact: short or extra horizons/quantile grids, and extra batch or
  variate dimensions, are rejected. NaNs in unrequested quantile columns are tolerated only
  within the correct exact shape.
- The Timestral optional dependency is pinned to a source commit because the official package
  is installed from its Git repository. The checkpoint weights have their own license.
- Array API Compatibility is an interoperability/validation dependency only; it does not make
  the native model runtimes portable or remove GPU/CPU and dtype transfer decisions.
- No license file has been added; this remains a packaging/release task outside the code
  milestone.

## Independent supervisor verification

After DeepSeek completed the second review round, the supervisor independently reran the full offline native suite: **129 passed in 2.13 seconds**, including both real-checkpoint parity tests and accelerator device checks on this host. `mypy src tests/typing_consumer.py` passed for all eight checked source files; `git diff --check` passed. The fresh core-only installed wheel imported without loading Torch or either SDK, and both missing-SDK calls produced installation guidance. All ten ADR metadata records passed checks for status, ISO date, unique identifiers, valid replacement references and absence of replacement cycles.

Review corrections covered negative-stride/read-only NumPy contexts, selected-value and post-cast finiteness, exact native shapes, empty Torch input and output typing. This accepts the first library milestone; the canvas/server integration remains pending.

## Demo server and canvas integration — 2026-10-20

This is a separate implementation increment; the library evidence above is unchanged. ADR
0004 governed the transport. No ADR was edited.

### Implemented surface

- `examples/lineup/server.py`: one FastAPI app with `GET /`, `/static`, `GET /api/models`,
  `POST /api/models/{key}/load`, `POST /api/forecast` and `/ws`. It lazily loads the two
  adapters through `ChronosBolt.from_pretrained` and `Timestral.from_pretrained`, keeps the
  loaded adapter objects for every later call, reports per-model `idle`/`loading`/`ready`/
  `error` state and allows a failed load to be retried independently.
- Model loading and prediction run through `asyncio.to_thread`; predictions share one
  `asyncio.Lock`. The WebSocket has one worker task per connection with one in-flight
  request and one latest pending request, so rapid drawing replaces an older pending
  request instead of creating a task per pointer move.
- HTTP and WebSocket transport carry an explicit `median` vector. There is no seven-row
  duplicated quantile matrix. Requests reject non-list/empty/non-finite/non-numeric or
  boolean sample values, non-integer or nonpositive horizons, horizons above the demo
  bound, unknown/duplicate model keys, oversized contexts and oversized HTTP bodies.
  Horizon checks against a loaded adapter's native `max_horizon` return a per-model error;
  they never clip or shorten the response.
- `examples/lineup/static/app.js` and `index.html`: removed forecast-value smoothing,
  bands, test/pinball scoring, score history and their controls. Kept pointer drawing,
  presets, model toggles, overlay/lanes layout, focus emphasis and stale-response guards by
  stroke id, `seq` and an incrementing UI generation. The example and drawing requests use
  `min(selected ready max_horizon)`, which is 64 with Chronos-Bolt.
- `pyproject.toml`: `demo` extra with FastAPI, Uvicorn and WebSocket support; `dev` adds
  `httpx` and `websockets` for transport tests.
- `tests/test_demo_server.py`: fake-adapter tests for static routes, lazy loading, adapter
  reuse, load failure/retry and isolation, native horizon errors, request validation, body
  bounds, HTTP/WebSocket median parity and latest-drawing coalescing.
- `tests/test_demo_native_smoke.py`: opt-in real-checkpoint HTTP/WebSocket smoke test. It
  loads both checkpoints from `.hf-cache` with `local_files_only=True` and compares the
  returned median exactly with the same loaded adapter's `predict` result.

### Commands and observed results

```text
$ .venv/bin/python -m pytest -o addopts='' -q
153 passed, 3 skipped

$ HF_HUB_OFFLINE=1 FORECASTERS_NATIVE=1 .venv/bin/python -m pytest -o addopts='' -q
156 passed

$ .venvs/core-dev/bin/python -m pytest -o addopts='' -q
22 passed, 107 skipped

$ .venv/bin/mypy src tests/typing_consumer.py
Success: no issues found in 8 source files

$ .venv/bin/mypy examples/lineup/server.py tests/test_demo_server.py tests/test_demo_native_smoke.py
Success: no issues found in 3 source files

$ node --check examples/lineup/static/app.js
# passed

$ git diff --check
# passed
```

The demo dependencies were installed into the existing `.venv` with
`UV_CACHE_DIR="$PWD/.uvcache" uv pip install --python .venv/bin/python -e '.[demo,dev]'`.
No weights were downloaded; the checks used the existing `.hf-cache`.

### Live local launch and browser evidence

Port 8000 was already occupied by unrelated static HTTP servers on this host, so the demo
was run on a free port:

```text
$ FORECASTERS_DEMO_CACHE="$PWD/.hf-cache" FORECASTERS_DEMO_LOCAL_ONLY=1 \
    .venv/bin/python -m uvicorn examples.lineup.server:app --host 127.0.0.1 --port 8010
PID 20286 (superseded by the post-review restart below); URL http://127.0.0.1:8010/; log /tmp/forecasters-lineup.log
```

Both default models reached `ready`: Chronos-Bolt tiny loaded in about 1.50 s with native
`max_horizon` 64 and context 2,048; Timestral 5M loaded in about 0.61 s with native
`max_horizon` 1,024 and context 32,768. A real HTTP request for a 48-value context and
horizon 8 returned two finite 8-value medians (about 22.7 ms Chronos, 6.9 ms Timestral on
that request). A WebSocket request returned `forecast` messages for both keys with the same
length and `seq`.

Headless Google Chrome loaded the page against the live server and, after the idle example
completed, rendered `2 of 2 on`, both model rows as `ready`, and status `2 forecasts · 3–6 ms
· all local`. This verifies the browser's catalog/example fetch path; the supervisor can
draw manually in a visible browser as the final visual check.

### Limitations

- The demo is a single local process. It has no authentication, deployment platform,
  multi-user fairness, model unloading, cross-process workers or generic scheduler.
- The demo bounds requests at 2,048 context values, horizon 1,024, 8 selected models and a
  256 KiB HTTP body. Model-native limits remain authoritative and are reported, not clipped.
- Inference is serialized globally, including across HTTP and WebSocket callers. The
  WebSocket coalescing is per connection.
- The first UI intentionally exposes only the median. Quantiles remain a library feature;
  bands, quantile scoring and sample-based forecasts are not implemented in this UI.
- `FORECASTERS_DEMO_CACHE` is a demo convenience for the repository's existing cache layout.
  When it is unset, standard Hugging Face `HF_HOME`/`HF_HUB_CACHE` resolution is used.

## Review corrections — 2026-10-20

The supervisor review reproduced a cancellation bug and requested a small set of targeted
fixes. No ADR decision was edited by implementation; the existing supervisor-authored ADR
0004 clarification is committed exactly as provided.

### Fixes

- Cancellation now cannot release the inference lock while a native thread is still running.
  `ModelManager.predict` shields the `asyncio.to_thread` task, waits for it to finish after
  cancellation, and only then leaves the lock. A canceled WebSocket caller therefore cannot
  overlap a second caller's native inference.
- WebSocket errors now use the same stroke/generation/sequence guard as successful
  forecasts. A newer error updates the model's sequence, stores a persistent row error and
  clears that model's old curve. Older successes or errors cannot overwrite it. A successful
  result for another model leaves the failed model's row error visible.
- Preview HTTP results now expose per-model errors instead of silently skipping them.
  Selection changes immediately null the current example target before scheduling a new
  preview, so an in-flight response for the old selection is ignored.
- `_parse_context` converts huge JSON integers inside the request-validation path; overflow
  is now a `RequestError`/HTTP 400 rather than an uncaught 500.

### Test review

Two private NumPy helper tests were removed because they duplicated stronger adapter tests
for reversed/read-only contexts and non-mutation while additionally locking internal
`InputSpec`/contiguity details. The meaningful NumPy-only scope, invalid-input and lazy
import checks remain. Demo assertions that only restated loaded adapter metadata (for
example native max-horizon values) were removed; the retained demo tests focus on transport
parity, lazy load/reuse, load failure/retry isolation, validation and WebSocket coalescing.
One controlled concurrency test was added: it cancels the first blocked prediction, starts a
second caller and proves the second cannot enter native inference before the first releases.

### Current commands and results

```text
$ .venv/bin/python -m pytest -o addopts='' -q
153 passed, 3 skipped

$ HF_HUB_OFFLINE=1 FORECASTERS_NATIVE=1 .venv/bin/python -m pytest -o addopts='' -q
156 passed

$ .venvs/core-dev/bin/python -m pytest -o addopts='' -q
22 passed, 107 skipped

$ .venv/bin/mypy src tests/typing_consumer.py
Success: no issues found in 8 source files

$ .venv/bin/mypy examples/lineup/server.py tests/test_demo_server.py tests/test_demo_native_smoke.py
Success: no issues found in 3 source files

$ node --check examples/lineup/static/app.js
# passed

$ git diff --check
# passed
```

### Browser regression injection

Temporary headless-Chrome injections against the restarted live server confirmed:
older same-stroke/error sequence ignored after a newer success; newer error replaced the
curve with a persistent row error; a later success for the other model did not hide that
row error; preview targets were immediately invalidated; preview per-model errors were
visible while the other model's preview curve remained. These checks were temporary and
added no browser framework or dependency.

The root/supervisor browser independently passed a manual 66-sample drawing, exact HTTP/WS
median equality for both models at horizon 64, Lanes, toggling Chronos off, and the trend
preset with only Timestral at horizon 75, with no JavaScript errors. Screenshots:
`/private/tmp/forecasters-demo-lanes.png` and `/private/tmp/forecasters-demo.png`.

The demo process was restarted on the same free port after the fixes: PID `21512`, URL
<http://127.0.0.1:8010/>, log `/tmp/forecasters-lineup.log`. A smoke check reported both
models `ready` (Chronos-Bolt max horizon 64, Timestral max horizon 1024).
