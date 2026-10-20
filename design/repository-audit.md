# Repository audit — median-only milestone and Line-up HTTP contract

> Subsequent user decision: [ADR 0002](../docs/adr/0002-minimal-predict-interface.md)'s direct,
> typed `predict(context, horizon=...)` interface is accepted. The other ADRs remain proposals.

**Scope.** Read-only audit. Baseline commit `21567e1` (`main`; worktree clean when the audit began). Inputs:
`README.md`, `CONTRACT.md`, `docs/adr/0001`–`0004` (plus the ADR index), `GLOSSARY.md`, `design/model-matrix.md`,
`design/interface-options.md`, `design/edge-cases.md`, `examples/lineup/static/app.js` (with the static HTML/CSS
for context), `pyproject.toml`, and the package sources. No credentials or network calls were used. Model facts
come from repository documents, not from executing an SDK.

**Revision note.** This audit was first written while CONTRACT v0.3 still appeared binding; it therefore treated
median-only as a contract violation and native quantile preservation as an obligation. The product direction has
since been reopened (median first, thin layer, native SDKs, minimal abstractions) and the ADRs are the current
proposal. Sections below keep the verified repository and frontend findings; the recommendation and the
conformance framing are corrected.

---

## 0. Current recommendation

Adopt the proposed median-first direction (ADR 0001, 0002, 0003, 0004):

1. **Core:** one in-process call, `predict(context, *, horizon=...) -> NDArray[np.float64]`, returning the
   predictive median; context passed on every call; no `fit` state, no worker protocol, no public quantile type.
2. **Adapters:** two concrete native-SDK adapters, each owning its SDK object; no registry, capability
   hierarchy, or plugin system yet.
3. **Serving:** a small example server under `examples/lineup` that implements the routes `app.js` already
   calls, sends an explicit median vector, runs blocking inference off the event loop, and coalesces superseded
   drawing requests.
4. **Frontend:** keep the canvas, drawing and model rail; remove forecast-value smoothing, disable bands and the
   pinball/coverage test, and respect each model's real horizon limit.
5. **Documents:** `CONTRACT.md` v0.3 (and its freeze/product overrides) is historical comparison, not the plan;
   README already states the median-first milestone and claims no compatibility. Do not edit ADRs from this
   audit; reconcile any remaining doc links when the ADRs are accepted.

Why the earlier "median-only misreports capabilities / fails conformance" conclusion no longer governs: it was
evaluated against CONTRACT v0.3, which required an adapter with native quantiles to publish them. Under a
median-only public contract that obligation does not exist — the surface exposes one statistic and says nothing
about quantiles, so it neither claims nor denies them. A conformance violation would only arise if v0.3 were
retained as the binding contract.

---

## 1. Repository state (observed at baseline)

| Area | State |
| --- | --- |
| Git | One commit, `main`, no tags; no tracked implementation. Proposal/research docs are untracked. |
| Package | `src/forecasters/__init__.py` only exports `__version__ = "0.1.0"`; `py.typed` is empty. |
| Public API | No `load`, `Forecaster`, `Forecast`, `Capabilities`, errors, adapters, worker, or `forecasters.testing`. |
| Tests | `tests/` is empty. |
| Packaging | `pyproject.toml` declares `numpy>=1.26` and `pytest` only; no web/server dependencies. |
| UI | Static assets under `examples/lineup/static/`; no server, no route implementation, no test. |
| Design | `CONTRACT.md` v0.3 is a historical proposal (its own header now says the direction is reopened); `README.md` describes a median-first milestone. |
| Model research | `design/model-matrix.md` covers 15 models; `chronos_bolt` is the recommended first adapter. |

The package docstring is accurate: the public API is not implemented. The UI is already copied into
`examples/lineup/static/`, and `README.md` describes its server as missing and the transport as needing a small
adaptation, so the earlier "will be ported" inconsistency no longer applies.

### 1.1 Documentation state

**Baseline findings, since corrected** (they described the files as of `21567e1` and are not current defects):

- `README.md` said "contract v0.2 under review" while `CONTRACT.md` said v0.3 — now the README states the
  research/median-first status and CONTRACT is marked historical.
- `README.md` called the release a 0.0.1 placeholder while `pyproject.toml` and `__init__.py` say 0.1.0 — the
  README no longer carries that version claim.
- `README.md` still described the demo as "will be ported later" and planned a `toto` adapter while
  `CONTRACT.md` used `toto2` — the README now describes the shipped static demo and no longer names `toto`.

**Still true, and worth fixing when the ADRs are accepted:**

- `design/edge-cases.md` assumes an existing `runtime.py`/`types.Forecast` and an older API with
  `predict_proba(levels=...)`, interpolation, crossing repair, `Superseded`, and restart-after-crash. None of
  that code exists, and those behaviours contradict both v0.3 and the ADRs. Treat it as stale design input.
- The model research and ADR 0005 were being written concurrently with this audit. Missing-file observations
  made during that interval are transient, not repository findings; see the final ADR index for their links.

### 1.2 Current proposal documents (markdown, not implementation)

- `GLOSSARY.md` defines context, horizon, median, quantile level, prediction interval, native forecast, and
  zero-shot forecasting.
- `docs/adr/0001-median-first-native-sdk-scope.md` proposes a first milestone that returns one median vector
  from native SDKs, with no training, covariates, timestamps, missing-value handling, calibration, public
  probabilistic output, or wrapper-side scaling/smoothing/interpolation. Sample-based adapters may estimate the
  median from native samples if the configuration and variation are documented.
- `docs/adr/0002-minimal-predict-interface.md` proposes direct
  `predict(context, *, horizon) -> ndarray`, one small structural type, concrete adapters, and no second ABC,
  result wrapper, registry, capability hierarchy, or generic SDK keyword forwarding for the first milestone.
- `docs/adr/0003-probabilistic-output-after-median.md` defers the public quantile surface, rejects
  `predict_proba` as an alias, and says the frontend's fixed quantile positions, scoring, and interval controls
  must not dictate the library contract.
- `docs/adr/0004-keep-serving-in-the-demo.md` keeps serving and dependency isolation outside the core, drops the
  worker/JSON-lines requirement for the first milestone, and specifies the existing UI routes in
  `examples/lineup` with an explicit median vector, no value smoothing, no bands/quantile scoring, real horizon
  limits, and request coalescing.
- `design/interface-options.md` (revised with this audit) now recommends the same median-only stateless
  interface, retains its verified scikit-learn/sktime evidence, and records which of its earlier
  recommendations were shaped by the v0.3 freeze.

These documents supersede CONTRACT v0.3 as a direction. They disagree with it on scope (one vector versus a
typed probabilistic `Forecast`), interface (`predict(context, *, horizon)` versus `fit(y).predict(h)`), and
isolation (in-process versus mandatory worker). Those disagreements are deliberate product decisions; the old
recommendations were conditioned on a freeze that has been reopened, not on evidence that changed.

---

## 2. Is the first milestone feasible as median-only?

### 2.1 What the models can supply (observed)

- `design/model-matrix.md:38-49` records that Chronos-Bolt's native `predict` returns `(batch, 9, H)`, that the
  returned "mean" is actually the 0.5 column, and that the nine levels span 0.1–0.9. The native call therefore
  already contains the median; extracting it is one column selection.
- The same document notes that most candidates expose a median column, and that sample-first models (Chronos v1,
  Moirai 1.1, Toto 1, Lag-Llama, Sundial) expose samples from which a median is estimated.
- TimesFM 2.5's native point/mean output must not be used as a median, but its quantile output includes 0.5:
  it is not a mean-only model. TTM r2/r2.1 released point-only outputs do not establish a predictive median.
  Neither family is in the planned first pair.

So a median milestone is achievable with no adapter-side interpolation and no fabricated numbers. It does mean
the public surface exposes one statistic while the SDK computes nine; that is an intentional reduction, not a
false capability claim.

### 2.2 Options under the current direction

| Option | Fit with ADRs | Milestone value | Main cost/risk |
| --- | --- | --- | --- |
| A. Median-only `predict` on Chronos-Bolt (recommended) | **Yes** | Smallest vertical slice; median is native | Native quantiles are computed and discarded; a later quantile surface needs a second review, not a rewrite |
| B. Median-only frontend over a full quantile core | Partial | Keeps future quantile work warm | Reintroduces `Forecast`/`Capabilities`/worker for a single rendered statistic |
| C. Full quantile surface now | No (ADR 0003 defers it) | Target end state | Level/grid semantics, UI mapping, and a frozen type before any evidence |
| D. Point-only `naive` adapter as the only milestone | Yes but weak | Exercises validation without an SDK | Proves nothing about native loading/inference |

A public median-only `Forecaster` layered on CONTRACT v0.3 without a revision would still be incoherent — that
specific criticism stands, but it describes a hybrid we are no longer proposing.

### 2.3 Recommendation

Option A. Implement the median extraction for Chronos-Bolt against the native SDK, add a second adapter that
also exposes a median (ADR 0005 proposes Timestral 5M), and keep the probabilistic question in ADR 0003's
study. If the SDK's median path later proves absent for a wanted model, prefer excluding the model or
documenting a sample-median estimate over widening the public surface.

---

## 3. Existing UI HTTP/WebSocket contract (observed only in `app.js`)

There is no server implementation or schema document. The following is the de facto contract the frontend
requires.

### 3.1 `GET /api/models`

- `fetch("/api/models", { cache: "no-store" })`; any non-2xx is a failure (`app.js:183-188`).
- JSON body: `{ models: Model[] }`.
- Required `Model` fields used by the UI:

| Field | Use |
| --- | --- |
| `key` | Stable id; used in `/api/models/:key/load`, forecast requests, response matching, and the `toto` smoothing special case. |
| `name`, `org`, `variant` | Rail display (`app.js:244-247`). |
| `color` | Hex color for the model swatch and forecast lines. |
| `note`, `url` | Tooltip text (`app.js:248`). |
| `default_on` | Initial on/off state (`app.js:190-193`, `260-265`). |
| `status` | One of `ready`, `loading`, `idle`, `missing`, `error`; unknown values render as "retry" (`app.js:222-228`). |
| `load_ms?` | Used in the polling signature (`app.js:203`). |
| `error?` | Error tooltip/status (`app.js:248`, `322-324`). |

- Polling: every 900 ms while a model is `loading` or the client has requested a load; 500 ms after a load
  request; 2.5 s after a fetch failure; one extra boot retry at 1.5 s (`app.js:211-218`, `304-305`, `1096`).
- The UI has no capability, horizon, context, level, or version metadata. It assumes all ready models can
  answer the same forecast request.

### 3.2 `POST /api/models/:key/load`

- `fetch(..., { method: "POST" })`; no request body (`app.js:298-299`).
- Response body is ignored; only non-2xx matters.
- Expected state transition is visible through `GET /api/models`: `idle|error` -> `loading` -> `ready|error`.
- `error` rows are retried by calling this endpoint again (`app.js:321`). `missing` rows are not loadable
  (`app.js:322-324`).
- There is no unload endpoint; switching a model off is client-side only (`S.off`, `app.js:326-327`).

### 3.3 `POST /api/forecast`

Used only for the idle example (`app.js:459-480`).

Request:

```json
{ "y": [0.12, 0.15, "..."], "h": 77, "models": ["chronos_bolt"] }
```

- `y` is the normalized 0..1 example series; it is not rounded here.
- `h` is `STEPS - 1 - t0` (`STEPS=160`), e.g. 77 for the built-in example.
- No context limit is applied by this call.

Response: 2xx with

```json
{ "results": { "chronos_bolt": { "q": [[...], ["..."]], "ms": 12.3 } } }
```

or `{ "results": { "key": { "error": "..." } } }`. The client silently skips an entry with `error` or no
`msg` (`app.js:476-479`). Failure of the HTTP request is silently ignored because the example is decoration
(`app.js:484`).

### 3.4 WebSocket `/ws`

Client connection and recovery (`app.js:344-373`):

- URL is `ws://` or `wss://` + `location.host` + `/ws`.
- No auth, subprotocol, version, or initial hello. Reconnect is every 1200 ms after close.
- Text or binary frames are accepted; binary is decoded as UTF-8.

Client -> server (`app.js:428-450`):

```json
{ "type": "forecast", "seq": 17, "y": [0.1, "..."], "h": 42, "models": ["chronos_bolt"] }
```

- `seq` is a monotonically increasing client integer.
- `y` is the last `MAX_CONTEXT=160` values, rounded to 5 decimals. Values are normalized 0..1 drawing
  coordinates.
- `h = min(160, 159 - tip)` and must be at least 4. So the client can request horizons far above a model's
  native cap.
- Requests are throttled to at most one per 85 ms unless `final=true` (`app.js:432-433`).
- The client keeps only the last 40 `seq` entries (`app.js:440`).

Server -> client (`app.js:375-390`):

```json
{ "type": "forecast", "key": "chronos_bolt", "seq": 17, "q": [[...], ["..."]], "ms": 12.3 }
```

or

```json
{ "type": "error", "key": "chronos_bolt", "error": "message" }
```

- One `forecast` message per model; responses may arrive out of order.
- `seq` must match an outstanding client request; `key` must be a model the client still has on.
- The client keeps the newest `seq` per model and ignores older ones (`app.js:382-389`).
- There is no completion, cancellation, capability, levels, or error-code message.
- Errors are reduced to the first line of text (`app.js:376-379`).

### 3.5 Hidden quantile-matrix assumptions

`msg.q` is not a generic quantile object. The UI treats it positionally:

- `q[0].length` is the horizon (`app.js:540`, `552`, `666`, `918`).
- `q[3]` is the median (`drawMedian`, `app.js:555`; score MAE, `app.js:921`).
- `q[1]` and `q[5]` are the 80% interval for bands (`app.js:607`, `671`); `q[2]`/`q[4]` are an inner band
  when focused (`app.js:608`).
- The test score assumes rows 1–5 are exactly `[0.1, 0.25, 0.5, 0.75, 0.9]` (`app.js:908-925`).
- The seven-level constant at `app.js:12` is **unused**; there is no runtime check that the matrix matches
  those levels.

The de facto requirement is therefore at least six equal-length rows with rows 1–5 mapped to
0.1/0.25/0.5/0.75/0.9. Row 0 is used only for length; row 6, if present, is unused. Passing a native
nine-decile matrix through unchanged would put the median at row 4, not row 3, and would score/render it
incorrectly.

### 3.6 Static serving

`index.html` requests `/static/style.css`, `/static/app.js`, and `/static/fonts/*`; the root document must be
served from `/`. No server code exists to do this.

---

## 4. Gaps between the UI and the median-first library

| # | Gap | Consequence |
| --- | --- | --- |
| G1 | **No server exists.** `CONTRACT.md` excludes serving as well. | Every endpoint above must be written in `examples/lineup`, not the library. There is no HTTP schema or transport test. |
| G2 | **UI requires a quantile matrix; the library returns a median vector.** | `drawBand`, `drawMedian`'s `q[3]`, preview loops and the pinball score all expect `q`. The renderer needs a median-only mode (ADR 0004). |
| G3 | **UI indexes rows; the library will return one vector.** | Either the transport wraps the median in a single-row `q` and the renderer stops reading `q[3]`, or the renderer switches to a `median` field. Do not fake rows. |
| G4 | **UI's 0.25/0.75 requirement.** | Those rows only matter for bands and the pinball score, both disabled for the milestone; they stop being a blocker once quantiles are deferred. |
| G5 | **Horizon mismatch.** | UI can request up to 151 steps (`app.js:436`); Chronos-Bolt's native horizon is 64 (`design/model-matrix.md:46`). The server must clamp the request to each model's real limit, or expose limits and let the UI ask less; silently truncating forecasts is not acceptable. |
| G6 | **Preload has no library API.** | The UI loads models before any drawing. With direct `predict`, the load endpoint simply constructs the adapter (no dummy `fit` context needed), which the current interface makes straightforward. |
| G7 | **No runtime status/load metadata API.** | The UI wants `status`, `load_ms`, and `error`; the example server owns this state. With one in-process adapter per model this is a dict plus a lock, not a supervisor. |
| G8 | **Terminal failures vs UI retry.** | A failed or crashed in-process adapter must be discarded and reconstructed on retry; the library will not restart anything itself. |
| G9 | **No cancellation/supersede.** | The UI sends a forecast every ~85 ms while drawing and ignores stale results client-side. Inference is serialized in the example server, so it needs its own coalescing/drop policy to avoid an unbounded queue. |
| G10 | **Model-specific statistics.** | With a median-only contract every adapter must return 0.5. TimesFM 2.5 requires selecting its quantile output rather than the separate mean/point output; point-only models without an established median (for example TTM r2) cannot be relabelled as median providers. |
| G11 | **HTTP error/status schema is absent.** | No structured error codes, no partial-result status, no version. UI silently drops example errors and flattens live errors to text. |
| G12 | **Model catalog metadata is UI-specific.** | `color`, `org`, `variant`, `note`, `url`, `default_on` are not adapter attributes; the example server maintains its own catalog keyed by adapter name. |
| G13 | **v0.3-only requirements are gone, not in conflict.** | No core worker/JSON-lines isolation is proposed (ADR 0004); README step 2's "type against the ADRs" replaces the earlier worker-based sequence. Nothing to resolve before coding, only to document. |
| G14 | **One source of truth is still needed.** | CONTRACT v0.3 (historical), the ADRs (proposal), and `design/interface-options.md` (revised research) must not be read as competing plans. The ADRs govern; CONTRACT needs a "superseded by" pointer once accepted. |

---

## 5. Concrete recommendations and open decisions

1. Accept or amend ADR 0001–0004 first; then add a one-line "superseded by ADRs" pointer to `CONTRACT.md` so
   the old freeze cannot be mistaken for the plan. Do not edit ADRs from an audit.
2. Implement the median adapter path: native 0.5 selection for Chronos-Bolt, explicit validation of context and
   horizon limits, no smoothing, no interpolation, no fabricated rows.
3. Add the second adapter before generalizing; only then decide whether a shared structural type earns its
   place (ADR 0002/0005).
4. Adapt the frontend minimally: send/receive one median vector, remove value smoothing, disable Bands and the
   pinball/coverage test, clamp or reject horizons per model, and keep the existing error states.
5. Make the example server own load status, timing, catalog metadata and request coalescing; run blocking SDK
   inference off the event loop.
6. Leave the probabilistic transport schema out of the milestone; when ADR 0003's study lands, add `levels`
   (requested versus native) and map by exact level, never by row index.
7. Verify all local document links once concurrent research is finished; mark `design/edge-cases.md` as historical
   rather than treating its unimplemented types as current facts.

**Open decisions for the lead:**

- Which second model validates the interface (similar size, CPU-feasible, native median)?
- Context longer than an adapter's limit: truncate to the last N points (native behaviour) or reject?
- Horizon above the native cap: reject, or allow Chronos-Bolt's own quantile-rollout beyond 64?
- Point-only means cannot recover medians. Sample-based models can estimate medians from their native samples;
  they are a separate case, not a remedy for a mean-only output.
- Does the canvas need a visible "median only" label so users do not read the curve as a mean forecast?
- When does the quantile study (ADR 0003) start, and what evidence gates a public probabilistic type?
