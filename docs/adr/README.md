# Forecasters architecture decisions

The first series is dated 2026-10-20. **ADRs 0001, 0002 and 0004–0010 are accepted; ADR 0003 is superseded.** The [submitted review and subsequent corrections](../../design/decision-review-2026-10-20.md) settle architecture, dtype/device behavior, limits and loading. Context inputs are NumPy arrays or Torch tensors only; Python sequences describe quantile levels, not context. Sample-based model support is explicitly deferred. For implementation, the supervisor chose floating context dtypes only, rejecting integer/bool/complex inputs explicitly under ADR 0006 rather than introducing an unapproved conversion. These ADRs revisit the earlier [contract](../../CONTRACT.md); their acceptance is not evidence that implementation or native validation has passed. Research and implementation are delegated through `pi` to `opencode-go/deepseek-v4.1-flash`; the orchestrator writes and reviews the decisions.

A separate read-only review through the same DeepSeek model found no blocking contradiction in the five proposals. The user subsequently accepted ADR 0002's interface choice. The initial implementation has now installed both SDKs, checked real-checkpoint median/quantile parity and measured CPU latency; see the [implementation evidence](../../design/implementation-validation.md). Connecting the canvas remains a later gate.

| ADR | Decision | Main trade-off |
| --- | --- | --- |
| [0001 — accepted](0001-median-first-native-sdk-scope.md) | Univariate 1-D first; reject context/horizon over limits | Explicit validation over truncation or rollout |
| [0002 — accepted](0002-minimal-predict-interface.md) | Typed `predict(context, horizon=...)` | Explicit zero-shot input over mutable `fit(y)` state; no immediate scikit-learn integration |
| [0003 — superseded](0003-probabilistic-output-after-median.md) | Earlier deferral of public probabilistic output | Replaced by optional quantiles in ADR 0007 |
| [0004 — accepted](0004-keep-serving-in-the-demo.md) | In-process core; server deferred to the example step | Simple library first, no mandatory worker framework |
| [0005 — accepted](0005-validate-with-two-small-models.md) | Chronos-Bolt Tiny + Timestral 5M | Real variation before a generic plugin system |
| [0006 — accepted](0006-preserve-array-backend.md) | NumPy in → NumPy values; Torch in → Torch values | Preserve caller's array family across native SDK differences |
| [0007 — accepted](0007-request-only-supported-quantiles.md) | Optional quantile levels on `predict`; validate native support | Exact native levels over implicit interpolation |
| [0008 — accepted](0008-explicit-pretrained-loading.md) | Concrete `Adapter.from_pretrained(...)` | Explicit loading over constructor side effects |
| [0009 — accepted](0009-defer-sample-based-forecasts.md) | Consider sample-based adapters later | Native quantile heads first; sampling semantics deferred |
| [0010 — accepted](0010-standard-array-api-helpers.md) | Use standard Array API helpers for shared backend handling | Standard helpers over a custom array abstraction; native SDK conversion remains explicit |

## Metadata convention

Use YAML frontmatter with stable, separate fields. Required fields are `id` (string `ADR-NNNN`, matching the filename number), `status` (one of `proposed`, `accepted`, `deprecated`, `superseded`) and `date` (quoted ISO date `YYYY-MM-DD`). Never embed references or explanations in `status`.

Only a superseded ADR has `superseded_by`: a nonempty list of existing ADR IDs, even when there is one replacement. Do not use filenames, titles or free-form sentences as IDs; do not duplicate the relation in a reverse metadata field that must be maintained separately. Each ID is unique and replacement references must not point to the same ADR or form a cycle. For example:

```yaml
---
id: ADR-0003
status: superseded
date: "2026-10-20"
superseded_by:
  - ADR-0007
---
```

This relation means full replacement of a decision. Later clarifications to an otherwise accepted ADR are recorded in its text with links to related decisions; they do not falsely mark the entire ADR superseded. No extra status vocabulary, metadata registry or tooling framework is needed.

## Review decisions

The user chose direct `predict(context, horizon=...)` in ADR 0002. Context-capturing `fit(y).predict(horizon)` and immediate scikit-learn integration were considered and set aside. Merely inheriting `BaseEstimator` would not resolve forecasting semantics or temporal evaluation.

The user corrected the input/output rule: context accepts only NumPy arrays and Torch tensors.
NumPy input returns NumPy and tensor input returns a tensor. Return that array/tensor directly,
without a result object or tuple. Quantile levels accept a `Sequence[float]`. The user's subsequent “ottimo”
confirmed `(H,)` for the default point and `(H, Q)` for exactly the requested quantiles, with no implicit
median column. These choices are recorded as accepted in ADR 0007.

A later integration adapter remains possible if a real consumer needs it. The user accepted floating dtype preservation and returning tensors on their input device. The earlier rule about converting Python context sequences to float32 is withdrawn: those inputs are not supported. Model compute device is separate. The checkpoint pair has passed installation and native median/quantile parity checks, with CPU timings recorded in the implementation evidence. The server is explicitly deferred for now, with FastAPI mentioned as a possible simple later implementation, not a framework requirement.

The first demo renders only the median, even though the library's accepted interface also supports requested native quantiles. The selected pair does not require sample-based inference. ADR 0009 explicitly defers those adapters to a later design pass.

## Implementation review gates

- Each native adapter must match the SDK's 0.5 output on the same context/horizon and chosen dtype tolerance; shape-only tests are insufficient.
- Invalid inputs and unsupported horizons fail explicitly; two distinct contexts do not leak mutable history across requests.
- NumPy inputs return NumPy forecast values and PyTorch inputs return PyTorch forecast values, with overloads preserving this relationship in static typing.
- Floating NumPy/Torch inputs preserve dtype, and tensor outputs return on the original input device. Reject Python context sequences before coercion; sequences remain valid for quantile levels. Test native compute and output conversion as separate concerns.
- Overlong contexts and horizons raise before inference; loading via `from_pretrained` is explicit and repeated predictions reuse the loaded weights.
- Unsupported quantile levels fail before inference, including requests whose count is small enough but whose levels are unavailable. Output level order matches the request; a requested 0.5 agrees with the separate median.
- Both real checkpoints load and run on the target CPU in a recorded environment; cold load and warm prediction timings are measured separately.
- The canvas sends the actual drawn series and model selection, and settled curves equal returned median values. Missing dependencies and failed models remain visible as errors.
- SDK inference is delegated to DeepSeek after ADR review; the orchestrator reviews behavior and scope before accepting it.

See [repository audit](../../design/repository-audit.md), [model research](../../design/research-summary-2026-10-20.md), [interface comparison](../../design/interface-options.md), and [glossary](../../GLOSSARY.md).
