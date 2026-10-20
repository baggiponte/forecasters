# Forecasters architecture decisions

These ADRs are the authority for accepted product and architecture choices, dated 2026-10-20. ADR 0003 is superseded by ADR 0007; all others are accepted. Implementation evidence and historical research live outside this directory; acceptance is not evidence that a check passed.

## Route by change

| ADR | Status | Read when changing or questioning |
| --- | --- | --- |
| [0001 — thin native-SDK scope](0001-median-first-native-sdk-scope.md) | accepted | Product scope, univariate/zero-shot support, context/horizon limits, rollout or truncation |
| [0002 — direct predict](0002-minimal-predict-interface.md) | accepted | Call shape, stateless `predict`, protocol, typing, scikit-learn non-goals |
| [0003 — old probabilistic deferral](0003-probabilistic-output-after-median.md) | superseded | Historical context only; current quantile decision is 0007 |
| [0004 — serving in the example](0004-keep-serving-in-the-demo.md) | accepted | Core/demo boundary, FastAPI example authorization, transport behavior |
| [0005 — two checkpoints](0005-validate-with-two-small-models.md) | accepted | Initial model pair, common demo horizon, catalog scope |
| [0006 — array families](0006-preserve-array-backend.md) | accepted | Context/result families, dtype/device preservation, subclass boundary |
| [0007 — native quantiles](0007-request-only-supported-quantiles.md) | accepted | Quantile arguments, grid membership, output layout |
| [0008 — from_pretrained](0008-explicit-pretrained-loading.md) | accepted | Loading, loaded-model reuse, construction surface |
| [0009 — deferred samples](0009-defer-sample-based-forecasts.md) | accepted | Sample-based models, sampling options, empirical quantiles |
| [0010 — Array API helpers](0010-standard-array-api-helpers.md) | accepted | Shared array helpers, backend scope, native conversion |
| [0011 — earned abstractions](0011-abstractions-that-earn-their-place.md) | accepted | Adding/removing an abstraction, helper or public interface |
| [0012 — useful tests](0012-useful-tests.md) | accepted | Adding/removing tests, deciding what to verify manually |
| [0013 — native prediction intervals](0013-native-prediction-intervals-in-lineup.md) | accepted | Optional Line-up bands, native endpoints and prediction-interval semantics |
| [0014 — local marimo AnyWidget](0014-local-marimo-anywidget-lineup.md) | accepted | Notebook host, shared canvas, widget state/lifecycle and local-first scope |

## Metadata convention

Use YAML frontmatter with stable fields: `id` (`ADR-NNNN`, matching the filename), atomic `status` (`proposed`, `accepted`, `deprecated` or `superseded`) and quoted ISO `date` (`YYYY-MM-DD`). Each ID is unique. Only a superseded ADR has `superseded_by`, a nonempty list of existing ADR IDs:

```yaml
---
id: ADR-0003
status: superseded
date: "2026-10-20"
superseded_by:
  - ADR-0007
---
```

Do not encode references or explanations in `status`, add reverse metadata fields, or use filenames or titles as IDs. A replacement reference must point to a different existing ADR and must not form a cycle. Full replacement marks the old ADR superseded; a clarification to an otherwise accepted ADR stays in its text with links. No extra status vocabulary or metadata tooling is needed.

## Evidence and history

- [Implementation validation](../../design/implementation-validation.md) for versions, commands, counts, parity results and timings.
- [Submitted decision review](../../design/decision-review-2026-10-20.md) for historical form answers and later corrections.
- [Repository audit](../../design/repository-audit.md), [interface comparison](../../design/interface-options.md) and [model research](../../design/research-summary-2026-10-20.md) for research and frontend context.
- [Glossary](../../GLOSSARY.md) for domain terms.
- [`CONTRACT.md`](../../CONTRACT.md) and the external reviews are historical proposals, not binding decisions.
