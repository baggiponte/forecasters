---
id: ADR-0007
status: accepted
date: "2026-10-20"
---

# Request optional quantiles through predict and reject unsupported levels

Use a single `predict(context, *, horizon, quantile_levels: Sequence[float] | None = None)` operation. With no quantile argument, the caller receives only the point forecast; retain the previously chosen predictive median as its meaning. Supplying a sequence requests those quantiles. Do not add a parallel public `predict_quantiles` or `predict_proba` method. This supersedes ADR 0003's deferral of the public quantile interface. ADR 0006 records the user's clarified choice of one array/tensor directly, not a t0-style result object.

`Sequence[float]` applies to `quantile_levels` only. The `context` parameter accepts NumPy arrays and Torch tensors, never Python sequences, under the corrected ADR 0006.

Each concrete adapter must validate requests against the levels actually supplied by the selected checkpoint/configuration, before native inference. The contract uses exact native quantile selection, with no implicit interpolation, extrapolation, clamping or nearest-level replacement. A nine-level head cannot accept 21 distinct levels; equally, three requested levels can be unsupported if one is absent from its grid. For example, `[0.1, 0.25, 0.9]` is invalid for a head supporting only `0.1, 0.2, ..., 0.9`.

Reject empty sequences, duplicates, nonnumeric/bool entries, nonfinite levels and probabilities outside `(0, 1)`. Keep the caller's requested order in the quantile axis. Matching may tolerate numerical representation differences such as float32's encoding of 0.1, but must never turn an unavailable statistical level into a nearby available one. A `ValueError` should identify the model/checkpoint, unsupported requested levels and supported levels. Invalid requests must not run native inference.

Adapters may expose their supported levels as one typed immutable tuple of probabilities; this does not require a capability hierarchy or registry. Verify that output quantile count and axis order match the accepted request, and that the default point forecast agrees with a request for only 0.5 on the same deterministic native result. Native SDKs may compute a full grid internally even for point-only calls; request semantics do not promise reduced native compute.

The finite-grid rule applies to quantile-head adapters. Sample-based adapters do not have a learned nine- or 21-level grid: any later support must explicitly document empirical quantile estimation and sampling configuration rather than inventing a fixed maximum. [ADR 0009](0009-defer-sample-based-forecasts.md) explicitly defers this support.

**Accepted layout:** following the user's “ottimo” in response to the proposed shapes, for one univariate context return `(H,)` by default and `(H, Q)` when quantile levels are supplied, including `(H, 1)` for one requested level. Return exactly the requested columns, with no hidden extra point column; request 0.5 explicitly to include the median among them.
