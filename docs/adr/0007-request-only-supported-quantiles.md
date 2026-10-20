---
id: ADR-0007
status: accepted
date: "2026-10-20"
---

# Request exact native quantiles and default to the median

Use one `predict(context, *, horizon, quantile_levels: Sequence[float] | None = None)` operation. With no quantile argument, return the native 0.5 median with shape `(H,)`. With supplied levels, return `(H, Q)` in the caller's requested order, including `(H, 1)` for one level. Return exactly the requested columns and do not insert a median column implicitly. Do not add a parallel public `predict_quantiles` or `predict_proba` method.

`Sequence[float]` applies only to `quantile_levels`; `context` remains restricted to NumPy arrays and Torch tensors under [ADR 0006](0006-preserve-array-backend.md). This supersedes ADR 0003's deferral of the public quantile interface.

Each concrete adapter validates requests against the levels actually supplied by the loaded checkpoint before native inference. Exact native-grid membership is required: do not interpolate, extrapolate, clamp or substitute a nearby level. A level absent from the loaded grid is unsupported even if it is close to an available probability. Matching may tolerate numerical representation differences such as float32's encoding of 0.1, but that tolerance must not turn an unavailable statistical level into an available one.

Reject empty sequences, duplicates, nonnumeric or boolean entries, nonfinite levels and probabilities outside `(0, 1)` before inference. A `ValueError` should identify the checkpoint, unsupported requested levels and supported native levels. Adapters expose their supported levels as one typed immutable tuple of probabilities; this does not require a capability hierarchy or registry. A nine-level grid cannot fulfil 21 distinct levels, and 0.25 is unsupported if absent even when the request count is small. Verify that output count and axis order match the request, and that the default median agrees with an explicit request for only 0.5 on the same deterministic native result. Native SDKs may compute a full grid even for a point-only call; request semantics do not promise reduced native compute.

The user confirmed the `(H,)` and `(H, Q)` layout. Sample-based adapters do not have a fixed learned grid; their eventual empirical-quantile semantics remain deferred under [ADR 0009](0009-defer-sample-based-forecasts.md).
