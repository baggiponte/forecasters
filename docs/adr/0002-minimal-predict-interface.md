---
id: ADR-0002
status: accepted
date: "2026-10-20"
---

# Call predict directly with context, horizon and optional quantiles

For this milestone the public operation is `model.predict(context, *, horizon, quantile_levels=None)`. Context is supplied on every call, and a model loaded through its concrete `from_pretrained(...)` method is reused across calls under [ADR 0008](0008-explicit-pretrained-loading.md). There is no `fit` step and no stored prediction history. [ADR 0006](0006-preserve-array-backend.md) owns accepted context families, dtypes and returned representation; [ADR 0007](0007-request-only-supported-quantiles.md) owns quantile arguments and output layout; [ADR 0001](0001-median-first-native-sdk-scope.md) owns horizon semantics and limits.

One small structural `Forecaster` protocol describes this operation. It also exposes `supported_quantile_levels`, `max_context_length` and `max_horizon`; [ADR 0007](0007-request-only-supported-quantiles.md) owns the levels and [ADR 0001](0001-median-first-native-sdk-scope.md) the limits. Concrete adapters satisfy the protocol and own their native SDK object. Typed overloads preserve the relationship between a NumPy input and a NumPy result and between a Torch input and a Torch result. Common validation may be a private function; optional SDKs are imported only when their adapter is requested. Native loading options stay explicit and typed on each concrete adapter.

The direct method is familiar from native forecasting SDKs and describes a real stateless operation rather than imitating an unrelated estimator protocol. A second adapter ABC, public result wrapper, registry, capability hierarchy and generic SDK keyword forwarding would add surface without hiding another SDK difference; [ADR 0011](0011-abstractions-that-earn-their-place.md) records that cross-cutting rule.

The rejected alternative is `fit(y).predict(horizon)`: it captures mutable context without fitting anything. A real scikit-learn estimator uses `fit(X, y)` and `predict(X)` with sample-wise semantics, parameter introspection, cloning and estimator checks. Passing an integer horizon as `X` would not make this contract work with `Pipeline` or `GridSearchCV`, so Forecasters does not advertise scikit-learn compatibility. If that compatibility becomes a requirement, this ADR must be revised with a concrete integration case or by adopting sktime's forecasting contract and its dependency cost.

**User decision, 2026-10-20:** direct, typed, minimal `predict(context, horizon=...)` is accepted over context-capturing `fit(y).predict(horizon)` and immediate scikit-learn integration. Acceptance is not evidence that an implementation exists.

Evidence: [scikit-learn estimator contract](https://scikit-learn.org/stable/developers/develop.html), [sktime BaseForecaster](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html), and the [interface comparison](../../design/interface-options.md). The direct form also appears in [The Forecasting Company's t0 quickstart](https://huggingface.co/theforecastingcompany/t0-alpha).
