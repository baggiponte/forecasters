---
id: ADR-0002
status: accepted
date: "2026-10-20"
---

# Pass context to predict and keep SDK objects behind typed adapters

> Read with accepted [ADR 0006](0006-preserve-array-backend.md) and [ADR 0007](0007-request-only-supported-quantiles.md):
> context accepts NumPy arrays and Torch tensors only, output preserves that array family, and optional
> quantile levels accept a Python sequence. These refine the direct prediction operation without adding `fit`.

For the first zero-shot milestone, use `model.predict(context, *, horizon, quantile_levels=None)`: context is passed on every call and pretrained weights are loaded once per model instance via concrete `from_pretrained` calls (ADR 0008). Context is a one-dimensional NumPy array or Torch tensor, never a Python list or tuple. Default output is a median vector `(H,)`; requested quantiles produce `(H, Q)` in the requested order. Typed overloads preserve the input array family; ADR 0006 specifies floating dtype and device preservation. Runtime validation enforces dimensions, finiteness, input types and checkpoint limits.

One small structural `Forecaster` protocol describes this operation. Concrete adapters satisfy it and own their native SDK object. No second adapter ABC, public result wrapper, registry, capability hierarchy or generic SDK keyword forwarding is needed for two concrete implementations. Common validation may be a private function; optional SDKs are imported only when their adapter is requested. Native loading configuration belongs on concrete adapters, with explicit typed parameters for supported checkpoint and device choices.

The alternative `fit(y).predict(horizon)` is familiar to forecasting users but stores mutable history without fitting anything. A real scikit-learn estimator uses `fit(X, y)` and `predict(X)` with sample-wise semantics, parameter introspection, cloning and estimator checks; calling an integer horizon `X` would not make this contract compatible with `Pipeline` or `GridSearchCV`. We therefore do not advertise scikit-learn compatibility. If compatibility is a requirement now, this ADR needs revision before implementation: choose and test a concrete integration use case, or adopt sktime's forecasting contract and its dependency cost.

**User decision, 2026-10-20:** accepted direct, typed, minimal `predict(context, horizon=...)` over context-capturing `fit(y).predict(horizon)` or immediate scikit-learn integration. This deliberately revisits the earlier contract rather than adding another layer on top of it. Acceptance of this interface is not evidence that an implementation exists.

Evidence: [scikit-learn estimator development contract](https://scikit-learn.org/stable/developers/develop.html), [sktime BaseForecaster](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html). The research comparison is recorded in [interface options](../../design/interface-options.md).

The direct form is also familiar from native forecasting SDKs: [The Forecasting Company's t0 quickstart](https://huggingface.co/theforecastingcompany/t0-alpha) uses `predict(context, horizon=...)`. Familiarity here comes from a meaningful operation rather than matching method names to an unrelated estimator protocol.
