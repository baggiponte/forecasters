---
id: ADR-0001
status: accepted
date: "2026-10-20"
---

# Start with one univariate median forecast through native SDKs

> Subsequent user decisions: [ADR 0006](0006-preserve-array-backend.md) replaces the NumPy-only assumption;
> [ADR 0007](0007-request-only-supported-quantiles.md) adds optional quantile requests to the interface.
> A median-only demo can still be the first implementation milestone, but public quantiles are no longer
> wholly deferred.

Forecasters normalizes inference for one regularly spaced, finite, univariate 1-D context and a positive integer horizon. The user accepted univariate input first; batching is deferred. Default prediction returns the predictive median (quantile 0.5) in the context's original units, with shape `(H,)`. Explicit quantile requests return `(H, Q)` under ADR 0007. Values preserve the array family, floating dtype and tensor device under ADR 0006. Each adapter delegates inference and model loading to the upstream SDK; it handles validation, representation and extraction of the requested statistics, not model implementation.

The previous [contract](../../CONTRACT.md) allowed `predict` to mean either mean or median depending on the model. That is unsuitable for comparing median curves: an SDK's output named `mean`, `point` or `predictions` must be checked rather than trusted. Use a native 0.5 quantile for the selected initial pair. A point-only mean cannot silently substitute for a median. The user subsequently chose to defer sample-based model support in [ADR 0009](0009-defer-sample-based-forecasts.md).

Training, fine-tuning, panels, covariates, timestamps, missing-value handling, calibration and model selection are outside this first milestone. The library interface includes requested native quantiles, while the first demo renders only the median. Native SDK scaling is retained; Forecasters adds no scaling, smoothing, clipping, interpolation or autoregressive rollout.

**Accepted limit policy:** reject context longer than the selected checkpoint supports and horizon longer than its native forecast limit, before inference. Raise `ValueError` identifying the requested length and supported limit. Do not silently keep the last N observations or enable the SDK's longer-horizon rollout. Adapter limits must be checkpoint/configuration-specific and verified from the native implementation.

The user accepted these scope and limit choices in the submitted [decision review](../../design/decision-review-2026-10-20.md). They are not implementation evidence. [ADR 0002](0002-minimal-predict-interface.md) records the callable interface; [ADR 0007](0007-request-only-supported-quantiles.md) records optional quantile selection.

Evidence: [Chronos-Bolt native implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py), [Timestral native inference](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py). See the [glossary](../../GLOSSARY.md) for median and prediction-interval terminology.
