---
id: ADR-0001
status: accepted
date: "2026-10-20"
---

# Keep Forecasters a thin, univariate native-SDK library

Forecasters is a thin Python normalization layer over pretrained time-series foundation-model SDKs. One prediction receives one nonempty, finite, one-dimensional floating context, ordered from oldest to newest and treated as regularly spaced, plus a positive integer horizon. It returns the loaded model's native median by default, or requested native quantiles under [ADR 0007](0007-request-only-supported-quantiles.md). The default requires a native 0.5 level; an SDK output labelled `mean` or `point` is not a substitute. Forecasters owns input validation, representation normalization and selection of native statistics; it does not implement or train models. There is no timestamp or frequency inference.

The first milestone supports one univariate series in a zero-shot call. Batching, panels, covariates, timestamps, missing-value policies, calibration, model selection, evaluation, training and fine-tuning are outside scope. Serving is outside the core library; the optional example is [ADR 0004](0004-keep-serving-in-the-demo.md). Scope expansion revisits this owning decision; it does not add a new scope ADR for each feature.

Reject a context longer than the loaded checkpoint supports and a horizon longer than its native one-call forecast limit before inference. `ValueError` identifies the requested and supported lengths. Never keep the last N observations, interpolate, scale, smooth, clip or enable SDK rollout to make a request fit. Adapter limits are checkpoint/configuration-specific and come from the loaded native object, not catalog assumptions. Native SDK defaults remain native; Forecasters adds no forecast post-processing of its own.

The limit policy keeps comparisons attributable to the native models and keeps the library small. A request that cannot be answered natively fails visibly instead of returning a quietly altered forecast. Catalog membership is not an implementation promise. Sample-based models remain deferred under [ADR 0009](0009-defer-sample-based-forecasts.md).

The user accepted this scope and limit policy in the [submitted decision review](../../design/decision-review-2026-10-20.md). [ADR 0002](0002-minimal-predict-interface.md) owns the public call, ADR 0006 the accepted array families and returned representation, ADR 0007 quantile requests and ADR 0008 loading. This decision is not evidence that implementation or native validation has passed.

Native implementation evidence: [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py), [Timestral](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py). See the [glossary](../../GLOSSARY.md) for median and prediction-interval terms.
