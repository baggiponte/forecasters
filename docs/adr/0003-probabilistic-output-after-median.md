---
id: ADR-0003
status: superseded
superseded_by:
  - ADR-0007
date: "2026-10-20"
---

# Study native probabilistic output before standardizing its interface

> Superseded by the user's decision in [ADR 0007](0007-request-only-supported-quantiles.md) to request
> optional quantiles through `predict`. The native-grid and statistical distinctions below remain research
> input; the deferral and candidate separate quantile method are no longer the current direction.

The median milestone does not commit to a public quantile result type, array orientation or distribution abstraction. The next design step compares native quantile heads, sample-based forecasts and point-only outputs across supported SDKs. They expose different statistical information: a sample median is estimated, a native quantile grid is discrete, and a point mean cannot recover either one.

A later quantile contract must state level units (probabilities in `(0, 1)`), requested versus native levels, axis order, sampling controls, unsupported levels, quantile crossing and the cost of a second prediction call. For quantile heads, the default proposal is exact native-level selection and an explicit error for unavailable levels; interpolation or tail extrapolation must not be hidden. For sample-based models, quantile estimation is a separate documented operation, not selection from a native learned grid.

Do not introduce `predict_proba` as an alias for quantiles: scikit-learn uses that name for class probabilities, while sktime uses it for predictive distributions. A later `predict_quantiles` is a candidate, not a frozen interface. Quantile endpoints describe **prediction intervals** for future observations, not confidence intervals or guaranteed calibrated coverage. Pointwise intervals also do not establish simultaneous coverage over an entire future trajectory.

The existing frontend's fixed quantile positions, quartile scoring and interval controls must not dictate the library's statistical contract. Disable those features for the median milestone; do not invent bands by copying or interpolating the median.

Evidence: [scikit-learn classifier interface](https://scikit-learn.org/stable/developers/develop.html), [sktime probabilistic forecasting interface](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html), [Chronos family: sampled versus quantile forecasts](https://github.com/amazon-science/chronos-forecasting).
