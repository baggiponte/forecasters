---
id: ADR-0003
status: superseded
superseded_by:
  - ADR-0007
date: "2026-10-20"
---

# Superseded: defer public probabilistic output while comparing native statistics

**Historical decision, not current guidance.** The user replaced this deferral with optional native quantiles on `predict` in [ADR 0007](0007-request-only-supported-quantiles.md).

The original median milestone did not commit to a public quantile result type, array orientation or distribution abstraction. The next design step was to compare native quantile heads, sample-based forecasts and point-only outputs. Those outputs carry different statistical information: a sample median is estimated, a native quantile grid is discrete, and a point mean cannot recover either one.

The research also recorded constraints that remain true: quantile endpoints describe prediction intervals for future observations, not confidence intervals or guaranteed calibrated coverage; pointwise intervals do not establish simultaneous coverage for a whole trajectory; and the old frontend's fixed quantile positions must not dictate the library contract. [ADR 0007](0007-request-only-supported-quantiles.md) now owns the public quantile decision, and [ADR 0009](0009-defer-sample-based-forecasts.md) owns the sample-based deferral.
