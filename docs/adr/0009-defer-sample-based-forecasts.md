---
id: ADR-0009
status: accepted
date: "2026-10-20"
---

# Defer sample-based forecasting adapters

Initial adapters use native quantile heads, including the native 0.5 level for the default median. They do not add forecast-sample generation, sample-count or seed options, or empirical quantile estimation to the public interface.

Models whose predictive quantiles must be estimated from sampled future trajectories need a separate later design pass covering sampling configuration, reproducibility, cost and statistical semantics. This keeps the first pair and its fixed-grid validation small without declaring those models permanently unsupported. The earlier open question is resolved: sample-based models remain deferred, not rejected permanently.
