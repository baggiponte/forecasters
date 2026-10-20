---
id: ADR-0009
status: accepted
date: "2026-10-20"
---

# Defer sample-based forecasting adapters

The user chose to consider sample-based models later. Initial adapters use native quantile heads, including native 0.5 for the default median; they do not add forecast-sample generation, sample-count/seed options or empirical quantile estimation to the public interface.

Models whose predictive quantiles must be estimated from sampled future trajectories need a separate later design pass covering sampling configuration, reproducibility, cost and statistical semantics. This keeps the first pair and its fixed-grid validation small without declaring those models permanently unsupported. The earlier “unsure / defer” form response is resolved by this subsequent explicit decision.
