---
id: ADR-0001
status: accepted
date: "2026-10-20"
---

# Keep a thin native-SDK scope

## Context

Native SDKs differ in input limits and forecast behavior. Automatically adapting an unsupported request would make model comparisons depend on Forecasters' own heuristics.

## Decision

Support univariate zero-shot forecasts from native quantile heads. Reject contexts and horizons beyond the loaded checkpoint's limits instead of truncating history or enabling rollout.

## Consequences

Forecast values remain attributable to the model. Callers must prepare regularly spaced, finite history and choose requests within the exposed limits.
