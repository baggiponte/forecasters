---
id: ADR-0006
status: accepted
date: "2026-10-20"
---

# Return requested native quantiles

## Context

Quantile heads provide a discrete grid. Interpolating missing levels would introduce statistics that the checkpoint did not supply.

## Decision

Return the native median as `(H,)` by default, or exactly the requested native levels as `(H, Q)` in request order. Match floating-point representations of a level, but reject unavailable levels rather than substituting nearby probabilities.

## Consequences

One `predict` method serves point and quantile forecasts. Requests are limited to the loaded grid, and computing fewer returned columns need not reduce native inference cost. This supersedes [ADR 0003](0003-probabilistic-output-after-median.md).
