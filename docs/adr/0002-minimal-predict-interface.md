---
id: ADR-0002
status: accepted
date: "2026-10-20"
---

# Predict from an explicit context

## Context

Pretrained inference needs no fitting. A `fit(y).predict(horizon)` interface would retain mutable history while suggesting scikit-learn semantics it does not implement.

## Decision

Use `predict(context, *, horizon, quantile_levels=None)` and a structural `Forecaster` protocol. Each call supplies its own history; concrete adapters own the loaded SDK object.

## Consequences

Models can be reused across independent series without a reset step. Scikit-learn pipeline compatibility would require a separate interface decision.
