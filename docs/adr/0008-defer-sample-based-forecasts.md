---
id: ADR-0008
status: accepted
date: "2026-10-20"
---

# Defer sample-based adapters

## Context

Empirical quantiles depend on sampling count, seed and computational budget, unlike a checkpoint's fixed quantile head.

## Decision

Keep the initial interface limited to native quantile heads. Defer adapters that estimate quantiles from sampled trajectories.

## Consequences

The API needs no sampling configuration yet. Supporting these models later requires an explicit decision about reproducibility, cost and statistical semantics.
