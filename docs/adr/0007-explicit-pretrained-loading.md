---
id: ADR-0007
status: accepted
date: "2026-10-20"
---

# Load checkpoints explicitly

## Context

Loading pretrained weights is expensive and may fail independently of a prediction request. A generic loader would also need to normalize different SDK loading options.

## Decision

Expose `from_pretrained` on each concrete adapter, with explicit SDK options, and reuse the resulting native model across predictions.

## Consequences

Loading failures occur at construction. Choosing another checkpoint requires another adapter instance; prediction calls do not reload weights.
