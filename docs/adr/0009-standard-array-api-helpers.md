---
id: ADR-0009
status: accepted
date: "2026-10-20"
---

# Use standard array interoperability helpers

## Context

The two adapters share array recognition, finiteness and device handling. A custom backend abstraction would duplicate behavior already provided by a small maintained library.

## Decision

Use `array-api-compat` helpers while keeping public inputs limited to NumPy and Torch.

## Consequences

This adds a core dependency but avoids maintaining a backend hierarchy. SDK-specific normalization remains explicit; accepting other array libraries requires a separate decision.
