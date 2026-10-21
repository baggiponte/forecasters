---
id: ADR-0005
status: accepted
date: "2026-10-20"
---

# Preserve the caller's array representation

## Context

NumPy callers should not need Torch in their application code. Torch callers may need forecasts on the same device as their other tensors.

## Decision

Accept floating NumPy arrays and Torch tensors, preserving their family and dtype and the Torch input device. Keep Torch optional until inference is requested.

## Consequences

Adapters own SDK conversion and output restoration. Returning the original dtype restores representation, not precision lost during native computation; copying and device transfers may be needed.
