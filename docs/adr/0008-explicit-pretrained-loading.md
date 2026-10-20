---
id: ADR-0008
status: accepted
date: "2026-10-20"
---

# Load concrete adapters with from_pretrained

The user chose explicit `ConcreteAdapter.from_pretrained(...)` loading over model loading in the constructor. Each concrete adapter delegates checkpoint loading to its native SDK and returns a ready adapter whose repeated `predict` calls reuse the loaded weights. Input context belongs to each prediction, not the loading call.

Checkpoint and supported execution options must be explicit and typed per adapter. No generic loader, registry, plugin framework or context-capturing `fit` is needed for the initial pair. Loading errors surface when loading is requested, before a valid prediction call. Exact option names and cleanup mechanisms are implementation details to review against the native SDK.
