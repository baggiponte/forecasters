---
id: ADR-0008
status: accepted
date: "2026-10-20"
---

# Load with concrete from_pretrained and reuse the loaded model

Public construction is `ConcreteAdapter.from_pretrained(...)`. Each concrete adapter delegates checkpoint loading to its native SDK and returns a ready adapter whose repeated `predict` calls reuse the loaded native object. Input context belongs to each prediction, not to the loading call.

Checkpoint and supported execution options are explicit, typed parts of each concrete adapter's public `from_pretrained(...)` signature; review them there. There is no generic loader, registry, plugin framework or context-capturing `fit`. Loading errors surface when loading is requested, before a valid prediction call. Private native-SDK cleanup and internal loading mechanics may change without changing this decision.

Consequences: swapping checkpoints creates a new adapter instance; prediction does not reload weights or retain previous context; optional native dependencies are imported only when the adapter needs them.
