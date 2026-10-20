---
id: ADR-0011
status: accepted
date: "2026-10-20"
---

# Keep only abstractions that hide a real SDK difference

Keep an abstraction only when it hides a real difference between native SDKs or protects a public contract that would otherwise be repeated or leak. The current justified public surface is one structural `Forecaster` protocol plus the concrete `ChronosBolt` and `Timestral` adapters. The protocol earns its place because callers get one `predict` contract plus the needed checkpoint metadata, while each adapter hides its SDK's actual shape and conversion: Chronos returns `(1, Q, H)` from `predict`, while Timestral returns `(1, 1, H, Q)` from `forecast`.

Private helpers for validation, native input normalization, output shape/finiteness/restoration and lazy optional imports are also justified when each centralizes an invariant with a real failure mode. A helper used at one boundary may still earn its place if it isolates native representation work; this is not a rule against justified internal helpers. Internal helpers stay private and carry no public compatibility promise.

The public method stays `predict` because that is the familiar native-SDK operation: direct, typed and stateless with respect to history. It does not pretend to be a scikit-learn estimator. Do not add a second adapter ABC, result wrapper, registry, plugin loader, capability hierarchy, generic SDK keyword forwarding or worker framework before a real consumer demonstrates a difference they hide. A naming wrapper around one SDK call is not enough.

**Criterion:** try deleting a proposed layer. If callers would then repeat backend handling or need to know SDK-specific shapes, dtypes or devices, the layer earns its place. If deleting it only moves the same calls and names into another module, that's not an abstraction, that's an indirection layer. Judge hidden work, not line count or caller count.

Consequences: adding an adapter is concrete native mapping work, not registration configuration. There is no provider matrix or YAML extension point. Internal helpers may change without a compatibility decision; public abstractions require the same review as any other interface. Related decisions: [ADR 0002](0002-minimal-predict-interface.md), [ADR 0006](0006-preserve-array-backend.md), [ADR 0008](0008-explicit-pretrained-loading.md), [ADR 0010](0010-standard-array-api-helpers.md).
