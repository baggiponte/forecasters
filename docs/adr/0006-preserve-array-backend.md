---
id: ADR-0006
status: accepted
date: "2026-10-20"
---

# Accept NumPy arrays and Torch tensors and preserve their representation

Within [ADR 0001](0001-median-first-native-sdk-scope.md)'s one-dimensional, nonempty, finite scope, `context` accepts only an ordinary `numpy.ndarray` or `torch.Tensor`. Python lists, tuples and other array libraries are rejected before any coercion. `Sequence[float]` applies only to optional quantile levels under [ADR 0007](0007-request-only-supported-quantiles.md).

NumPy input returns a NumPy array; Torch input returns a Torch tensor. Preserve the input's floating dtype, and return Torch results on the input tensor's device. Explicit typed overloads preserve this relationship. Reject integer, boolean and complex arrays or tensors with a clear `TypeError` before inference; callers convert explicitly if that is what they want. PyTorch remains optional for NumPy-only consumers.

Return one array or tensor directly, never a result object or a tuple containing point and quantile fields. The same family, dtype and device rules apply to the default point forecast and to requested quantiles. Return shape is owned by [ADR 0007](0007-request-only-supported-quantiles.md) and does not add batching or multivariate support.

This contract concerns ordinary NumPy arrays and Torch tensors, not arbitrary subclasses. It does not promise preservation of a literal subclass. Recognizing additional Array API namespaces for internal interoperability does not make them public context types; [ADR 0010](0010-standard-array-api-helpers.md) owns that boundary.

Native computation may use another dtype or device; casting the result back restores representation, not precision already lost during inference. No zero-copy or autograd behaviour is promised.

**Decision provenance:** the user accepted floating dtype preservation and returning Torch results on the input tensor's device. The supervisor chose floating-only contexts with explicit rejection of other dtypes rather than silent conversion. Historical form answers and corrections are in the [submitted decision review](../../design/decision-review-2026-10-20.md).
