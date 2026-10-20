---
id: ADR-0006
status: accepted
date: "2026-10-20"
---

# Preserve the caller's NumPy or PyTorch array family

The user's corrected rule applies to the actual return value: `context` accepts only a NumPy array or a PyTorch tensor. NumPy input returns a NumPy array and PyTorch input returns a PyTorch tensor. Python lists, tuples and other Python sequences are not supported context inputs. `Sequence[float]` describes only the optional quantile levels in ADR 0007. Point forecasts and requested quantiles use the same array-family rule. Native SDK representation is an adapter detail.

Use explicit typed overloads: NumPy input returns NumPy and tensor input returns tensor. Do not annotate `context` with `ArrayLike` or a Python `Sequence`, which would advertise unsupported inputs. Validate the input type before any coercion; do not silently accept a list by passing it to `asarray` or `as_tensor`. Static checking should preserve the relationship rather than returning an unconstrained NumPy-or-Torch union. Keep PyTorch optional for NumPy-only consumers. The contract concerns ordinary `numpy.ndarray` and `torch.Tensor` values, not arbitrary subclass preservation.

**Accepted dtype policy:** preserve floating input dtype for NumPy arrays and PyTorch tensors. The form's earlier rule for Python sequences is withdrawn by the user's input-type correction. Native computation can use a different dtype; casting the result back does not recover precision lost during inference.

**Supervisor implementation policy:** accept floating context arrays/tensors only in this milestone. Reject integer, boolean and complex dtypes with a clear `TypeError` before inference, explaining that floating input is required. This resolves the previously unspecified integer case without silently casting inputs or returning integer forecasts. It is a conservative implementation choice made under the user's instruction to supervise implementation, not a claim that the earlier float32 conversion suggestion was accepted.

**Accepted device policy:** return PyTorch results on the input tensor's device. Native model compute device is configured separately and may require transfers. This does not imply zero-copy execution or autograd support.

Return one n-dimensional array/tensor directly, never a result object or tuple containing point and quantile fields. The user explicitly resolved this distinction after the earlier t0-style container question. Return shape is separate from return type: it depends on whether quantiles were requested and on the supported forecasting dimensions. This decision does not by itself add batching or multivariate support to the first milestone.

This replaces ADR 0002's NumPy-only, fixed-float64 return assumption. It adds no inference implementation or public dependency on a shared model framework.
