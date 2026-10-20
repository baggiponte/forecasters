---
id: ADR-0010
status: accepted
date: "2026-10-20"
---

# Use standard Array API helpers without broadening the public input contract

Use the public helpers from `array-api-compat` for array recognition, namespaces, dtype checks, finiteness and device handling where they reduce backend-specific code. One small standard dependency is preferable to maintaining a custom array wrapper or backend strategy hierarchy.

This is array interoperability, not a public Array API promise. Context support remains limited to NumPy arrays and Torch tensors under [ADR 0006](0006-preserve-array-backend.md); an arbitrary object exposing `__array_namespace__` is still rejected. `__array__` requests a NumPy representation, while `__array_namespace__` is the backend-aware standard hook. `array-api-compat` does not make model runtimes portable or imply scikit-learn estimator compatibility. Native SDKs still require their own input representation: transfers, contiguous copies where needed, output dtype/device restoration and post-conversion validation remain explicit adapter responsibilities. No zero-copy execution is promised.

The supervisor adopted the dependency after reviewing its use in the adapters. Use public helpers rather than private `sklearn.utils._array_api` functions. [ADR 0011](0011-abstractions-that-earn-their-place.md) owns the general rule for when such a shared helper earns its place.

Sources: [scikit-learn Array API support](https://scikit-learn.org/stable/modules/array_api.html), [NumPy interoperability](https://numpy.org/doc/stable/user/basics.interoperability.html), [Array API namespace protocol](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html), [array-api-compat](https://data-apis.org/array-api-compat/) and its [public helpers](https://data-apis.org/array-api-compat/helper-functions.html).
