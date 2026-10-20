---
id: ADR-0010
status: accepted
date: "2026-10-20"
---

# Use standard Array API helpers where they reduce backend handling

The user pointed to scikit-learn's Array API support while reviewing NumPy/Torch interoperability. Use the public helpers from `array-api-compat` for array recognition, namespaces, dtype checks, finiteness and device handling where applicable rather than maintaining a parallel internal array abstraction; one small standard dependency is preferable to a new custom array wrapper or backend strategy hierarchy.

`__array__` requests a NumPy representation and does not preserve a Torch backend/device. The relevant standard is `__array_namespace__`; `array-api-compat` supplies compatible namespaces even where native support is missing. This is array interoperability, not scikit-learn estimator compatibility or model-runtime portability. Keep public context support restricted to NumPy arrays and Torch tensors, with explicit typed overloads; recognizing additional array namespaces does not make them supported inputs.

Native SDKs still require their own input representation. Transfers, contiguous copies where needed, output dtype/device restoration and post-conversion validation remain explicit adapter responsibilities. Do not promise zero-copy execution or use private `sklearn.utils._array_api` helpers. The supervisor adopted this dependency after reviewing its use in the initial adapters: public family recognition, namespace-based finiteness checks and Torch dtype/device inspection replace backend-specific helper logic without introducing a custom array wrapper. Native SDK conversion remains explicit. This is an implementation decision under the authorized supervision scope; it does not change ADR 0006's public input contract.

Sources: [scikit-learn Array API support](https://scikit-learn.org/stable/modules/array_api.html), [NumPy interoperability](https://numpy.org/doc/stable/user/basics.interoperability.html), [Array API namespace protocol](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html), [array-api-compat](https://data-apis.org/array-api-compat/) and its [public helpers](https://data-apis.org/array-api-compat/helper-functions.html).
