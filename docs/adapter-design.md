# Adding a forecasting adapter

`Forecaster` is the interface consumers use. `BaseForecaster` is an optional authoring interface that supplies the prediction contract for native Torch quantile models. They have different jobs: consumers do not need to know which adapters inherit the base.

An adapter supplies four SDK facts: how to load the model, its checkpoint metadata, its tensor layout/dtype, and how to call inference. Validation, quantile matching and selection, native shape checks, and caller-representation restoration are inherited. No registration is required.

## The layout

```text
forecasters/
    _forecaster.py       Structural consumer protocol
    _base.py             BaseForecaster: one typed predict lifecycle
    _validation.py       Context, horizon and native quantile validation
    _outputs.py          Torch bridge, exact layout checks, output restoration
    _deps.py             Lazy optional SDK imports
    chronos_bolt.py      Loading + metadata/layout + native SDK call
    timestral.py         Loading + metadata/layout + native SDK call
```

```text
caller: NumPy array or Torch tensor
    │
    ▼
BaseForecaster.predict
    validate context, horizon, checkpoint limits and requested levels
    capture caller family/dtype/device; prepare native Torch context
    │
    ▼
adapter._predict_native                         ← model-specific operation
    call SDK; extract its full native quantile tensor
    │
    ▼
shared output handling
    check exact raw shape; normalize axes to (H, native Q)
    select median or requested columns; check selected values
    restore caller family/dtype/device; check conversion overflow
    │
    ▼
caller: (H,) median or (H, requested Q) quantiles
```

The `predict` overloads live once in the base and once in the structural protocol, not in every adapter. The public method is `@final` for type checkers; the customization method is `_predict_native`. There is no runtime method replacement, metaclass registration, callback dispatch or generic SDK keyword forwarding.

## A two-method adapter

This standalone extension loads in `__init__`, as a third-party author might prefer. Built-in adapters retain their existing native-object constructors and explicit `from_pretrained` factories, so callers and SDK tests need no migration.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from forecasters import BaseForecaster

if TYPE_CHECKING:
    import torch


class MyBolt(BaseForecaster):
    def __init__(self, model_name: str = "amazon/chronos-bolt-tiny") -> None:
        import torch
        from chronos import ChronosBoltPipeline

        self.pipeline = ChronosBoltPipeline.from_pretrained(model_name)
        super().__init__(
            model_name=model_name,
            supported_quantile_levels=self.pipeline.quantiles,
            max_context_length=self.pipeline.model_context_length,
            max_horizon=self.pipeline.model_prediction_length,
            native_layout=("batch", "quantile", "horizon"),
            native_dtype=torch.float32,
        )

    def _predict_native(self, context: torch.Tensor, *, horizon: int) -> object:
        return self.pipeline.predict(
            context, prediction_length=horizon, limit_prediction_length=True
        )
```

Call `MyBolt().predict(context, horizon=8)` exactly as for the built-in adapters. There are no overloads or validation calls to copy into this class. Metadata and layout are irreducible model knowledge; hiding them behind SDK attribute-name guessing would make adding a model less reliable, not easier.

### The native hook contract

- Input is a validated one-dimensional floating Torch tensor. `native_dtype=None` preserves its bridge dtype; passing a floating Torch dtype requests conversion. Integer, boolean and complex native dtypes are rejected. The caller's original dtype is retained separately for output restoration. Native bfloat16 output is widened to float32 before the NumPy bridge, which preserves its values even though NumPy has no bfloat16 dtype.
- Input device is the caller's device. The SDK or native hook handles transfer to its inference device if needed. Native compute device and returned caller device are separate concerns.
- Return the full native quantile tensor, in loaded-checkpoint grid order, without selecting columns. Returning `object` deliberately treats SDK output as untrusted until the shared checks establish its type and shape.
- `native_layout` contains exactly one `"horizon"` and `"quantile"`, with optional unique `"batch"` and `"variate"` axes. The latter must have size one: these labels do not enable batching or multivariate forecasting.
- Layout declaration is required, even for `("horizon", "quantile")`. Chronos uses `("batch", "quantile", "horizon")`; Timestral uses `("batch", "variate", "horizon", "quantile")`. Other orders work without new model-specific shape-checking code.
- Do not override `predict`, retain request-local context on the instance, truncate inputs, repair native shapes, interpolate quantiles, or mutate the caller's context. The bridge may share storage; it is not a defensive copy of every input. An SDK that mutates inputs needs an explicit copy in its native hook.

The complete native shape is checked **before** axes are moved or singleton dimensions are removed. Checking only the element count would let a transposed or extra-batch result silently become a plausible forecast. Conversely, finiteness is checked **after** selection: an unrequested NaN column is not a failure of a requested median forecast. Singleton horizons and explicit singleton quantiles keep their distinct public shapes.

Shape checks cannot establish axis semantics when the horizon and quantile counts are equal. Declare the SDK layout explicitly and check real-SDK parity with asymmetric horizon/quantile counts; a square tensor can hide an incorrect declaration.

The base covers the existing contract, not every conceivable forecasting model. A sample-only, point-only, non-Torch or multivariate model needs a separate contract decision; it can still satisfy `Forecaster` independently where its behavior fits. There is no claim that changing the SDK call alone can accommodate a different statistical product.

## Alternatives explored

### 1. Shared functions only

Authors supply a concrete `predict`, overloads and calls to each stage. This has the lowest migration cost, but authors still wire the contract and can omit a check. Keep functions as implementation, not the authoring interface.

### 2. Base with canonical `(H, Q)` hook

Authors supply loading, metadata, the SDK call and native shape normalization. This is the viable runner-up, but the base cannot verify the original SDK shape once the hook has squeezed or reshaped it.

### 3. Base with raw native hook and declared layout — chosen

Authors supply loading, metadata/layout and the SDK call. One ordinary inherited lifecycle enforces the full contract, with no per-model orchestration.

### 4. Explicit method decorator

Authors supply metadata, loading and a decorated native `predict`. This is closest in spelling to “predict just calls the SDK”, but the decorated method has a different public signature. Correct overloads require descriptor/binding machinery rather than a simple signature-preserving decorator.

### 5. Composed prediction runner

Authors supply a native callable, metadata, preparation/decoding functions and assembly. It offers good flexibility, but preserving today's concrete adapter identities requires forwarding methods/overloads, or changing factories to return a different type. Better suited to whole-call observation wrappers.

### 6. Automatic subclass method wrapping

Authors supply metadata and a native-shaped `predict`, replaced at class creation. Hidden method replacement makes inheritance, introspection and static typing harder. No current global output configuration justifies that cost.

The winning module has depth because one small authoring interface hides meaningful forecasting behavior. Its value is not the line count: a new adapter cannot accidentally forget the horizon limit, return the SDK's mean instead of its median, or skip caller-dtype restoration simply by omitting a helper call.

### What the familiar libraries actually do

These comparisons were made against pinned upstream source, not just familiar class names:

- [scikit-learn 1.7.2 `BaseEstimator` and `TransformerMixin`](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/base.py): the base supplies shared estimator machinery; concrete estimators still implement their operations. Forecasters borrows a small inherited authoring interface, **not** `fit`, parameter cloning or a claim of scikit-learn compatibility.
- [scikit-learn 1.7.2 `_SetOutputMixin`](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/utils/_set_output.py): `__init_subclass__` wraps `transform`/`fit_transform` to implement configurable output containers, coupled to feature-name support. This is a real precedent for wrapping, but not a reason to introduce its machinery here.
- [Transformers 4.57.1 `Pipeline`](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/pipelines/base.py): `run_single` sequences `preprocess → forward → postprocess`; `forward` owns shared inference/device handling around `_forward`. Forecasters borrows that explicit stage ownership, with fewer customization methods because validation and output selection do not vary by model.
- [Transformers 4.57.1 `PreTrainedModel`](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py): weight loading and model implementation are distinct concerns. Forecasters retains model-specific typed `from_pretrained` factories rather than copying a large model framework or promising universal loader options.

The familiarity comes from ordinary constructors, explicit metadata, an inherited public operation and a narrow protected native operation—not from inheriting either dependency.

## Extension locations, not implemented features

### Logging, latency and metrics

A whole-call observing wrapper can implement `Forecaster` and delegate to any forecaster, including one that does not inherit this base. Counts, errors and elapsed time need no changes in native adapters. If a real consumer later needs native-inference time rather than end-to-end time, the shared lifecycle provides one place to add observation around `_predict_native`.

There is deliberately no empty callback method, observer registry, `callbacks=` constructor argument or event schema now. Future work must choose ordering, failure propagation and concurrency semantics rather than inheriting accidental behavior from unused hooks. Forecast-quality metrics additionally require realized future observations; they are not something native inference can compute alone.

### Dataframes and Narwhals

Container preservation is a representation concern, not just another numeric dtype. Currently `validate_context` captures an `InputSpec`, the Torch bridge prepares inference input, and `restore_output` reconstructs the caller's representation. A future dataframe extension can change those shared stages and the overloads in `Forecaster`/`BaseForecaster` without changing native hooks.

It must first define series-column selection, null handling, quantile column labels, and future index/timestamp semantics; restoring the original history index onto future observations would be wrong. A separate tabular wrapper around `Forecaster` is also possible when those semantics should stay outside the array interface. Narwhals can help with container interoperability but does not decide the forecasting contract. No dataframe dependency, coercion or plugin registry is added here.

## Testing an adapter

Use a tiny SDK fake or mock and call the public `predict` interface. Assert independently specified values with different horizon/quantile counts, requested order, singleton shapes, representation preservation, and useful failures. Mock the external SDK, not the shared validation or restoration code. The third-adapter contract test demonstrates that no built-in adapter code is required.

Keep direct native-SDK parity tests as well: a fake cannot establish integration correctness. Compare against independently indexed raw SDK results, not the shared layout helper. Existing static consumer checks verify inherited dtype-sensitive overloads, and isolated installed-distribution smoke tests verify optional dependencies stay optional.
