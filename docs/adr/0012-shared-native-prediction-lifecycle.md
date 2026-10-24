---
id: ADR-0012
status: proposed
date: "2026-10-23"
---

# Inherit the prediction contract, implement only native mapping

## Context

The two native adapters repeat request validation, checkpoint limits, conversion, quantile selection and restoration. Shared helpers alone still require every new adapter to wire those steps correctly and duplicate the public overloads. The requested architecture should make new adapters small while leaving room for observation and additional caller representations without implementing either now.

## Decision

Introduce an optional public `BaseForecaster` for native Torch quantile models. Its typed, final `predict` sequences the existing shared behavior. A concrete adapter declares checkpoint metadata, native dtype and axis layout, and implements `_predict_native` to call its SDK and extract the raw quantile tensor. Exact native shape checking happens centrally before normalization. Keep `Forecaster` structural and preserve existing native-object constructors and model-specific `from_pretrained` factories.

Prefer this explicit template method over decorators, automatic subclass wrapping and composed forwarding objects. A canonical `(H, Q)` hook is a credible alternative, but cannot enforce the original native shape after a concrete adapter has reshaped it. Axis labels express the two demonstrated SDK layouts without a registry or model-name dispatch.

## Consequences

New adapters inherit the validation and representation contract without copying orchestration or overloads. Exporting the base makes its constructor, layout vocabulary and native hook a supported authoring interface; the public protocol remains available to independent implementations. The base deliberately covers quantile tensors, not arbitrary future model outputs. Observation can wrap the structural interface or be added at the shared lifecycle; dataframe semantics belong at the shared representation stages or an outer tabular adapter. No callbacks, dataframe support or plugin machinery ship with this decision.
