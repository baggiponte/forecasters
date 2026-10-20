# Submitted design decisions — 2026-10-20

> Historical form submission below, followed by explicit user corrections. Current decisions are in the ADRs;
> the original response mentioning Python sequences is retained as provenance, not current context support.

Source: user-submitted Markdown from `forecasters-design-review-v1`, generated 2026-10-20T16:10:10.721Z. This is an explicit answer handoff, not an autosaved browser draft. No decisions were delegated in that submission.

| Stable question ID | Explicit answer | Recorded in |
| --- | --- | --- |
| `scope-univariate-1d` | Univariate 1-D first | ADR 0001 |
| `initial-pair` | Chronos-Bolt Tiny + Timestral 5M | ADR 0005 |
| `architecture` | In-process core + simple example server | ADR 0004 |
| `horizon-rollout` | Reject horizons beyond the native limit | ADR 0001 |
| `context-overmax` | Reject contexts beyond the maximum, raising an error | ADR 0001 |
| `dtype` | Preserve floating input dtype; Python sequences become float32 | ADR 0006 |
| `torch-device` | Return on the input tensor's device | ADR 0006 |
| `demo-quantiles` | Median only in the first demo | ADRs 0001, 0004 |
| `loading-api` | Concrete `adapter.from_pretrained(...)` | ADR 0008 |
| `sample-based-models` | Unsure / defer; explanation requested | Open; not approval of a sampling policy |

User rationale for `architecture`: do not worry about the server now; it can simply use FastAPI, or possibly BentoML. Neither framework was formally selected. User rationale for `context-overmax`: raise an error. User rationale for `sample-based-models`: “non ho capito”. Other rationales, alternatives and general notes were empty.

## Subsequent explicit user corrections

The user clarified that `context` accepts only NumPy arrays and Torch tensors for now. Python `Sequence[float]` applies to the requested quantile levels. The form's sequence-context and sequence-to-float32 language was a misunderstanding and is withdrawn in ADR 0006, not an accepted requirement.

After the sampling explanation, the user explicitly chose to consider sample-based models later; ADR 0009 records that deferral. The earlier “unsure / defer” response is no longer an open question. The suggestion to convert integer arrays/tensors to float32 was not accepted. When the user authorized supervised implementation, the supervisor chose floating context dtypes only and explicit rejection of integer/bool/complex inputs, recorded as an implementation policy in ADR 0006 rather than as a user-selected conversion rule.

The user also requested simple, robust ADR metadata: use an atomic `status` and a separate `superseded_by` list of stable ADR IDs. The ADR index documents the convention, applied consistently to all ADR files.

Installation compatibility, checkpoint limits, output parity, memory and latency are verification work rather than user preference questions. The form answers do not establish that these checks have passed or authorize publishing/deployment.
