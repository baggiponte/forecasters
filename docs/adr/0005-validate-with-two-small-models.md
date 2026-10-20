---
id: ADR-0005
status: accepted
date: "2026-10-20"
---

# Validate the abstraction with Chronos-Bolt Tiny and Timestral 5M

The user selected this pair in the [submitted decision review](../../design/decision-review-2026-10-20.md). This accepts the model choice, not a claim that installation or inference validation has passed. Model loading uses concrete `Adapter.from_pretrained(...)` calls under ADR 0008.

Start with `amazon/chronos-bolt-tiny` (9M parameters) through `chronos-forecasting`, then `mistralai/Timestral-5M` through Mistral's native inference package. Both supply a 0.5 quantile and have documented CPU execution paths. The pair exercises different SDKs and output representations without sampling, providing evidence for a shared median interface before adding a generic plugin mechanism.

At selection time these were candidates pending an installation and inference spike, not measured performance recommendations. The subsequent [implementation validation](../../design/implementation-validation.md) records successful installation, native parity and measured timings for this pair. Chronos-Bolt provides a mature packaged baseline; Timestral provides a small second implementation but currently needs a repository-based install and has separate code and weight licenses. Record exact SDK versions or source revisions, checkpoint revisions, hardware, dtype, context length, horizon, cold loading time and warm prediction latency when validating. A parameter count does not establish interactive latency or memory use.

Begin with a common demo horizon no greater than Chronos-Bolt's native 64-step horizon and a context inside both models' supported limits. Reduce the UI's existing 160-step maximum explicitly; do not conceal SDK rollout or resample predictions to fill the canvas. Before adding the second adapter, verify that both SDKs actually install together on the target Python and CPU; if they do not, document the concrete conflict and reconsider the pair or example-level isolation.

The broader research catalog includes the requested Forecasting Company t0 models, Chronos generations, TimesFM, Toto and Moirai, plus other credible families. Catalog membership is not an implementation promise. Toto 2's tiny checkpoints are worth evaluating as alternatives; Moirai and sample-based models introduce additional glue and are deferred until the first pair works. Hosted inference must be identified as remote rather than included under the UI's “all local” claim.

Evidence: [Chronos-Bolt Tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny), [Chronos-Bolt inference source](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py), [Timestral 5M model card](https://huggingface.co/mistralai/Timestral-5M), [native Timestral SDK](https://github.com/mistralai/timestral). The [updated research](../../design/research-summary-2026-10-20.md) records the wider catalog and uncertainties.
