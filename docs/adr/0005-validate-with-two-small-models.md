---
id: ADR-0005
status: accepted
date: "2026-10-20"
---

# Validate the abstraction with Chronos-Bolt Tiny and Timestral 5M

Use `amazon/chronos-bolt-tiny` through `chronos-forecasting` and `mistralai/Timestral-5M` through Mistral AI's native `timestral` package as the initial pair. Both expose a native 0.5 quantile and a documented CPU path. The pair exercises different SDKs and output representations without sampling, giving evidence for the shared median interface before any generic plugin mechanism.

At selection time these were candidates pending an installation and inference spike, not measured performance recommendations. Exact SDK and checkpoint revisions, hardware and timings are not product guarantees.

Use the minimum native `max_horizon` among the selected ready adapters as the demo request cap; with Chronos-Bolt selected this is 64, while a Timestral-only selection permits its larger native cap. Keep the context inside both selected limits. The browser leaves room for the forecast within the current canvas; rollout or resampled predictions do not fill it. The pair is valid only if both SDKs install together on the target Python and CPU; a concrete conflict would require reconsidering the pair or example-level isolation rather than expanding the core.

Timestral currently needs a source-based install and has separate code and weight licenses; these are packaging constraints, not reasons to build a generic loader. The wider catalog is research input; catalog membership is not an implementation promise. Hosted inference must be labelled remote rather than included under the UI's "all local" claim.

Evidence: [Chronos-Bolt Tiny](https://huggingface.co/amazon/chronos-bolt-tiny), [Chronos-Bolt source](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py), [Timestral 5M](https://huggingface.co/mistralai/Timestral-5M) and [native Timestral SDK](https://github.com/mistralai/timestral).
