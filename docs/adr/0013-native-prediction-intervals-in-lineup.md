---
id: ADR-0013
status: accepted
date: "2026-10-20"
---

# Add an optional native 80% prediction interval to Line-up

Line-up can display the central prediction interval between native quantiles 0.1 and 0.9, alongside the native 0.5 median. The user requested intervals after validating the median demo. This advances [ADR 0004](0004-keep-serving-in-the-demo.md)'s initial median-only UI; the library's public prediction interface remains [ADR 0007](0007-request-only-supported-quantiles.md).

Call `predict` once with levels `[0.1, 0.5, 0.9]` when bands are requested, and extract the median and endpoints from that result. With bands disabled, use the existing default median call. Check the loaded model's actual supported levels; an unavailable endpoint fails visibly. The first implementation offers this one central interval rather than arbitrary coverage controls.

Label the band an **80% prediction interval**. It concerns future observations, not a confidence interval for a fitted parameter or a guarantee of calibrated or simultaneous trajectory coverage. Forecasters adds no interpolation, smoothing, endpoint repair or artificial uncertainty.

The HTTP, WebSocket and notebook consumers must agree on level semantics and preserve the selected native values. Rendering, image export and model changes carry the same band state; stale responses must not restore bands after the user switches them off. Keep the median visible when intervals are displayed. Tests earn their place by checking actual native endpoints and transport agreement, under [ADR 0012](0012-useful-tests.md).
