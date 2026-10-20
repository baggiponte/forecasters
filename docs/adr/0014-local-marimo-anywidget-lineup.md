---
id: ADR-0014
status: accepted
date: "2026-10-20"
---

# Run Line-up locally in marimo with an AnyWidget

Provide a marimo notebook and a purpose-built AnyWidget as a local Line-up example. The user wants the drawing UI and Python forecasting to work together in one notebook runtime, with possible Hub deployment considered later. The notebook calls the existing native-SDK adapters in its own Python kernel; it does not require the separate FastAPI demo process. Notebook and widget dependencies remain optional and outside the core inference API.

Reuse one plain-JavaScript `mountLineup(root, host)` component in the standalone page and the widget. These two actual consumers justify the seam under [ADR 0011](0011-abstractions-that-earn-their-place.md): it separates drawing/rendering from browser transport while preserving one implementation of bands, model selection, counts and image export. Each mount owns its state, DOM queries, container sizing and cleanup. Widget styles stay inside its root and assets resolve without the standalone server.

The kernel owns loaded `Forecaster` instances and reuses their weights. Only JSON-compatible requests, model metadata and results cross the AnyWidget boundary; native model objects stay private in Python. Use the public AFM custom-message channel for forecast requests and replies, with synchronized metadata and result snapshots for inspection. This keeps the request/reply operation separate from the host's reactive trait-update machinery. The widget handles requests without a downstream reactive cell writing into its own input widget; verify the actual marimo interaction instead of assuming that a trait callback provides reliable immediate replies. Correlate replies with a request identifier so separate views cannot apply each other's results, and release request listeners on completion or cleanup.

For the first notebook, commit prediction requests when the user finishes a stroke or explicitly changes a preset, selected model or band setting. Pointer movement paints locally. Synchronous inference on the two small CPU models is sufficient for this slice; add scheduling only after a concrete need. The standalone page may retain its existing streaming behavior.

Use the same median and native interval semantics as [ADR 0007](0007-request-only-supported-quantiles.md) and [ADR 0013](0013-native-prediction-intervals-in-lineup.md). Shared example-level Python helpers may serve both hosts when they remove duplicated catalog, request or serialization logic. They introduce no web dependency or alternate model interface into the core library.

Start with notebook, widget and assets in this repository's example directory and project environment. A separate widget distribution, frontend build system and Hub deployment are later decisions. This milestone is complete when real hand-drawn data produces native forecasts in a running local notebook, model weights are reused, and the standalone consumer still works.

References: [marimo AnyWidget integration](https://docs.marimo.io/api/inputs/anywidget/), [AnyWidget front-end lifecycle and model API](https://anywidget.dev/en/afm/), [drawdata](https://github.com/koaning/drawdata) and [wigglystuff](https://github.com/koaning/wigglystuff). Their useful patterns are synchronized traits, explicit view cleanup and local JavaScript/CSS assets; they are references rather than required runtime dependencies.
