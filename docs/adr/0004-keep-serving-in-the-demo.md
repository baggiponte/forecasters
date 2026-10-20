---
id: ADR-0004
status: accepted
date: "2026-10-20"
---

# Keep serving in an example and the core in process

The core library is an in-process Python inference layer. Serving, browser transport, model status and request scheduling belong to an optional example under `examples/lineup`. The user authorized connecting and running the demo; the supervisor selected FastAPI with Uvicorn for that step. This adds no serving dependency to the core installation, and the framework is not a core-library commitment. The user later requested a local marimo/AnyWidget host under [ADR 0014](0014-local-marimo-anywidget-lineup.md); that host runs inference in its notebook kernel without the FastAPI process.

The example owns the existing canvas: `GET /api/models`, `POST /api/models/{key}/load`, `POST /api/forecast` and `/ws`. A single local server process is sufficient. Model loading and blocking SDK inference run outside the async event loop, inference is serialized, and the WebSocket keeps one in-flight request plus one latest pending drawing request. The first demo rendered only the explicit median vector; settled values equal library output. The user subsequently requested native prediction bands, decided in [ADR 0013](0013-native-prediction-intervals-in-lineup.md). Forecast-value smoothing and quantile scoring remain outside that UI. Failed models remain visible as errors and are never replaced with synthetic predictions.

The earlier contract required a worker interpreter, JSON-lines protocol, handshake, supervisor, locks, timeouts and lifecycle states for every model call. That framework may help a large model zoo, but the current repository has no demonstrated dependency conflict for the two chosen checkpoints, so it would make a generic process system the prerequisite for the first useful prediction. Start with one environment for both checkpoints. Add environment isolation only if a real incompatibility blocks them, and put the remedy in the example or deployment before expanding the core interface.

Native dependencies remain optional per adapter. No deployment platform, authentication model, scheduler or multi-user serving guarantee is implied.

See [implementation evidence](../../design/implementation-validation.md) for the current routes, tests and browser checks, and the [repository audit](../../design/repository-audit.md) for transport and renderer details.
