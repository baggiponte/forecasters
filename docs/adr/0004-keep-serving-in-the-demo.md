---
id: ADR-0004
status: accepted
date: "2026-10-20"
---

# Keep serving and dependency isolation outside the core library

**User review:** accepted in-process core and a simple example server, with the clarification “do not worry about the server now.” The first implementation step targeted the library and first adapters. FastAPI is a possible later server choice; BentoML was mentioned as an alternative, not selected or required. The user subsequently authorized connecting and running the demo, with small logical commits implemented by DeepSeek. For that step the supervisor selected FastAPI with Uvicorn as the simple example server; this choice adds no serving dependency to the core installation. The serving details below guide this demo step. The first demo shows the median only; library quantile support does not require bands in that first UI.

Forecasters is an in-process Python inference layer. A small server under `examples/lineup` owns the browser transport, model list, loading status and request scheduling. Native dependencies are optional per adapter; start by resolving one environment for the two selected small checkpoints. Introduce environment isolation only if an actual incompatibility blocks those checkpoints, with the remedy in the demo or deployment before expanding the core interface.

The previous contract put a worker interpreter, versioned JSON-lines protocol, handshake, supervisor, locks, timeouts and terminal lifecycle states into every model call. Those features may help a large model zoo, but the current repository has no implemented adapter or demonstrated dependency conflict for the first pair. Requiring them now would make a generic process framework the prerequisite for the first useful prediction.

Preserve the existing canvas and layout. The current UI expects `GET /api/models`, `POST /api/models/{key}/load`, `POST /api/forecast` and `/ws`; a small server can meet those routes without adding an IPC protocol inside Forecasters. A single server process and serialized inference are sufficient initially. Run blocking SDK inference outside the async event loop, retain request sequence checks, and coalesce superseded drawing requests so a CPU model does not accumulate an unbounded queue.

For the first milestone, send and render an explicit median vector, remove forecast-value smoothing, disable bands and quantile scoring, and respect the selected models' actual horizon limits. Animation may move between draws, but settled values must equal the library output. HTTP and WebSocket paths must return the same median semantics; the example preview is a consumer too. Preserve the UI's error states and do not replace failed models with synthetic predictions.

The exact web framework is an implementation choice, not a core-library commitment. No deployment platform, model scheduling framework or multi-user serving guarantee is implied. See [repository audit](../../design/repository-audit.md) for the current transport and renderer details.
