# forecasters

A thin, model-agnostic adapter layer over pretrained time-series foundation models.

Status: **design phase** (contract v0.2 under review). The implementation follows
[`CONTRACT.md`](CONTRACT.md). The PyPI name `forecasters` is reserved, but the current 0.0.1 release is a
placeholder; 0.1.0 will be the first real release.

## What it is

`forecasters` does not implement forecasting. Each adapter calls one model's native SDK and returns that SDK's
own point forecast and quantiles. The library provides:

- a small, uniform `fit` / `predict` / `predict_proba` surface;
- one worker process per model, so conflicting torch/transformers pins can coexist;
- a `Forecast` value object carrying native levels and values;
- capability checks and a small error taxonomy.

It is **not** a modelling framework. No training, scaling, imputation, multivariate/exogenous plumbing,
interpolation between quantile levels, or quantile-crossing repair.

```python
from forecasters import load

with load("chronos_bolt", python=".venvs/chronos/bin/python") as f:
    f.fit(y)                    # capture context; load the native model once
    point = f.predict(24)             # native point forecast, (24,)
    q = f.predict_quantiles(24)       # native quantile levels (canonical)
    lo, hi = q.interval(0.8)          # only when 0.1 and 0.9 are native
    same = f.predict_proba(24)        # product-requested alias of predict_quantiles
```

`fit` is context capture, not training. The quantile method returns a `Forecast` carrying the model's native
levels and values; it does not interpolate. `predict_proba` is the requested sklearn-flavoured alias of the
canonical `predict_quantiles`.

## Design

Read the contract first:

- [CONTRACT.md](CONTRACT.md) — public API, adapter contract, worker protocol, errors, non-goals.
- [design/model-matrix.md](design/model-matrix.md) — native APIs of 15 foundation models.
- [design/prior-art.md](design/prior-art.md) — sktime, Darts, GluonTS, NeuralForecast, Transformers, sklearn.
- [design/edge-cases.md](design/edge-cases.md) — inputs, outputs, lifecycle, protocol, invariants, tests.
- [design/opencode-review.md](design/opencode-review.md) and
  [design/opencode-review-2.md](design/opencode-review-2.md),
  [design/opencode-review-3.md](design/opencode-review-3.md) — external reviews and verification by Opus 5.5
  via GitHub Copilot.

## Reference app

The Line-up drawing app is the reference consumer: one canvas, several models, native intervals. It will be
ported to `examples/lineup` after the 0.1 core and the first adapters land.

![Line-up reference app](docs/overlay.png)

## Planned adapters (0.1)

| Adapter | Native SDK | Notes |
| --- | --- | --- |
| `chronos_bolt` | `chronos-forecasting` | first reference adapter; CPU-friendly |
| `timesfm25` | `timesfm[torch]` | compile-time horizon; no public device arg |
| `moirai2` | `uni2ts` | 512-token context; CC-BY-NC-4.0 weights |
| `toto` | `toto-ts` | sampled quantiles, needs a derived grid |
| `timestral` | `timestral` (git) | 21 native levels |
| `naive` | none | random-walk baseline for contract tests |

Adapters are added only after the contract and worker protocol are settled. Each adapter must pass the
conformance suite in `forecasters.testing`.

## Development plan

1. Implement `Forecast`, `Capabilities`, the adapter ABC and the error taxonomy.
2. Implement the worker protocol and supervisor (handshake, timeouts, cleanup, protocol fd).
3. Implement `chronos_bolt` plus the `naive` test adapter and the conformance suite.
4. Port the Line-up app as `examples/lineup` on top of the library; keep the UI.
5. Add the remaining adapters behind optional extras/venvs; publish 0.1.0.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

No license file is included yet; add one before publishing.
