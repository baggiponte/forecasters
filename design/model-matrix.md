# Native TSFM inference APIs — model matrix

Scope: what each pretrained time-series foundation model exposes through its **native** Python SDK, for a v0.1
adapter that is univariate, context-in/quantiles-out, one model per worker. Compiled 2026-10-20 from model cards,
package docs, and repo source; unverified points are marked **uncertain**. Nothing is inferred from the old
prototype adapters.

## At a glance

| Adapter key | Model ids | Package (PyPI) | Levels | Point | Samples | Context | Horizon | Weights licence |
|---|---|---|---|---|---|---|---|---|
| `chronos_bolt` | `amazon/chronos-bolt-{tiny,mini,small,base}` | `chronos-forecasting` 2.3.2 | 0.1–0.9 (9) | 0.5 col | no | 2048 | 64 native | Apache-2.0 |
| `chronos` | `amazon/chronos-t5-{tiny..large}` | `chronos-forecasting` | empirical | sample mean | yes | 512 | 64 | Apache-2.0 |
| `chronos2` | `amazon/chronos-2`, `autogluon/chronos-2-{small,synth}` | `chronos-forecasting` ≥2.0 | 0.01–0.99 (21) | 0.5 col | no | 8192 | 1024 default | Apache-2.0 |
| `timesfm25` | `google/timesfm-2.5-200m-pytorch` | `timesfm[torch]` | 0.1–0.9 (9)+mean | point head | no | 16384 | compile-set | Apache-2.0 |
| `timesfm3` | `google/timesfm-3.0-pytorch` | `timesfm` ≥3.0.2 (`timesfm3`) | 0.1–0.9 (9) | median | no | 15360 | no fixed cap | **non-commercial** |
| `moirai11` | `Salesforce/moirai-1.1-R-{small,base,large}` | `uni2ts` 2.0.0 | empirical | median | yes | 512 tokens | construct-set | CC-BY-NC-4.0 |
| `moirai2` | `Salesforce/moirai-2.0-R-small` | `uni2ts` | 0.1–0.9 (9) | median | no | 512 tokens | construct-set | CC-BY-NC-4.0 |
| `toto` | `Datadog/Toto-Open-Base-1.0` | `toto-ts` 0.2.0 | empirical | median | yes | not documented | free | Apache-2.0 |
| `toto2` | `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}` | `toto-2` 2.0.0 (py≥3.12) | 0.1–0.9 (9) | median | no | free | free | Apache-2.0 |
| `timestral` | `mistralai/Timestral-{5M,28M,123M,209M}` | `timestral` (git) | 0.01–0.99 (21) | median | no | 32768 | free | Modified MIT |
| `tirex` | `NX-AI/TiRex` | `tirex-ts` 1.4.2 | 0.1–0.9 (9) | median | no | ~2048 | free | NXAI community |
| `tirex2` | `NX-AI/TiRex-2` | `tirex2` (git) | 0.1–0.9 (9) | median | no | 2048 | free | Apache-2.0 |
| `lag_llama` | `time-series-foundation-models/Lag-Llama` | `lag-llama` (git) | empirical | median | yes | 2048 max | free | Apache-2.0 |
| `sundial` | `thuml/sundial-base-128m` | `transformers==4.40.1` | empirical | mean/median | yes | 2880 | 720/pass | Apache-2.0 |
| `ttm` | `ibm-granite/granite-timeseries-ttm-r2` | `granite-tsfm` | none | point | no | per checkpoint | per checkpoint | Apache-2.0 |

“construct-set” = horizon baked into the forecaster at construction; “free” = arbitrary positive
`prediction_length`, possibly by rolling/stitching.

## Priority models

### Amazon Chronos-Bolt (and original Chronos)

- **Ids / package**: `amazon/chronos-bolt-{tiny,mini,small,base}` (9M–205M); original `amazon/chronos-t5-*`
  (8M–710M). One package, `chronos-forecasting` 2.3.2 (py≥3.10), `from chronos import BaseChronosPipeline`
  ([README](https://github.com/amazon-science/chronos-forecasting/blob/main/README.md)).
- **Load**: `BaseChronosPipeline.from_pretrained(id, device_map="cpu", dtype=torch.float32)`; config field
  `chronos_pipeline_class` dispatches Bolt/Chronos/Chronos-2. `device_map`/`dtype` are transformers `AutoModel`
  kwargs (`torch_dtype` pre-4.56); results return float32 CPU tensors
  ([base.py](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/base.py)).
- **Calls**: `predict(inputs, prediction_length)` → `(batch, 9, H)`; `predict_quantiles(..., levels)` →
  `((batch, H, Q), (batch, H))`. Official levels 0.1…0.9; other levels interpolated, with a warning outside
  0.1–0.9; the returned “mean” is actually the 0.5 column
  ([chronos_bolt.py](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py)).
- **Limits**: context 2048, native horizon 64; longer horizons roll using the nine quantiles as pseudo-samples and
  quality degrades. Univariate, no covariates, deterministic. NaN left-padding allowed; internal nan-aware instance
  norm + optional arcsinh. Zero-shot. Apache-2.0.
- **Friction**: transpose quantile axis; cap/free horizon decision at 64; CPU-friendly — best first adapter.
- **Original Chronos**: `predict` → `(batch, num_samples, H)` samples; `predict_quantiles` uses `torch.quantile`
  and sample mean. Context 512, horizon 64.

### Chronos-2

- **Ids / package**: `amazon/chronos-2` (120M), `autogluon/chronos-2-small` (28M), `autogluon/chronos-2-synth`;
  `chronos-forecasting>=2.0`, `from chronos import Chronos2Pipeline` ([card](https://huggingface.co/amazon/chronos-2),
  [pipeline.py](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py)).
- **Load/calls**: `Chronos2Pipeline.from_pretrained(id, device_map="cuda")`; `predict(inputs, prediction_length)`
  returns a **list** per series of `(n_variates, 21, H)`; `predict_quantiles` returns lists of
  `(n_variates, H, Q)`/`(n_variates, H)`; `predict_df` is the pandas front end. Native levels
  0.01, 0.05, 0.10 … 0.90, 0.95, 0.99; arbitrary levels interpolated.
- **Limits**: context 8192 (reducible per call), default horizon 1024 (64×16 patches); beyond that an unrolled
  quantile heuristic runs with batch ×9. Univariate, multivariate, native numeric/categorical past and
  future covariates (`cross_learning=True` shares across batch). Deterministic quantile head; NaN allowed; internal
  arcsinh + instance norm. Zero-shot (pipeline `fit` exists for optional fine-tuning). Apache-2.0.
- **Friction**: list return and 21-level grid (richest native grid); `predict_proba` must not clamp to 0.1/0.9.

### Google TimesFM 2.5

- **Id / package**: `google/timesfm-2.5-200m-pytorch`; `timesfm[torch]` (3.0.2 ships both generations; 2.0.2 was
  the 2.5-era release); `timesfm.TimesFM_2p5_200M_torch` ([card](https://huggingface.co/google/timesfm-2.5-200m-pytorch),
  [api ref](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/api_reference.md)).
- **Load**: `from_pretrained(id, torch_compile=False)` then **mandatory** `compile(ForecastConfig(...))`. Device is
  auto-selected (cuda:0 if available, else CPU; no public `device=` on the 2.5 class). `forecast(horizon,
  inputs=[1-D np arrays])` → point `(N, H)`, quantiles `(N, H, 10)`: col 0 is the **mean**, cols 1–9 are 0.1…0.9.
- **Limits**: context 16384; `max_context + max_horizon ≤ 16384`; `horizon ≤ max_horizon`; continuous quantile head
  ≤1024. `max_context` rounds to 32, `max_horizon` to 128. Univariate only; covariates only through XReg
  (`forecast_with_covariates`, `timesfm[xreg]`, a separate linear model). Leading NaNs stripped, interior NaNs
  linearly interpolated; deterministic. Zero-shot. Apache-2.0.
- **Friction**: `compile()` fixes `max_horizon` (declare it as the adapter limit, or recompile per request);
  mean column must be dropped; CPU cannot be forced via the public API.

### Google TimesFM 3.0

- **Id / package**: `google/timesfm-3.0-pytorch`; `timesfm` ≥3.0.2 exposes `timesfm3`
  (`TimesFM3Forecaster`, `TimesFM3Evaluator`, `ModelConfig`)
  ([card](https://huggingface.co/google/timesfm-3.0-pytorch),
  [forecaster.py](https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/timesfm3_forecaster.py)).
- **Load/calls**: `TimesFM3Forecaster.from_pretrained(id, device="cuda")`; `predict(context, horizon,
  return_quantiles=True)` → `ForecastOutput(forecast=(H,), quantiles=(H, 9))`; multivariate context `(V,T)` gives
  `(V,H)`/`(V,H,9)`; `predict_batch` yields one output per input. Native nine deciles, median index 4. Covariates
  are native: `past_only_covariates=(C,T)`, `past_future_covariates=(C,T+H)`.
- **Limits**: context capped at 15360 (32-step patches; older points truncated); outputs in 64-step patches, so the
  effective horizon rounds up; ≤32 target variates per forward (larger inputs are chunked by the evaluator).
  Leading NaNs stripped, interior interpolated; deterministic. Zero-shot. **Weights are
  `timesfm-non-commercial-license-v1.0`** (self-hosted commercial use not permitted); code Apache-2.0.
- **Friction**: different package/API from 2.5, non-commercial weights, dataclass output, nine levels only.

### Salesforce Moirai 1.1

- **Ids / package**: `Salesforce/moirai-1.1-R-{small,base,large}`; `uni2ts` 2.0.0 (py≥3.10)
  ([README](https://github.com/SalesforceAIResearch/uni2ts),
  [forecast.py](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai/forecast.py)).
- **Load/call**: `module = MoiraiModule.from_pretrained(id)`, then `MoiraiForecast(module=module,
  prediction_length=H, context_length=C, patch_size="auto", num_samples=100, target_dim=1,
  feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)`, then `model.create_predictor(batch_size, device="auto")`.
  `predictor.predict(dataset)` yields GluonTS `SampleForecast` with `samples=(num_samples, H)`; quantiles empirical.
- **Limits**: token budget `max_seq_len=512`, patch sizes 8–128 (`"auto"`) → usable context is `tokens × patch`;
  horizon fixed at construction. Multivariate `target_dim`; native future/past real covariates. Missing values via
  GluonTS observed masks/imputation; `PackedStdScaler`; mixture sampling (non-deterministic). Zero-shot; fine-tuning
  scripts in repo. **Weights CC-BY-NC-4.0**, code Apache-2.0.
- **Friction**: heaviest glue — GluonTS dataset/estimator plumbing, fixed horizon, sampling cost.

### Salesforce Moirai 2.0

- **Id / package**: `Salesforce/moirai-2.0-R-small`; same `uni2ts` (`Moirai2Forecast`/`Moirai2Module`), example
  `prediction_length=100, context_length=1680, target_dim=1`
  ([card](https://huggingface.co/Salesforce/moirai-2.0-R-small)).
- **Calls**: direct multi-token **quantile** head; GluonTS `QuantileForecast` with nine levels 0.1…0.9
  (config `quantile_levels`); deterministic, no sampling. Same `create_predictor(batch_size, device)` route.
- **Limits**: token budget 512, patch 16, `num_predict_token=4`; context/horizon fixed at construction. Multivariate
  and covariate dims via the same constructor; causal-mean imputation; `PackedStdScaler`. Zero-shot. CC-BY-NC-4.0
  weights, Apache-2.0 code.
- **Friction**: simpler than 1.1 (no sampling), still GluonTS-shaped and horizon-at-construction; good default for
  the zoo.

### Datadog Toto 1.0

- **Id / package**: `Datadog/Toto-Open-Base-1.0` (151M); `toto-ts` 0.2.0; `Toto.from_pretrained(id).to(device).eval()`,
  `TotoForecaster(toto.model)` ([card](https://huggingface.co/Datadog/Toto-Open-Base-1.0),
  [forecaster.py](https://github.com/DataDog/toto/blob/main/toto/inference/forecaster.py)).
- **Call**: build `MaskedTimeseries(series=(B,V,T), padding_mask, id_mask, timestamp_seconds,
  time_interval_seconds)` (timestamps mandatory), then `forecast(inputs, prediction_length, num_samples=N,
  samples_per_batch=...)` → `Forecast` with `samples=(B,V,H,N)`, `mean=(B,V,H)`; `.quantile(q)` empirical.
- **Limits**: decoder-only, variable context/horizon (no documented context cap; examples use 4096 —
  **uncertain**); autoregressive per step, cost scales with `N × B × V`. Multivariate; known-future exogenous only
  by putting exogenous channels last and passing `future_exogenous_variables=(B,E,H)`. Missing via masks; internal
  `CausalPatchStdMeanScaler`; Student-T mixture sampling; zero-shot. Apache-2.0.
- **Friction**: synthesise `MaskedTimeseries` timestamps; quantiles need ≥128 samples; sampling must be seeded.

### Datadog Toto 2.0

- **Ids / package**: `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}`; `toto-2` 2.0.0 (**py≥3.12**, torch ≥2.5)
  ([README](https://github.com/DataDog/toto/blob/main/toto2/README.md)).
- **Call**: `Toto2Model.from_pretrained(id).to(device).eval()`;
  `forecast({"target": (B,V,T), "target_mask": bool, "series_ids": (B,V)}, horizon=H,
  decode_block_size=None|N, has_missing_values=bool)` → quantiles `(9, B, V, H)` at levels 0.1…0.9 (median idx 4).
  `decode_block_size=None` is one pass; a block (e.g. 768) trades speed for long-horizon stability. Deterministic
  head; internal `PatchedCausalStdScaler` + asinh; missing via mask. Multivariate. `known_dynamic` covariate inputs
  exist in code but the README says exogenous support is planned — **uncertain**. Zero-shot. Apache-2.0.
- **Friction**: dict input + `series_ids`; Python 3.12 pin; quantile-major tensor with a variate axis.

### Mistral Timestral

- **Ids / package**: `mistralai/Timestral-{5M,28M,123M,209M}`; `timestral`, installed from GitHub (py≥3.10,
  torch≥2.4; **not on PyPI** at the time of writing) ([README](https://github.com/mistralai/timestral),
  [forecaster.py](https://github.com/mistralai/timestral/blob/main/src/timestral/forecaster.py)).
- **Load/call**: `TimestralForecaster.from_pretrained(id, device="cpu")`; `forecast(context=(B,V,T) or (T,),
  prediction_length=H, past_covariates=None, known_covariates=None, observation_mask=None)` → `Forecast` with
  `quantiles=(B,V,H,21)`, `.median=(B,V,H)`, `.quantile(level)`. Levels: `0.01, 0.05, 0.1, 0.15, …, 0.9, 0.95, 0.99`.
- **Limits**: context 32768 (most recent kept); decode block 32 patches = 1024 steps, longer horizons roll with
  median feedback. Univariate/multivariate with past-only, known-future, and categorical covariates (numeric codes,
  target-encoded). NaN marks missing; `observation_mask` adds more. Internal asinh scaling and per-block
  denormalisation; deterministic non-crossing quantiles. Zero-shot. Code Apache-2.0; **weights Modified MIT**.
- **Friction**: clean API, but shapes are always `(B,V,H,Q)` (squeeze twice); 21 levels include 0.01/0.99 tails.

### NX-AI TiRex (v1)

- **Id / package**: `NX-AI/TiRex` (35M; also `TiRex-1.1-gifteval`); `tirex-ts` 1.4.2 (py≥3.10);
  `from tirex import load_model` ([card](https://huggingface.co/NX-AI/TiRex),
  [model.py](https://github.com/NX-AI/tirex/blob/main/src/tirex/models/tirex.py)).
- **Load/call**: `load_model("NX-AI/TiRex", device=None, backend="torch")`; `model.forecast(context,
  prediction_length=H, output_type="torch"|"numpy"|"gluonts", full_rollout=False, dynamic_padding=False)` →
  `(quantiles=(B,H,9), mean=(B,H))`, where mean is the median. Native 0.1…0.9.
- **Limits**: trained context 2048 (docs); implementation pads/truncates to `train_ctx_len`, while
  `max_context_length` defaults to 2016 — **uncertain**; output patch 32, longer horizons roll one patch per pass
  (`full_rollout=True` is faster but degrades). Univariate, no covariates; NaNs masked internally; patch tokenizer
  handles scaling; deterministic. Zero-shot. **NXAI Community License** (non-OSI; check commercial terms).
- **Friction**: simplest quantile API after Chronos-Bolt; watch context padding cost and optional CUDA-kernel compile.

### NX-AI TiRex-2

- **Id / package**: `NX-AI/TiRex-2` (+ `-g`, `-gp`, `-f` decontaminated variants); `tirex2` from
  [`NX-AI/tirex-2`](https://github.com/NX-AI/tirex-2) (Pixi-first, **not on PyPI**)
  ([card](https://huggingface.co/NX-AI/TiRex-2)).
- **Load/call**: `load_model("NX-AI/TiRex-2", device="cpu")`; `forecast([TimeseriesType(target=(V,T),
  past_covariates=None, future_covariates=None)], prediction_length=H, output_type="numpy")` returns one
  `(V, 9, H)` array per series. Native 0.1…0.9; arbitrary levels can be interpolated.
- **Limits**: config `context_len=2048`, `future_len=320` per forward (longer horizons roll); patch 32. Native
  multivariate and past/future covariates (the reason to prefer it over v1). Target-mask handling via
  `TimeseriesType` is **uncertain** — check `types.py` before claiming missing-value support. Deterministic;
  zero-shot. Apache-2.0.
- **Friction**: young repo/API, `(V,9,H)` shape, `TimeseriesType` wrapper per series.

## Optional models

**Lag-Llama** — `time-series-foundation-models/Lag-Llama`, Apache-2.0, no PyPI (git install). GluonTS estimator:
`LagLlamaEstimator(prediction_length=H, context_length=None (10×H), max_context_length=2048, ckpt_path=...,
num_samples=100)`, then `create_transformation()` / `create_lightning_module()` /
`create_predictor(transformation, module)`. `SampleForecast` samples `(num_samples, H)`; quantiles empirical;
univariate, any frequency, horizon fixed at construction; per-step sampling is slow; RoPE scaling recommended
beyond trained context ([estimator.py](https://github.com/time-series-foundation-models/lag-llama/blob/main/lag_llama/gluon/estimator.py)).

**Sundial (Timer 3.0)** — `thuml/sundial-base-128m`, Apache-2.0, via `transformers` remote code:
`AutoModelForCausalLM.from_pretrained(id, trust_remote_code=True)`; `generate(seqs=(B,T), max_new_tokens=H,
num_samples=N)` → samples `(B,N,H)` (shape inferred from `ts_generation_mixin.py` — **uncertain**). Context 2880,
patch 16, multi-patch output 720; generative/flow-matching (non-deterministic), quantiles from ≥20 samples; requires
`transformers==4.40.1`, which conflicts with most other models ([card](https://huggingface.co/thuml/sundial-base-128m)).

**IBM TTM** — `ibm-granite/granite-timeseries-ttm-r2` (r3 exists), Apache-2.0, `granite-tsfm` 0.3.10 (py≥3.11).
Each checkpoint is trained for one `(context_length, prediction_length)` pair, selected by
`get_model(model_path, context_length=..., prediction_length=...)`; inference
`model(past_values=(B,C,ctx), observed_mask=...)` → `prediction_outputs=(B,H,C)`. Point-only for released r2/r2.1
weights; newer modeling code has an optional 9-quantile head but r3 checkpoint support is **uncertain**. External
standard scaling via `TimeSeriesPreprocessor`; zero-shot or few-shot; fixed context/horizon makes it a poor v0.1 fit.

## What this means for the adapter contract

1. **Output shapes vary widely**: `(B,Q,H)`, `(B,H,Q)`, `(Q,B,V,H)`, `(V,Q,H)`, or list-per-series. Every adapter
   must reduce to the contract’s levels-major `(Q,H)` for one series.
2. **Three level families**: nine deciles; 21 levels 0.01–0.99 (Chronos-2, Timestral); none, derive from samples
   (Chronos v1, Moirai 1.1, Toto 1, Lag-Llama, Sundial). Native-level `predict_proba` is the honest choice;
   `interval(0.95)` must raise for nine-decile models rather than clamp.
3. **Point semantics differ**: most models already expose a median column; TimesFM 2.5 has a separate point head
   plus a mean column 0; Chronos v1 and Toto 1 return sample means unless the adapter takes the sample median.
4. **Horizon policy is the main per-adapter decision**: fixed (Chronos-Bolt 64, TTM checkpoint), compile-set
   (TimesFM 2.5), construction-set (Moirai, Lag-Llama), or free (Chronos-2, TimesFM 3, Toto, Timestral, TiRex).
   Declare `max_horizon`; TimesFM 2.5 either caps at the compiled horizon or recompiles (slow).
5. **Context caps** range 512–2048 (older univariate) through 8192 (Chronos-2), 15360 (TimesFM 3), 32768
   (Timestral); truncating to the last `max_context` points is native behaviour everywhere.
6. **Missing values are native in most SDKs but not all**: Chronos/TimesFM/Timestral/TiRex accept NaN; Moirai and
   TTM need adapter-side imputation — declare `missing_values=False` for those in v0.1.
7. **Determinism**: quantile-head models are deterministic; sample-based models are not, so `.samples` needs a
   fixed seed/num_samples or documented run-to-run variation.
8. **Environment isolation is not optional**: `toto-2` needs py3.12, Sundial needs `transformers==4.40.1`, `uni2ts`
   pulls GluonTS/Lightning, `granite-tsfm` needs py≥3.11, and `timestral`/`lag-llama`/`tirex2` are git-only. The
   `package` hint should be the exact install string, not just the import name.
9. **Licences constrain defaults**: Apache-2.0 (Chronos, Toto, TiRex-2, Lag-Llama, Sundial, TTM), CC-BY-NC-4.0
   (Moirai), Modified MIT (Timestral), NXAI Community (TiRex v1), non-commercial (TimesFM 3 weights). Only
   Apache-2.0/Modified-MIT models are safe defaults for a distributed package.

## Uncertainties / follow-ups

- Toto 1: no documented max context (only the 4096-step example).
- TiRex v1: docs say 2048 trained context, code property defaults to 2016.
- Toto 2: `known_dynamic` exists in code, announced as planned in the README.
- TTM: r2/r2.1 are point-only; the 9-quantile head in newer modeling code was not confirmed for r3 checkpoints.
- TimesFM 3: no documented upper horizon bound beyond the 15360 context truncation.
- Sundial: `generate` output shape inferred from source; confirm by running the quickstart.
- TimesFM 2.5: no public `device=` override; use `CUDA_VISIBLE_DEVICES` to force CPU.
