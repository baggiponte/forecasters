# Forecasters research snapshot — 2026-10-20

Bounded, median-only snapshot. I checked current primary READMEs/model cards with `curl --max-time 20`; I did **not** download weights, run benchmarks, or audit source files. `design/model-matrix.md` is treated as historical, unverified prior research. “Verified” below means visible on the linked primary page; “prior” means carried over and not rechecked here.

## Findings

For the accepted v0 contract — direct typed `predict(context, horizon=...)`, median first, no public quantiles yet — split candidates by how the median is obtained:

- **Native 0.5 output, best fit:** Chronos-Bolt Tiny, Timestral-5M, Toto-2.0-4m, TimesFM 2.5/3.0, Moirai-2.0-R-small, t0-alpha.
- **Sample-derived median, avoid for v0:** original Chronos, Moirai 1.1, Toto 1.0, Lag-Llama, Sundial.
- **No median / unknown:** TTM, Timer, MOMENT, Time-MoE are point-only; TimeGPT is hosted/closed with unclear median semantics; TabPFN-TS has probabilistic features but no verified median API.

The two small candidates remain sensible, with one important correction to expectations: **“runs on CPU” is documented for both, but neither has a measured CPU latency here.** Chronos-Bolt Tiny is the simpler first adapter. Timestral-5M is the stronger second candidate but has a much longer context/window story.

**TimesFM 2.5 caveat:** its separate point/mean column does **not** mean it is mean-only. The card returns `quantile_forecast` as mean + 10th–90th quantiles; the 0.5 column supplies the median. Never take column 0 as the median.

## Family table

Smallest checkpoint is the smallest documented/verified member I found, not every checkpoint. “Local” means weights can be run in-process.

| Family / version | Native SDK or package | Median source | Smallest verified checkpoint | Local / hosted | Primary source |
|---|---|---|---|---|---|
| Mistral **Timestral** 5M–209M | `timestral` (git; PyTorch) | **native 0.5** (21 quantiles) | **5M** | local CPU/GPU | [README](https://github.com/mistralai/timestral) |
| Amazon **Chronos-Bolt / Chronos-2 / Chronos-1** | `chronos-forecasting` | **native 0.5** for Bolt/2; **samples** for Chronos-1 | Bolt Tiny **9M**; Chronos-1 Tiny 8M; Chronos-2 Small 28M | local CPU/GPU | [README](https://github.com/amazon-science/chronos-forecasting) |
| Google **TimesFM 2.5** | `timesfm[torch]` (2.5 class) | **native 0.5** from quantile output; separate mean col | **200M** | local | [card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| Google **TimesFM 3.0** | `timesfm` ≥3 (`timesfm3`) | **native 0.5** (9 deciles, index 4) | **330M** | local | [card](https://huggingface.co/google/timesfm-3.0-pytorch) |
| Datadog **Toto 2.0** | `toto-models` | **native 0.5** (9 quantiles) | **4M** | local | [README](https://github.com/DataDog/toto) |
| Datadog **Toto 1.0** (legacy) | `toto-ts` | **samples** | 151M | local | [README](https://github.com/DataDog/toto) |
| Salesforce **Moirai 2.0** | `uni2ts` + GluonTS | **native 0.5** (9 quantiles) | **~11M** (R-small) | local | [README](https://github.com/SalesforceAIResearch/uni2ts) |
| Salesforce **Moirai 1.1** | `uni2ts` + GluonTS | **samples** | R-small (size not rechecked) | local | [README](https://github.com/SalesforceAIResearch/uni2ts) |
| The Forecasting Company **t0-alpha** | `tfc-t0[torch]` / `[mlx]` | **native 0.5** (trained at 0.1/0.25/0.5/0.75/0.9) | **102M** | local; also hosted via Retrocast | [card](https://huggingface.co/theforecastingcompany/t0-alpha) |
| The Forecasting Company **t0-beta / t0-gamma** | `tfc-t0[torch]` / `[mlx]` | **native 0.5**, 21 native levels | **256M** each | local; also hosted via Retrocast | [beta card](https://huggingface.co/theforecastingcompany/t0-beta), [gamma card](https://huggingface.co/theforecastingcompany/t0-gamma) |
| NX-AI **TiRex v1** | `tirex-ts` | **native 9 quantiles**; point mean semantics not stated on card | **35M** | local | [card](https://huggingface.co/NX-AI/TiRex) |
| NX-AI **TiRex-2** | `tirex2` (git, lead) | **native 0.5** per prior note; not re-fetched | unknown | local | [repo](https://github.com/NX-AI/tirex-2) |
| Tsinghua **Sundial / Timer** | `transformers` remote code | Sundial **samples**; Timer **unavailable** (point-only) | Sundial **128M**; Timer **84M** | local | [Sundial card](https://huggingface.co/thuml/sundial-base-128m), [Timer card](https://huggingface.co/thuml/timer-base-84m) |
| **Lag-Llama** | `lag-llama` (git) + GluonTS | **samples** | HF checkpoint ~2.4M stored parameters | local | [card](https://huggingface.co/time-series-foundation-models/Lag-Llama) |
| IBM **TTM** r2/r2.1 | `granite-tsfm` | **unavailable** (point-only) | **~1M** class | local | [card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| Prior Labs **TabPFN-TS** 1.3 | `tabpfn-time-series` | **unknown** (probabilistic path exists; median API not verified) | unknown (TabPFN-3.5 size not documented) | local + cloud client | [README](https://github.com/PriorLabs/tabpfn-time-series) |
| **MOMENT** 1.x | `momentfm` | **unavailable/unknown** (forecast head is point) | **38M** (small) | local | [card](https://huggingface.co/AutonLab/MOMENT-1-small) |
| **Time-MoE** | `transformers` remote code | **unavailable** (point-only autoregressive) | TimeMoE-**50M** name | local | [README](https://github.com/Time-MoE/Time-MoE) |
| Nixtla **TimeGPT 2.1** | `nixtla` SDK | **unknown** (closed; point + intervals) | n/a (closed) | hosted only | [README](https://github.com/Nixtla/nixtla) |
| LG AI **EXAONE Finance / Demand 1.0** | `exaone-forecast` (git) | **native 0.5** (21 quantiles) | unknown | local | [README](https://github.com/LGAI-Research/EXAONE-Forecast) |
| **Tabby** | `transformers` remote code | **native 0.5** (99 quantiles, index 49) | **147M** | local | [card](https://huggingface.co/paris-noah/Tabby) |

## Requested six, briefly

**Timestral.** Current family is 5M/28M/123M/209M. The API is unusually close to what v0 needs: `TimestralForecaster.from_pretrained(...)`, `forecast(context, prediction_length=H)` returns `(B,V,H,21)` quantiles and a `.median` convenience property. Native levels include 0.5. Context up to 32,768; one forward pass covers 1,024 steps, then later blocks roll using the **median** as feedback — compatible with median-only operation. CPU/GPU documented. Code Apache-2.0; weights Modified MIT.

**Chronos.** Current README presents Chronos-2 as the latest, but Chronos-Bolt is still supported and is the lightweight median candidate. Chronos-Bolt Tiny is 9M and returns quantile forecasts directly; the 0.5 column is the median. Native horizon is 64; longer horizons roll using the nine quantiles. Original Chronos returns samples, so median would require sampling. Apache-2.0. Chronos-2 Small is 28M with richer covariates but more surface area; not the smallest first step.

**TimesFM.** 2.5 is 200M, Apache-2.0 weights, context up to 16k, with a separate point head and a quantile array of mean + deciles. The median is the 0.5 quantile, not the mean. 3.0 is 330M, native multivariate/covariates, nine deciles with median at index 4, context ~15,360, but **weights are non-commercial** while code is Apache-2.0. For a median-first local package, TimesFM 2.5 is the safer licence/default; 3.0 is a capability upgrade with a distribution constraint.

**Toto.** Toto 2.0 is the recommended current release; `toto-models` exposes `Toto2Model.forecast(...)` returning quantiles `(9,B,V,H)` at 0.1–0.9, so median is native. Smallest is 4M, Apache-2.0, variable context/horizon, local. Toto 1.0 is legacy, sample-based (`toto-ts`), and only relevant if its covariate/fine-tuning path is needed now. The prior matrix’s `toto-2` package name is stale relative to the current README.

**Moirai / uni2ts.** Moirai 2.0 uses a deterministic quantile head with nine levels; Moirai 1.1 is sample-based. Both go through GluonTS predictors with horizon fixed at construction. Moirai 2.0 R-small is ~11M and is the practical Moirai median candidate. Weights are CC-BY-NC-4.0 (code Apache-2.0), so not a safe default for a distributed commercial package.

**t0.** `t0-alpha` is an open 102M Apache-2.0 checkpoint with native 0.5 output (trained quantiles 0.1/0.25/0.5/0.75/0.9), local PyTorch/MLX runtimes, and hosted Retrocast access. Horizons up to 1,024 decode in one pass; longer horizons autoregress. The orchestrator additionally checked the [beta](https://huggingface.co/theforecastingcompany/t0-beta) and [gamma](https://huggingface.co/theforecastingcompany/t0-gamma) primary cards: both are local open 256M checkpoints under Apache-2.0, with 21 native levels and an explicit median output. Beta requires `tfc-t0[torch]>=0.5.0`, gamma `>=0.6.0`; both document MLX `>=0.6.0`. They are not hosted-only models. Alpha remains the smallest verified t0 checkpoint in this snapshot.

## Adjacent families

**TiRex.** v1 is 35M, `tirex-ts`, native quantiles, local, NXAI community licence. The card does not explicitly say the returned “mean” is the median; the prior matrix says it is, so treat that as prior unless confirmed in code. TiRex-2 is the Apache-2.0 successor and adds covariates, but it was not re-fetched here; keep it as a lead.

**Sundial / Timer.** Sundial is generative: `generate(..., num_samples=N)` yields samples; quantiles/median are empirical. Context 2,880, multi-patch 720, Apache-2.0, but requires `transformers==4.40.1`, which is an environment-isolation cost. Timer is point-only, so median is unavailable.

**Lag-Llama.** GluonTS sample-based, Apache-2.0, local, horizon fixed at estimator construction. Median requires sample quantiles and is slower/non-deterministic. Not a v0 median-first choice.

**TTM / PatchTST-FM.** TTM r2/r2.1 is point-only, ~1M-class, Apache-2.0, local, and each checkpoint is tied to a context/prediction-length pair. Median is unavailable. PatchTST-FM was not verified in this pass; mark as a lead rather than adding it to the matrix.

**TabPFN-TS.** v1.3 uses TabPFN-3.5 locally or via cloud client; it supports point and probabilistic forecasting, but the README does not expose a verified median API or checkpoint size. Treat median extraction and smallest checkpoint as unknown.

**MOMENT.** 38M small, MIT, `momentfm`, local. The forecasting head is point-oriented in the card; no median path verified. Its [original project](https://moment-timeseries-foundation-model.github.io/) reports linear probing for forecasting: do not assume that a general pretrained representation checkpoint is a ready zero-shot forecasting SDK.

**Time-MoE.** Decoder-only, point-only autoregressive, Apache-2.0, context+horizon ≤4096, local. The “50M” name is the documented small checkpoint; HF metadata shows ~113M stored parameters. Median unavailable.

**TimeGPT.** Hosted/closed; the `nixtla` SDK is Apache-2.0 but all inference is API-side. Point and interval outputs are documented; median semantics are not. No local checkpoint and no smallest-local-checkpoint answer.

**EXAONE.** Finance and Demand 1.0 checkpoints, 21 quantiles with a `.point()` median, local, CPU works but GPU recommended. Code BSD-3-Clause-LG AI Research; weights EXAONE AI Model License 1.2-NC. Checkpoint sizes are not documented on the README.

**Tabby.** 147M, 99 quantiles 0.01–0.99, median index 49, context window 8,192, local `transformers` remote code, CC-BY-NC-4.0. Good median coverage but non-commercial weights.

## Practical median-only decisions

1. **Ship the smallest native-median pair first.** Chronos-Bolt Tiny (9M, Apache-2.0, horizon 64 native, context 2,048) is the least ambiguous adapter. Timestral-5M (5M, Modified MIT weights, horizon 1,024 per block, context 32,768) is the second, with the caveat that long context can dominate CPU time.
2. **Do not equate CPU support with measured latency.** Both candidates document CPU inference; neither page gives a representative end-to-end latency for this adapter. A latency check is a separate benchmark task.
3. **Do not make sample models median-first.** For original Chronos, Moirai 1.1, Toto 1, Lag-Llama, and Sundial, median-first means drawing samples and taking an empirical median; that adds nondeterminism and cost. Prefer native-quantile versions.
4. **Keep quantiles internal.** The accepted public API exposes only the median. For the first pair, extract native 0.5; never substitute a mean or fabricate a median from neighbouring levels. Later sample-based adapters may explicitly estimate the median from native samples with documented configuration, as ADR 0001 proposes. A truly point-only mean supplies no median.
5. **Treat horizon limits as adapter policy.** Chronos-Bolt caps at 64 natively; Timestral handles longer horizons blockwise with median feedback; TimesFM requires compiled context/horizon limits; Moirai/TTM fix horizon at construction. Typed `predict(context, horizon=...)` should either honour a documented roll policy or reject with a clear limit.
6. **Default to local Apache-2.0/Modified-MIT weights.** Avoid CC-BY-NC (Moirai, Tabby), non-commercial (TimesFM 3.0, EXAONE), and community licences (TiRex v1) as defaults. TimeGPT and TabPFN cloud paths add network dependencies.
7. **Correct the prior matrix where verified:** Toto 2 uses `toto-models`; t0-alpha is a real Apache-2.0 local median model; TimesFM 3 weights are non-commercial; Chronos-2 is current but Bolt remains the small median option.

## Verification status and gaps

Verified from primary pages: family existence/versioning, native package names, median/quantile shape claims where quoted, documented context/horizon, CPU support, and code-vs-weight licences above. Verified checkpoint parameter counts via Hugging Face model metadata where listed.

Not verified here: exact PyPI versions for every SDK; TiRex v1 “mean is median”; TiRex-2 current API/licence; TabPFN-TS median semantics; EXAONE checkpoint sizes; TimeGPT median semantics; PatchTST-FM; any CPU/GPU latency or accuracy benchmark; whether every checkpoint in a family shares the same levels. These remain prior notes or open questions, not claims.

No other files were edited; no weights downloaded; no benchmarks run.

## Existing wrappers and further leads

[FoundationForecast](https://github.com/TimeCopilot/foundationforecast) already supplies a common interface
for many of these families; see the [interface comparison](interface-options.md) for the trade-off against
its DataFrame contract and broad runtime dependencies. Its catalog also identifies FlowState and Tafsut.
Those two are additional research leads here: their native interfaces and median semantics have not been
verified in this pass, so they are not claimed as supported models. This snapshot is broad but not exhaustive.
