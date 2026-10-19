# Prior art: forecasting APIs and what they teach a minimal contract

Scope: sktime, Darts, GluonTS, Nixtla NeuralForecast, Hugging Face Transformers, scikit-learn. Read at library `main` plus released docs (sktime 1.2; Darts master; GluonTS dev; NeuralForecast main; Transformers main) on 2026-10-20. Only contract lessons are extracted; no code was changed.

## Comparison at a glance

| Library | Fit vs zero-shot | Probabilistic output | Capability signalling | Dependency / loading | Notable trap |
|---|---|---|---|---|---|
| sktime | `fit(y, X, fh)`, required except `fit_is_empty` | quantile/interval DataFrames; `predict_proba` → skpro distribution | tags, `capability:pred_int` | soft deps via `python_dependencies`; skpro optional | four overlapping probabilistic methods |
| Darts | `fit(series, ...)`; foundation models zero-shot but still call `fit()` | samples (`num_samples`), likelihood params, derived quantiles | `supports_*` attrs + `likelihood` | `[torch]` extra; HF Hub cache, `local_dir` | ceremonial `fit()`; heavy torch |
| GluonTS | `Estimator.train(...) → Predictor`; some predictors zero-shot | `SampleForecast`, `QuantileForecast` (keys-major) | none; output type is the capability | `[torch]`/`[prophet]`/`[shell]` extras; serialize | fixed `prediction_length`; NaN for missing quantiles |
| NeuralForecast | `fit(df)`; horizon in constructor | long df with `-lo-/-hi-` or `_ql` columns | none; model class + loss | torch hard dep; safetensors + restricted unpickler | `level` is 0–100, `quantiles` 0–1; `fit` returns `None` |
| HF Transformers | pretrained only; no `fit` | `sequences` tensors or TimesFM tensors | none | `from_pretrained` + HF cache | no forecasting pipeline; TimesFM-only auto mapping |
| scikit-learn | `fit(X, y)` returns self; `requires_fit` allows stateless | `predict_proba` = class probabilities | public tags since 1.6; `available_if` | params/clone only | `predict(X)`, not `predict(h)` |

## sktime

Surface: `fit(y, X=None, fh=None)`, `predict(fh=None, X=None)`, `predict_quantiles(fh=None, X=None, alpha=None)`, `predict_interval(fh=None, X=None, coverage=0.90)`, `predict_proba(fh=None, X=None, marginal=True)`, plus `predict_var`, `update`, `score`. The horizon may be passed at fit or predict, governed by `requires-fh-in-fit`.

Capabilities are introspectable tags: `capability:pred_int` (default `False`) gates all four probabilistic methods; `capability:pred_int:insample`, `capability:exogenous`, `capability:multivariate`, `fit_is_empty` and `python_dependencies` cover the rest. Tag values are inherited and overridable per instance, and `_DelegatedForecaster._set_delegated_tags` clones capability tags from a wrapped estimator. The brief's `_PredictProbaMixin` no longer exists under that name; the equivalent machinery is `_DelegatedForecaster` in `_delegate.py` (delegates `_predict_interval`, `_predict_quantiles`, `_predict_var`, `_predict_proba` and clones tags) and `BaseProbaForecaster` in `_base_proba.py` (vectorized `predict_proba`). The public contract is still "`predict_proba` gated by `capability:pred_int`".

Representation: quantiles are a DataFrame with MultiIndex columns `(variable, alpha)`, intervals `(variable, coverage, lower|upper)`; a full distribution is an skpro `BaseDistribution`, the only piece needing an optional heavyweight dependency (absent skpro yields a warning, then `ImportError` from the default `_predict_proba`). Defaults differ per method (`coverage=0.90`, `alpha=[0.05, 0.95]`), so "the default interval" is not one convention.

Lesson: declaration-driven gating is right; four methods plus a distribution-object dependency is the wrong size for a thin library.

## Darts

`ForecastingModel.fit(series, verbose=None)` / `predict(n, num_samples=1, ...)`; `GlobalForecastingModel.fit(series, past_covariates=None, future_covariates=None, ...)` / `predict(n, series=None, past_covariates=None, future_covariates=None, num_samples=1, predict_likelihood_parameters=False, ...)`; `historical_forecasts(series, ..., forecast_horizon=1, start=None, stride=1, retrain=True, last_points_only=True, ...)`, where `retrain=False` is explicitly the pre-trained/zero-shot mode.

`FoundationModel` is an abstract subclass of `MixedCovariatesTorchModel` with a Lightning backend. Concrete models (Chronos-2, TimesFM, TiRex) are documented as zero-shot, download weights through `hf_connector` from `hub_model_name`/`hub_model_revision` into `local_dir` or the HF cache, and optionally fine-tune under `enable_finetuning`. Yet `predict` checks `_fit_called` and raises `ValueError("The model must be fit before calling predict().")` — zero-shot still requires a ceremonial `fit()`.

Output is sample-first: `num_samples > 1` raises `ValueError` unless `supports_probabilistic_prediction`; a `likelihood` (`QuantileRegression(quantiles=[...])`, `GaussianLikelihood`, …) defines the distribution; `predict_likelihood_parameters=True` returns likelihood parameters as `TimeSeries` components; quantiles come from `TimeSeries.quantile(q)`. Capabilities are class attributes/properties (`supports_probabilistic_prediction`, `supports_{past,future,static}_covariates`, `likelihood`) and unsupported covariates fail with a clear `ValueError` at fit. Extras are split: core install excludes torch; `darts[torch]` adds torch/Lightning/huggingface-hub/safetensors; `darts[notorch]` adds sklearn-style backends.

## GluonTS

`Estimator.train(training_data, validation_data=None) → Predictor`, with `prediction_length` a constructor/class attribute. `Predictor(prediction_length, lead_time=0)` exposes `predict(dataset, **kwargs) → Iterator[Forecast]` and `serialize`/`deserialize`; `RepresentablePredictor.predict_item(item)` handles one entry. Data is a `ListDataset` of dicts with `start` (a `pd.Period`) and `target` (`np.ndarray`). Zero-shot predictors (seasonal naive, trivial predictors, some StatsForecast wrappers) are predictors constructed directly, bypassing `Estimator`; GluonTS ships no foundation-model wrappers.

`QuantileForecast(forecast_arrays, start_date, forecast_keys, item_id=None, info=None)` stores a `(len(keys), prediction_length)` array — keys-major, the same layout `forecasters` uses. Keys are normalized strings (`"0.1"`, …, plus optional `"mean"`); `SampleForecast` stores samples instead. `Forecast.quantile(q)` interpolates between available keys with an exponential-tail approximation and returns `NaN` when the key is missing or fewer than two quantiles exist.

No capability tags: the output type tells you what a predictor produces. `prediction_length` is fixed at construction, so per-call horizons are not representable. `ParallelizedPredictor(base_predictor, num_workers, chunk_size)` is the only prior art for worker processes; it propagates worker exceptions but is documented to hang with tqdm when one occurs — weak evidence that process pools around model objects are fragile, which supports a tiny JSON-lines protocol over pickling predictors.

## Nixtla NeuralForecast

`NeuralForecast(models, freq, local_scaler_type=None)`; `fit(df=None, static_df=None, val_size=0, ..., prediction_intervals=None) -> None`; `predict(df=None, futr_df=None, level=None, quantiles=None, h=None)`; `predict_insample(step_size=1, level=None, quantiles=None)`; `cross_validation`; `save`/`load`. Horizon is a model constructor argument (`h=`), asserted equal across models, and `predict(h=...)` can only *increase* it, not with historic exogenous features or conformal intervals.

Probabilistic parameters are the sharpest naming trap here: `level` is confidence percentage (0–100), `quantiles` is probability (0–1), and both together raise `ValueError`. Output columns are `{model}`, `{model}-lo-80`, `{model}-hi-80`, `{model}-median`, or `{model}_ql0.1`. Error styles are inconsistent: `ValueError` for horizon mismatch, `NotImplementedError` for multivariate in-sample and conformal in-sample, and a bare `Exception("You must fit the model before predicting.")` when unfitted. Loading is careful: safetensors + JSON for new directories, legacy pickles refused unless `allow_pickle=True`, remote paths require `trust_remote=True`; torch is a hard dependency with only `[spark]` as an extra.

## Hugging Face Transformers

There is no forecasting pipeline: `SUPPORTED_TASKS` has no time-series/forecasting task, and the prediction-head auto mapping (`AutoModelForTimeSeriesPrediction`) lists TimesFM variants only. Forecasting is model-class-only: `TimeSeriesTransformerForPrediction`, `PatchTSTForPrediction`, `InformerForPrediction`, `TimesFmModelForPrediction`, `TimesFm2_5ModelForPrediction`. Inputs are torch tensors; outputs disagree by family — `SampleTSPredictionOutput.sequences` is `(batch, num_samples, prediction_length)`, while TimesFM returns `mean_predictions`/`full_predictions` under config (`context_length=512`, `horizon_length=128`, fixed quantile grid 0.1…0.9). Loading is `from_pretrained` with the HF cache; there are no capabilities, no `fit`, and no evaluation contract. Even the largest model hub declines to standardize forecasting inference — per-model adapters are the state of the art, not a smell.

## scikit-learn

`fit(X, y)` returns `self`; `predict(X)` returns array-like; `predict_proba(X)` exists for classifiers only, `(n_samples, n_classes)` ordered by `classes_`. `get_params`/`set_params` expose constructor parameters, `clone()` rebuilds a fresh unfitted estimator, and `check_is_fitted` looks for trailing-underscore attributes, supports `__sklearn_is_fitted__`, and honours `requires_fit=False` (stateless/zero-shot). Optional methods are hidden with `available_if`, so `hasattr` is the de-facto capability probe; tags became public in 1.6 via `__sklearn_tags__`.

The valuable half is contract vocabulary: `requires_fit`, `clone`, `check_is_fitted`, and errors that name what is missing. The incompatible half is every signature: `fit(X, y)`/`predict(X)` assume in-memory sample matrices, cheap construction, and parameter-only state. A process-backed, weight-backed, zero-shot forecaster satisfies none of those assumptions.

## Cross-cutting lessons

- **Horizon.** Four encodings: `fh` (sktime, may be absolute), `n` (Darts, relative steps), `prediction_length` (GluonTS, fixed), `h` (NeuralForecast, fixed at fit). Per-call relative `h` is cleanest, but every adapter needs `max_h`/context limits and a `h > max_h` rule.
- **Level units.** sktime probabilities (0–1), NeuralForecast percentages (0–100), GluonTS string keys, Darts a quantile list on the likelihood. State units and ordering, and validate eagerly.
- **Axis order.** GluonTS and `forecasters` are levels-major; HF has a sample axis; Darts stores samples in the value array. The likeliest silent-corruption bug in an adapter.
- **Unsupported paths.** `NotImplementedError` (sktime, partly NeuralForecast), `ValueError` (Darts), silent `NaN` (GluonTS); none typed. The existing typed error hierarchy is better; a capability-specific subclass is the missing piece.
- **Dependencies.** Extras per backend family (Darts, GluonTS), hard dependencies (NeuralForecast), lazy soft-dependency checks (sktime), hub cache + `local_dir` (Darts, HF), safetensors + restricted unpickling + no remote by default (NeuralForecast). The per-model interpreter (`ModelSpec.python`) is stronger isolation than any of these because it makes conflicts impossible rather than optional.

## Copy this

1. **Capability metadata on the model spec, introspectable before loading**: native quantile grid, max horizon, context window, multivariate/exogenous support, fit mode, install hint. sktime tags and Darts `supports_*` are the models; sklearn's `requires_fit` is the fit-mode vocabulary.
2. **One probabilistic method, one shape.** A quantile grid as `(len(levels), horizon)` (already `Forecast.values`) beats sktime's proba/quantiles/interval/var quartet; `median`/`interval` are conveniences on the result, not protocol calls.
3. **Derive capabilities from the wrapped model, don't hardcode them** — copy sktime's tag cloning so an adapter reports what its SDK actually supports.
4. **Typed, actionable errors** naming the capability and the exact extra, as sktime soft-dependency checks and Darts covariate errors do; never `NaN` for an unavailable quantile.
5. **Zero-shot as a first-class mode**, no ceremonial `fit()`; sklearn `requires_fit=False` and sktime `fit_is_empty` supply the vocabulary.
6. **Adapter-owned loading policy**: pinned revision, configurable cache, optional `local_dir`, remote loading off by default, safe artifact formats (safetensors/JSON over pickle).
7. **Explicit level normalization at the boundary**: accept only 0–1 probabilities, keep the model's grid in `native_levels`, and reject out-of-grid requests deterministically (current `Forecast.quantile` already does this).
8. **Per-model interpreter isolation** as the primary dependency strategy, SDK imported lazily inside the worker, so a missing package fails as `ModelUnavailable` at startup rather than an `ImportError` mid-forecast.

## Avoid this

1. **Calling `fit(y)`/`predict(h)` "sklearn-like".** sklearn is `fit(X, y)`/`predict(X)`, and the names promise training state and sample-wise prediction a zero-shot model lacks. Keep the stateless `forecast(key, y, h)`; if a fine-tunable adapter needs training, add `fit(y)` there and return `self`, not `None`.
2. **`predict_proba` for quantiles.** In sklearn it means class probabilities; in sktime a distribution object. Use `predict_quantiles`/`forecast`.
3. **Bare `levels` with no stated units.** NeuralForecast's `level` is 0–100; name the parameter `quantiles`/`alphas` and document the range.
4. **Multiple overlapping probabilistic entry points plus a heavyweight optional return type** (sktime's four methods and skpro).
5. **Silent `NaN` or extrapolation** outside a model's grid (GluonTS); raise with the available grid in the message.
6. **Horizon fixed at fit/construction** (NeuralForecast, GluonTS) while advertising per-call `h`; pick per-call `h` and handle stitching and limits in adapters.
7. **Pickle as the artifact format** (NeuralForecast legacy path) — it executes code on load and makes cross-environment workers unsafe.
8. **One giant `all_extras` install** (sktime); it defeats the purpose of worker isolation.

## Open questions for the Lead

1. **Is `fit` in scope?** Current code is stateless `Forecast`/`Worker.submit`. Recommendation: no `fit` in v1; declare `fit_mode: "zero-shot"` in `ModelSpec.metadata`, and let a future fine-tuning adapter add it without changing `Forecast`.
2. **API name and units.** `predict_quantiles(h, quantiles=[...])` vs the existing `levels` argument: keep `levels` (0–1) or rename, and accept 0–100 anywhere?
3. **Is the quantile grid the only currency?** Prior art is mostly sample-first (Darts, HF) or distribution-first (sktime); the protocol is quantile-only. Confirm no samples/distributions in v1.
4. **Minimum capability surface.** Proposed `ModelSpec.metadata`: `native_levels`, `max_h`, `context_window`, `multivariate`, `exogenous`, `fit_mode`, `install_hint`, `model_id`. Which are required vs optional?
5. **Unsupported semantics.** Add `Unsupported(ForecasterError, NotImplementedError)`, raised in the supervisor when known and in the worker otherwise? Or reuse `AdapterError`?
6. **`h > max_h` policy.** Reject, truncate, or autoregress in the adapter? TimesFM's fixed 128, Chronos (Darts warns zero-shot differs beyond 1024) and Darts' `output_chunk_length` mean the answer differs per model; pick one contract.
7. **Out-of-grid levels.** Keep "interpolate within native range, error outside" and expose `native_levels`, or reject anything off-grid? GluonTS interpolates then NaNs; NeuralForecast cannot request off-loss quantiles at all.
8. **Multivariate scope.** All surveyed frameworks except our protocol support multivariate input; if v1 is univariate-only, declare it and reject 2-D `y` explicitly.

## Sources

- sktime `BaseForecaster`: <https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html>; tags: <https://www.sktime.net/en/stable/api_reference/tags.html>; sources: <https://github.com/sktime/sktime/blob/main/sktime/forecasting/base/_base.py>, `_delegate.py`, `_base_proba.py`; estimator guide: <https://www.sktime.net/en/stable/developer_guide/add_estimators.html>
- Darts guide: <https://unit8co.github.io/darts/userguide/forecasting_overview.html>; API: <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html>; sources: <https://github.com/unit8co/darts/blob/master/darts/models/forecasting/forecasting_model.py>, <https://github.com/unit8co/darts/blob/master/darts/models/forecasting/foundation_model.py>, <https://github.com/unit8co/darts/blob/master/darts/models/forecasting/chronos2_model.py>
- GluonTS: <https://ts.gluon.ai/stable/api/gluonts/gluonts.model.predictor.html>, <https://ts.gluon.ai/stable/api/gluonts/gluonts.model.estimator.html>, <https://ts.gluon.ai/stable/api/gluonts/gluonts.model.forecast.html>, <https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html>
- NeuralForecast: <https://nixtlaverse.nixtla.io/neuralforecast/core.html>, <https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/predictInsample.html>, <https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/save_load_models.html>, <https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/uncertainty_quantification.html>
- Transformers: <https://huggingface.co/docs/transformers/main/en/model_doc/time_series_transformer>, <https://huggingface.co/docs/transformers/main/en/model_doc/timesfm>, <https://huggingface.co/docs/transformers/main/en/main_classes/pipelines>
- scikit-learn: <https://scikit-learn.org/stable/developers/develop.html>, <https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html>, <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>, <https://scikit-learn.org/stable/modules/generated/sklearn.utils.metaestimators.available_if.html>
