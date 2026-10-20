# Forecasters

Forecasters compares pretrained time-series forecasts through a common interface.

## Language

**Context**:
The observed history of one time series, ordered from oldest to newest.

**Horizon**:
The number of consecutive future observations to forecast after the context.

**Median forecast**:
The pointwise predictive quantile at probability 0.5. For sampled forecasts it is an estimate from the generated trajectories; it is not necessarily the mean or a coherent sampled trajectory.
_Avoid_: Mean forecast as a synonym, median interval.

**Quantile level**:
A probability between zero and one identifying a predictive quantile, such as 0.5 for the median.
_Avoid_: Percentage without stated units.

**Prediction interval**:
A range of future values bounded by predictive quantiles. A central 80% interval has endpoints at levels 0.1 and 0.9; its nominal coverage does not imply empirical calibration.
_Avoid_: Confidence interval for future observations.

**Native forecast**:
The output supplied by a model's own inference implementation, before Forecasters normalizes representation or selects a statistic.

**Zero-shot forecasting**:
Forecasting a supplied context with pretrained weights without training or fine-tuning on that context.
