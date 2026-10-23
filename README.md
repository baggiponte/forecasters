# 🔮 Forecasters

A small interface for pretrained time-series models, inspired by 🤗 Transformers. Supports Chronos-Bolt and Timestral through their native SDKs.

```bash
uv add 'forecasters[chronos]'
# For Timestral, also install its official SDK:
uv add 'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
```

```python
import numpy as np
from forecasters import ChronosBolt

model = ChronosBolt.from_pretrained()
context = np.sin(np.arange(64, dtype=np.float32) / 5)

median = model.predict(context, horizon=8)  # (8,)
quantiles = model.predict(context, horizon=8, quantile_levels=[0.1, 0.5, 0.9])  # (8, 3)
```

`predict` preserves the input's NumPy or Torch dtype and device. Requested quantiles must be available in the loaded checkpoint.

## Writing an adapter

Subclass `BaseForecaster`, declare the checkpoint metadata and native tensor layout in `__init__`, and implement `_predict_native` with the SDK call. The inherited `predict` supplies validation, exact native shape checks, quantile selection and input/output representation preservation. No registration, decorator or repeated overloads are needed.

See the [authoring guide and architecture comparison](https://github.com/baggiponte/forecasters/blob/main/docs/adapter-design.md) for a two-method example and the extension locations for future observation and dataframe support. Neither feature is implemented yet.

## Development

Run `just setup` to install development dependencies and hooks; `just --list` lists the available commands.
