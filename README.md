# 🔮 Forecasters

A small, typed interface for pretrained time-series models, inspired by 🤗 Transformers.
Supports Chronos-Bolt and Timestral through their native SDKs.

```bash
uv pip install 'forecasters[chronos]'
```

```python
import numpy as np
from forecasters import ChronosBolt

model = ChronosBolt.from_pretrained()
context = np.sin(np.arange(64, dtype=np.float32) / 5)

median = model.predict(context, horizon=8)  # (8,)
quantiles = model.predict(context, horizon=8, quantile_levels=[0.1, 0.5, 0.9])  # (8, 3)
```

NumPy in, NumPy out; Torch in, Torch out, preserving dtype and device.
Forecasts use native quantiles, without interpolation or training.

See the [package documentation](packages/forecasters/README.md) for Timestral installation
and the API contract.

## Development

```bash
uv sync --locked
prek install --allow-missing-config
prek run --all-files
uv run --no-sync --package=forecasters pytest
```

[Development and releases](docs/development.md) · [Architecture decisions](docs/adr/README.md)
