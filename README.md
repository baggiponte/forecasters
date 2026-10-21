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

## Development

Run `just setup` to install development dependencies and hooks; `just --list` lists the available commands.

[Line-up drawing demo](apps/lineup/README.md).
