# 🔮 Forecasters

A small interface for pretrained time-series models, inspired by 🤗 Transformers. Supports Chronos-Bolt and Timestral through their native SDKs.

```bash
uv pip install 'forecasters[chronos]'
# For Timestral, also install its official SDK:
uv pip install 'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'
```

```python
import numpy as np
from forecasters import ChronosBolt

model = ChronosBolt.from_pretrained()
context = np.sin(np.arange(64, dtype=np.float32) / 5)

median = model.predict(context, horizon=8)  # (8,)
quantiles = model.predict(context, horizon=8, quantile_levels=[0.1, 0.5, 0.9])  # (8, 3)
```

Pass a nonempty, finite, one-dimensional floating array. NumPy in, NumPy out; Torch in, Torch out, preserving dtype and device. Native compute device is set separately with `from_pretrained(device=...)`. Quantiles come from the loaded checkpoint's grid; unsupported levels, contexts and horizons raise instead of being interpolated or truncated. `import forecasters` works without Torch or model SDKs installed.

## Development

```bash
uv sync --locked
prek install --allow-missing-config
prek run --all-files
uv run --no-sync --package=forecasters pytest
```

Tests live in `packages/forecasters/tests/unit` and `integration`. Native integration tests need downloaded checkpoints in `.hf-cache/`:

```bash
uv sync --locked --group=native
FORECASTERS_NATIVE=1 uv run --no-sync --package=forecasters pytest packages/forecasters/tests/integration
```

## Releases

The release workflow publishes only the library on a `v<package-version>` tag reachable from `main`. Before the first release, add a license and configure the PyPI trusted publisher for `baggiponte/forecasters`, workflow `release.yaml`.

[Architecture decisions](docs/adr/) record the reasons behind the public boundaries.
