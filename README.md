# Forecasters

A uv workspace for the `forecasters` library and its local Line-up examples.

## Packages

| Path | Distribution | Purpose |
| --- | --- | --- |
| [`packages/forecasters`](packages/forecasters/README.md) | `forecasters` (public) | Thin, typed adapters over pretrained time-series foundation-model SDKs |
| `apps/lineup` | `forecasters-lineup` (private) | Shared Line-up engine, FastAPI server and static browser renderer; added in the next stacked change |
| `apps/marimo` | `forecasters-lineup-marimo` (private) | Local marimo notebook and AnyWidget host; added after the standalone application |

Only `forecasters` is published to PyPI. The applications are local development and
demonstration packages marked `Private :: Do Not Upload`.

## Development

This checkout uses Python 3.12. The library itself supports Python 3.11 and newer.

```bash
uv sync --group dev
uv run --package forecasters pytest
uv run --package forecasters mypy packages/forecasters/src packages/forecasters/tests/typing_consumer.py
```

The default development group includes Torch for fake-boundary tests. Install the native
SDKs for real-checkpoint tests with:

```bash
uv sync --group dev --group native
```

The `native` group installs `chronos-forecasting` and the Timestral SDK from its official
Git repository. It does not download model weights. Real-checkpoint parity tests remain
opt-in with `FORECASTERS_NATIVE=1`, and expect checkpoints in `.hf-cache/`.

The pinned pre-commit checks use `prek` and cover workflow security, dependency audit,
lockfile consistency and secret detection:

```bash
prek install --allow-missing-config
prek run --all-files
prek run --all-files --hook-stage manual
```

The manual stage also scans full Git history for secrets. In a linked worktree that shares
the main checkout's hooks directory, use the worktree-specific hook path or avoid installing
hooks there; `--allow-missing-config` keeps checkouts without this configuration working.

CI syncs the locked development environment once, then runs tests and mypy with
`uv run --no-sync`. Both wheel and sdist are installed into fresh NumPy-only environments
and tested; the installed package must retain `py.typed`. A separate archive check keeps
application code out of both distributions.

The library job uses `setup-uv` to cache dependencies and prune downloaded wheels before saving, following
[Astral's CI guidance](https://docs.astral.sh/uv/guides/integration/github/#caching).
The checks job caches prek's hook environments by configuration and tool version, without
a second uv cache for the small tool launcher.
Release builds use no restored cache and publish through a separate OIDC job.

## Build and release

Build only the public package into its dedicated artifact directory:

```bash
uv build --package forecasters --no-sources --out-dir dist/forecasters
```

The release workflow triggers only for `v*` tags. It verifies that the tagged commit is
reachable from `origin/main` and that the tag is `v` followed by the version in
`packages/forecasters/pyproject.toml`. It then builds only `forecasters` and publishes those
artifacts with PyPI trusted publishing. Before the first actual tag, configure the trusted
publisher on PyPI for project `forecasters` with owner `baggiponte`, repository
`forecasters` and workflow `release.yaml`. Pushing a branch or merging a pull request does
not publish a package.

## Architecture

- [Architecture decisions](docs/adr/README.md), including
  [ADR 0015](docs/adr/0015-uv-workspace-and-library-publishing.md) for this workspace.
- [Glossary](GLOSSARY.md).
- [`CONTRACT.md`](CONTRACT.md) is a historical design proposal, not the implemented contract.
