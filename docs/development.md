# Development and releases

Use Python 3.12 and uv from the repository root. The library supports Python 3.11+.
The workspace contains the public library in `packages/forecasters`; the next stacked
changes add the private Line-up and marimo applications under `apps/`.

```bash
uv sync --locked
uv run --no-sync --package=forecasters pytest packages/forecasters/tests/unit
uv run --no-sync --package=forecasters mypy packages/forecasters/src packages/forecasters/tests/typing_consumer.py
uv check --locked --package=forecasters
uv run --isolated --locked --no-default-groups --group=pyrefly \
  python packages/forecasters/typecheck/pyrefly/run.py
prek run --all-files --hook-stage=manual
```

`uv check` runs ty with the version pinned in `uv.lock`. It checks the same scope as
mypy: the library plus the general consumer fixture in `tests/typing_consumer.py`. ty is
configured in `packages/forecasters/pyproject.toml` to target Python 3.11, the oldest
supported interpreter. mypy remains for the existing consumer-typing guarantees.

The Pyrefly tensor-shape check is experimental and runs in the isolated `pyrefly`
dependency group. The shape-aware NumPy and Torch stubs overlay the real libraries'
annotations, so that group must not be installed into the normal development environment;
`--isolated` keeps it separate from ty and mypy. `pyrefly-numpy-stubs` requires Python
3.12+, so this one check runs on 3.12 even though the library supports 3.11+. Its fixtures
prove that `predict` turns a one-dimensional context into `(horizon,)` or
`(horizon, quantiles)`, preserving the NumPy dtype and the NumPy/Torch backend. Separately,
Pyrefly checks the real `src/forecasters/_native.py` helper that both adapters use to
normalize Chronos `(1, Q, H)` and Timestral `(1, 1, H, Q)` before selecting quantiles. The
helper keeps its shape symbols in `TYPE_CHECKING`, so `shape_extensions` is a dev-only
checker dependency and is never imported at runtime. The exact quantile count and the Torch
element dtype remain gradual, and runtime validation is still authoritative for dynamic
dimensions and native SDK boundaries. See
`packages/forecasters/typecheck/pyrefly/README.md` for the full coverage matrix.

Prek runs Ruff linting and formatting, workflow security checks, dependency auditing,
lockfile validation and secret scanning. Its manual stage scans full Git history.
Use `prek install --allow-missing-config` to enable commit hooks; linked worktrees share
Git hooks, so configure a worktree-specific hook path if needed.

Tests are split by boundary:

- `tests/unit/`: input validation and adapters with fake native SDKs.
- `tests/integration/`: package imports, real-checkpoint parity and the distribution smoke script.
- `tests/typing_consumer.py`: static checks run by mypy and ty.
- `typecheck/pyrefly/`: checker-only shape stubs, the native-helper shape check, and the
  positive/negative Pyrefly fixtures.

Paths above are relative to `packages/forecasters/`. Running `pytest` without a path
collects both suites; native-checkpoint tests remain opt-in. CI shows unit and integration
tests as separate steps, runs mypy and `uv check`, and runs the smoke script against built
distributions. The Pyrefly shape check is a separate job because its environment must stay
isolated.

For real-checkpoint parity tests, install the optional native SDKs:

```bash
uv sync --locked --group=native
FORECASTERS_NATIVE=1 uv run --no-sync --package=forecasters pytest packages/forecasters/tests/integration
```

Tests expect downloaded checkpoints in `.hf-cache/`; installing SDKs does not download weights.
The default development group already includes Torch for fake-boundary tests.

CI syncs once, runs tests, mypy, ty and then smoke-tests both distributions in isolated
SDK-free environments. A separate job runs the Pyrefly shape fixtures in an isolated
environment. It caches uv dependencies and prek environments.

## Publishing

```bash
uv build --package=forecasters --no-sources --out-dir=dist/forecasters
```

Only `forecasters` is published. A pushed `v<package-version>` tag triggers the release
workflow, which requires the tagged commit to be reachable from `main`. An uncached build
job passes its distributions to a separate OIDC job using `uv publish --trusted-publishing=always`.

Before the first release, configure PyPI trusted publishing for project `forecasters`,
owner `baggiponte`, repository `forecasters` and workflow `release.yaml`. Merging a PR
does not publish anything. Add a license before the first public release.

References: [Astral's CI and publishing guide](https://docs.astral.sh/uv/guides/integration/github/),
[workspace architecture](adr/0015-uv-workspace-and-library-publishing.md).
