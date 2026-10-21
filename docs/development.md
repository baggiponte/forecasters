# Development and releases

Use Python 3.12 and uv from the repository root. The library supports Python 3.11+.
The workspace contains the public library in `packages/forecasters`; the next stacked
changes add the private Line-up and marimo applications under `apps/`.

```bash
uv sync --locked
uv run --no-sync --package=forecasters pytest packages/forecasters/tests/unit
uv check --locked --package=forecasters
uv run --isolated --locked --no-default-groups --group=pyrefly \
  python packages/forecasters/typecheck/pyrefly/run.py
prek run --all-files --hook-stage=manual
```

Prek runs Ruff linting and formatting, workflow security checks, dependency auditing,
lockfile validation and secret scanning. Its manual stage scans full Git history.
Use `prek install --allow-missing-config` to enable commit hooks; linked worktrees share
Git hooks, so configure a worktree-specific hook path if needed.

Tests are split by boundary:

- `tests/unit/`: input validation and adapters with fake native SDKs.
- `tests/integration/`: package imports, real-checkpoint parity and the distribution smoke script.
- `tests/typing_consumer.py`: static checks run by ty.
- `typecheck/pyrefly/`: static shape checks run by Pyrefly.

Paths above are relative to `packages/forecasters/`. Running `pytest` without a path
collects both suites; native-checkpoint tests remain opt-in. CI shows unit and integration
tests as separate steps, runs `uv check`, and runs the smoke script against built
distributions.

For real-checkpoint parity tests, install the optional native SDKs:

```bash
uv sync --locked --group=native
FORECASTERS_NATIVE=1 uv run --no-sync --package=forecasters pytest packages/forecasters/tests/integration
```

Tests expect downloaded checkpoints in `.hf-cache/`; installing SDKs does not download weights.
The default development group already includes Torch for fake-boundary tests.

CI syncs once, runs tests and ty, then smoke-tests both distributions in isolated
SDK-free environments. It caches uv dependencies and prek environments.

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
