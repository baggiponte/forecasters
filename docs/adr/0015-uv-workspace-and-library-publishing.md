---
id: ADR-0015
status: accepted
date: "2026-10-20"
---

# Separate the library and local applications in a uv workspace

Use one uv workspace and lockfile for the library, the standalone Line-up application and its marimo host. The repository root coordinates development and contains no distributable Python package. Put the public library under `packages/forecasters`, the shared Line-up renderer and Python engine under `apps/lineup`, and the notebook and AnyWidget host under `apps/marimo`.

Keep the dependency direction explicit: both applications consume the library; the marimo host reuses Line-up's engine and static renderer. The shared Line-up package exposes the web-server dependencies as an optional extra so the notebook can use the shared renderer and engine without requiring FastAPI. Native model SDKs remain optional for library users. Workspace sources resolve sibling packages locally without modifying their public import names.

Publish only `forecasters`. Its wheel and source distribution contain the library and its own package documentation, with no application code, notebooks, assets or workspace paths in runtime metadata. Keep Git-sourced SDK installation in workspace configuration and document source installation for SDKs that are unavailable on PyPI; a PyPI distribution must not advertise an extra that cannot be installed from its declared dependencies. Internal application distributions are marked private and are never selected by the release workflow.

The release workflow builds `forecasters` explicitly into a dedicated artifact directory and publishes only those artifacts when a `v*` tag is pushed. Before publishing, require that the tagged commit is reachable from `origin/main` and that the tag equals `v` followed by the library package version. Use PyPI trusted publishing; configuring the matching publisher on PyPI is an account setup step. Opening or merging these changes does not itself release a package; publishing starts only from the matching tag on main. Do not add application publishing or cloud deployment in this milestone.

Introduce the workspace in three dependent pull requests: library and publishing, standalone demo, then marimo. Each branch builds on the preceding branch and remains independently reviewable. Preserve the existing working-demo branch while transplanting its reviewed behavior onto the repository's existing main history.

References: [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), [uv package builds](https://docs.astral.sh/uv/guides/package/) and [trusted publishing](https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi).
