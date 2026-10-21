---
id: ADR-0010
status: accepted
date: "2026-10-20"
---

# Publish a standalone library distribution

## Context

Library users need inference dependencies only. Development can require Git-sourced SDKs such as Timestral, which cannot be advertised as a normal PyPI extra.

## Decision

Keep the public distribution in `packages/forecasters` and development sources in the workspace configuration. Build the library independently with `uv_build` and publish through PyPI trusted publishing.

## Consequences

Published metadata contains installable dependencies without workspace paths or Git-only extras. Release tags must match the library version and point to commits reachable from `main`.
