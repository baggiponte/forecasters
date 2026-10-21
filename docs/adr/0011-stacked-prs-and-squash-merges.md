---
id: ADR-0011
status: accepted
date: "2026-10-21"
---

# Use stacked PRs and squash merges

## Context

Dependent changes need small, reviewable PRs while keeping the project's history linear.

## Decision

Prefer native GitHub stacks through `gh stack` for dependent PRs. Update branches with rebase and merge each PR with squash, using its title as the commit title. Delete merged head branches automatically.

## Consequences

Each PR lands as one commit. Use `gh stack rebase` and `gh stack push` for stack updates, `gh pr update-branch --rebase` for standalone PRs, and `gh stack sync` after merging a stack layer.
