---
id: ADR-0000
status: accepted
date: "2026-10-21"
---

# Record core-library and contribution decisions

## Context

Future maintainers need the reasons behind lasting library and contribution choices, without treating demos or implementation details as architectural commitments.

## Decision

Keep ADRs for significant decisions about the core library's interfaces, dependencies and distribution, plus repository-wide contribution conventions. Use a short title, status and date, followed by Context, Decision and Consequences, following [Michael Nygard's format](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions). Write concise paragraphs explaining the trade-off rather than restating code.

## Consequences

Number records consecutively from `0000`; preserve superseded core decisions with a link to their replacement. Demo choices and routine implementation details belong in their code or usage instructions.
