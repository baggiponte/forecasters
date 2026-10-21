---
id: ADR-0004
status: accepted
date: "2026-10-20"
---

# Start with two small native-quantile models

## Context

A common interface needs validation against different SDKs without making model downloads and inference prohibitively expensive for development.

## Decision

Start with Chronos-Bolt Tiny and Timestral 5M. Both support CPU inference and native medians, but return different quantile layouts.

## Consequences

Native parity tests exercise real representation differences. This pair establishes the initial adapter boundary; it is not a performance recommendation or a promise to support a model catalog.
