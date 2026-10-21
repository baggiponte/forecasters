---
id: ADR-0012
status: accepted
date: "2026-10-20"
---

# Keep tests that name an observable failure

Keep a test when it names a plausible product regression, a domain rule, a real external integration failure or a concurrency/integrity bug. Prefer observable public behavior, parity with the native SDK, model independence and reuse, meaningful dtype/device/quantile boundaries, transport parity and actual concurrency failures. A gated real-checkpoint test earns its place because it checks the external SDK integration that fakes cannot establish.

Remove tests that only repeat a private helper already covered through public behavior, copy the implementation into an assertion, or freeze a reversible cosmetic detail. Temporary browser or manual checks are valid evidence for cosmetic changes and need not become committed tests. Consequential security, data-integrity and protocol boundaries remain tested even when their failure is rare.

No coverage percentage, test-count target, rule that every change requires a test, or blanket ban on tests. Add or strengthen a test for a plausible regression introduced or exposed by the change; when uncertain, make one temporary mutation and keep it only if the test fails for the right reason.

Review heuristic: be able to say what failure a changed test catches, or what surviving coverage makes a removed test safe. This is a judgment check, not a paperwork requirement or a rule that every change needs a test.
