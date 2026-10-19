# Review: `forecasters` CONTRACT.md 0.2

## Verdict

0.2 is a big improvement. Most of the first-round contradictions are resolved, and the API surface is close to sign-off.

- **Public API:** go, after the doc fixes listed under "Smaller fixes".
- **Worker protocol and adapters:** no-go until the eight items in §4 are fixed. Most are one-paragraph changes. Items 1 to 3 would force redesign if found during implementation.

---

## 1. Are the prior contradictions resolved?

| Area | Status | What's left |
|---|---|---|
| Native-only numbers (C1, C2) | ✅ Resolved | There is no rule that `levels` must be strictly increasing and unique. A duplicate level makes `quantile()` ambiguous. The worker should check this; it is not number repair. |
| Point semantics (C3, C4) | ✅ Mostly | `Capabilities.point` is a class attribute, but the real value can depend on the checkpoint. See §2.1. |
| Error taxonomy | ⚠️ Partial | §6 says 2-D input raises "`UnsupportedCapability`/`InvalidInput`". Pick one; `InvalidInput` fits better. `WorkerError.phase` only allows `{"load", "forecast"}`. That leaves no place for malformed replies, an `id` mismatch, a handshake mismatch, or a crash between calls. Add `"protocol"` (or `"handshake"` plus `"crash"`). |
| Protocol / cleanup (C8, C9, C11) | ⚠️ Partial | Terminal states are defined, but the wire format for load failures is missing, and so is stderr handling. See §4. |
| Adapter registration (C7, Q7) | ✅ Mostly | Strings and entry points are right. Two gaps: the adapter must be importable in the **parent** too (it reads the class attributes there), and there is no rule for two entry points claiming the same name. Also, the custom-adapter example doesn't subclass `Adapter` and has no `ADAPTER_API`. Say whether `Adapter` is an ABC to subclass or a `Protocol` to match. |
| `meta` default (C13) | ✅ | `meta` has no default now. Say whether `MappingProxyType` is used. |

## 2. Remaining contradictions and implementation traps

### 2.1 Capabilities are static, but the handshake compares them strictly (critical)

`load(model=...)` lets you pick a different checkpoint, for example a different TimesFM version or a different Chronos-Bolt size. Those can differ in `max_context`, `max_horizon`, or even `point`. As written, the worker reports the real values, the parent compares them with the class attribute, and the result is a `WorkerError`. So the override can't be used whenever the checkpoint's limits differ from the default. §2.4 also says capabilities are "derived from the wrapped model", which contradicts "class attributes, readable without loading".

**Fix:** treat the class attribute as a description of the default model, for use by `available()`. The handshake value is authoritative. `Forecaster.capabilities` takes the handshake value after the worker starts. Only reject the handshake if it violates hard invariants, such as `quantiles` changing type or the adapter name not matching.

### 2.2 `fit` ordering is circular

`fit` needs `max_context` and `min_context` to truncate and validate, but the authoritative values only arrive at the handshake. "A failed `fit` leaves the previous context intact" also doesn't make sense if the first fit's load fails, because that failure is terminal.

**Fix:** spell out the order:

1. Validate shape, dtype and finiteness locally.
2. Start the worker if needed, bounded by `load_timeout`.
3. Apply `min_context` and truncation using the handshake capabilities.
4. Commit the new context.

A load failure is terminal. A validation failure keeps the previous context.

### 2.3 Which failures are terminal?

The contract says a crash or timeout is terminal. It doesn't say what happens after an adapter exception (`WorkerError(phase="forecast")`) or after `MissingDependency`.

**Fix:** add a short table. Adapter exceptions should be non-terminal. Load-time failures, crashes, timeouts and protocol errors should be terminal. Also state that threads waiting on the lock get `ClosedForecaster` once another call has closed the object.

### 2.4 The wire format can't express load failures

The only error example is a reply to request `id: 7`. `MissingDependency` and other load errors happen **before** `ready`, so there is no request to reply to. Separately, `"kind": null` on success responses is noise, and the set of `kind` values is never listed.

**Fix:** add `{"v":1,"op":"error","phase":"load","kind":...,"error":...,"traceback":...}` as the pre-ready failure message. List the `kind` values and map each one to an exception class. Require the parent to check that each response's `id` matches its request.

### 2.5 Handshake fields are incomplete

§7 says the worker reports `ADAPTER_API`, but the example handshake doesn't include it. Because the worker venv gets `forecasters` through a separate `--no-deps` install, the parent and worker library versions **will** drift. Yet §7 assumes they "come from the same installed version".

**Fix:** add `adapter_api` and `forecasters_version` to `ready`. Define the policy: require an exact match, or at least the same `v` plus a warning.

### 2.6 Adapter options are passed on the command line

Two problems with `--options <json>`:

- Arguments are visible in `ps`, so a token or private path passed as an option leaks.
- `**adapter_kwargs` must be JSON-serialisable, so `dtype=torch.bfloat16` can't work.

**Fix:** send the options in an `init` message on stdin. State that `adapter_kwargs` must be JSON-serialisable (strings for dtypes).

### 2.7 The stderr pipe can deadlock

stdout is redirected to stderr, so tqdm and HF logging fill stderr. If the parent uses `stderr=PIPE` and doesn't read it, the worker blocks once the pipe buffer (about 64 KB) fills. Model downloads hit this quickly.

**Fix:** either inherit the parent's stderr, or drain it with a reader thread into a bounded ring buffer and attach the tail to `WorkerError`. Pick one.

### 2.8 Smaller traps

- **Platform:** fd duplication and SIGTERM/SIGKILL escalation are POSIX-specific. State "POSIX only in 0.1" or define the Windows behaviour.
- **Timeout implementation:** `timeout` requires a selector or reader thread on the protocol fd. Blocking `readline()` can't be bounded. Not a contract change, but note it.
- **Worker numpy:** the worker venv's numpy comes from the SDK. State the minimum numpy version the worker code supports.
- **`meta` keys:** say which keys are required and which are optional (`revision` may be unknown). Say where `ms` comes from (per response).
- **Leftover duplicate decisions:** "Open decisions" repeats items already decided in the body (timeouts, `meta`, point-only representation, `ADAPTER_API`). It also leaves the `predict_quantiles` alias open while §10.2 says "no alias". Delete or resolve them.
- **Doc nits:** the intro says overrides are "recorded in §9", but they are in §10. §8 cites `forecasters.edge-cases.md`, which should be `design/edge-cases.md`.

## 3. Risk of the two product overrides

### `fit` / `predict` / `predict_proba` surface: low to medium technical risk

**Cost of the stateful design.** `fit` followed by `predict` is two locked calls, not one atomic call. If two threads share a `Forecaster`, thread A can fit `y1`, thread B can fit `y2`, and A then predicts on `y2`. The per-call lock doesn't prevent this. The contract only says "use separate objects for parallelism"; it should also say "one object, one thread". The alternative is an internal atomic path, but that is the stateless `forecast()` the product rejected.

**Sklearn duck typing.** Some sklearn tooling checks `hasattr(est, "predict_proba")` and then calls `predict_proba(X)`. Examples are soft voting, calibration, scorers and `_check_response_method`. This is safe **only if `h` validation is strict**. Reject ndarrays and sequences explicitly, with an error message that mentions sklearn incompatibility. The contract already rejects `bool`; add arrays, and test it.

**Expectation gap.** In sklearn, `fit` is often expected to be either slow and training or fast. Here the first `fit` may download weights for minutes. That is already documented, so the risk is acceptable.

### `predict_proba` name: medium risk

- **Name clash in the forecasting ecosystem.** In sktime and skpro, `predict_proba` returns a **distribution object**, and quantiles come from `predict_quantiles`. Users moving between libraries will expect `.pdf`, `.cdf` or sampling. Code that does `np.asarray(f.predict_proba(h))` gets an object array instead of an error.
  **Mitigation:** don't add `__array__` to `Forecast`. Fail loudly instead.
- **Renaming later is breaking.** "Rename or alias before 1.0" is a semver break if the name has already shipped.
  **Mitigation:** decide now. The cheapest safe hedge is to also ship `predict_quantiles` as the canonical name, with `predict_proba` as a documented alias, sharing one implementation and one test. The other option is to commit to `predict_proba` permanently. Leaving it open is the worst choice.

Neither override blocks implementation.

## 4. Must fix before implementing the worker protocol and adapters

1. Make capabilities authoritative from the handshake, per checkpoint (§2.1).
2. Specify the `fit` order and which failures are terminal (§2.2, §2.3).
3. Pass options in an `init` message on stdin, not on the command line, and require JSON-serialisable kwargs (§2.6).
4. Add the pre-ready error message, the list of `kind` values and their exception mapping, `id` checking, and the extra `phase` values (§2.4, §1).
5. Add `adapter_api` and `forecasters_version` to the handshake, with a version policy (§2.5).
6. Choose a stderr policy: inherit, or drain into a ring buffer (§2.7).
7. Validate `levels` as strictly increasing and unique. Pick one exception for 2-D input (§1).
8. Say whether `Adapter` is an ABC or a `Protocol`, and fix the custom-adapter example to match. Require the adapter to be importable in both the parent and the worker (§1).

**Recommended later, not blocking:** reject ndarray `h` with a sklearn-specific message, decide on `predict_proba` vs the alias now, and state POSIX-only support.

## Recommendation

**Conditional go.** Freeze the public Python API once the doc fixes in §2.8 are in. Start implementing the worker and adapters after the eight items in §4 are in the contract. They are small edits, but items 1 to 3 would force a redesign if they were only discovered in code.
