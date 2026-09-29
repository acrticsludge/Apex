# ADR-001: Break the JEV/dashboard import cycle with dependency injection

- Status: Accepted
- Date: 2026-09-29
- Supersedes: nothing

## Context

`apex_dashboard` and `apex_jev` need to share configuration: which JEV features
are enabled. The first implementation had `apex_jev` import it:

```python
# apex_jev.py
from apex_dashboard import cfg
```

Production starts with `gunicorn apex_dashboard:app`, so the dashboard is
imported first. At the point it reaches `import apex_jev`, `apex_dashboard` is
only partially initialised — `cfg` is not assigned until 30 lines later. The
`from ... import cfg` therefore raises:

```
ImportError: cannot import name 'cfg' from partially initialized module
```

That exception was caught by a bare `except Exception` at the import site and
discarded, leaving `_JEV_AVAILABLE = False`.

**Consequence: every JEV risk gate was disabled in production for the entire
life of the feature.** The test suite stayed green because
`tests/test_jev_gates.py` imports `apex_jev` first — the opposite order, which
works. Nothing in the build, the logs, or the dashboard indicated a problem.

The same cycle existed in the other direction: `trading_agent/integration/rl_signal.py`
does `from apex_dashboard import _signals` inside a function. That happened to
work only because the import was function-local, but it was the same latent
hazard.

## Decision

**A module must never import another module's partially initialised globals.
Configuration flows one way, by explicit call.**

1. `apex_jev` imports nothing from `apex_dashboard`. It owns its own defaults
   sourced from env, plus `configure(flags: dict)` which the dashboard calls
   once after `cfg` exists and again whenever `/api/config` changes a `jev_*`
   key.
2. The guarded import at the dashboard's import site now logs at ERROR when it
   fails and records the reason in `_JEV_IMPORT_ERROR`, which `/api/jev/status`
   surfaces. A risk layer that vanishes is worse than one that is absent.
3. CI asserts `_JEV_AVAILABLE is True` under the production import order, so
   the defect cannot return unnoticed.

## Alternatives considered

**Import `apex_jev` after `cfg` is defined (reorder only).** Rejected: it
leaves the cycle in place, and the next person who moves a line above `cfg`
silently disables a risk layer again. It treats the symptom.

**Make `apex_jev` mandatory — let the import error propagate.** Rejected for
now: a transient TypeSafe outage at boot would prevent the dashboard from
starting at all, taking the read-only views with it. The trade-off is
deliberate: the app starts, the JEV gates are off, and the operator is told
loudly. A future ADR should revisit making it hard-fail once the feature is
load-bearing.

**Merge the two modules.** Rejected: `apex_jev` is 621 lines of a cohesive
TypeSafe client and is independently testable. Merging would push
`apex_dashboard` past 3,000 lines and make neither unit testable alone.

## Consequences

- A cycle is now structurally impossible between these two modules, not merely
  absent.
- `apex_jev` is testable without the Flask app; `tests/test_jev_gates.py` no
  longer needs to reason about import order.
- Adding a JEV flag means adding it to `cfg` *and* to `apex_jev._FLAG_ENV`, or
  it silently defaults to on. A test asserts the six flags exist in `cfg`.
- The CI import-order gate is a real, non-obvious check. It exists because this
  failure was invisible for an entire development cycle.
