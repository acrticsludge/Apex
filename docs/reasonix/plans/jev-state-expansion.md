# Implementation Plan: JEV State Expansion v2 — IMPLEMENTED 2026-09-28

## Overview
Implement `docs/reasonix/specs/jev-state-expansion.md`: real market context,
current-portfolio maths, and per-position context for `position_action`, within
the 10-calls/cycle budget. Additive only — old keys keep working.

(NOTE: `tasks/plan.md` + `tasks/todo.md` still hold unchecked items from the
original JEV rollout, so this plan lives here instead of overwriting them.)

## Architecture Decisions
- Pure builder functions in `apex_jev.py`, no I/O except cached fetchers.
- Callers in `apex_dashboard.py` stay thin: build ctx → call → log.
- Budget enforced at call site, not inside client.
- `position_action` resolved from per-symbol answers, not system answers.

## Task List

### Phase 1: Foundation (builders + tests)
- [ ] Task 1: Indicator extensions — `bb_pos`, `ema_gap_pct`, `atr_pct` in state builder + `tests/test_jev_state_v2.py`
  - Acceptance: bb_pos in [0,1], gap sign correct, atr_pct > 0, old keys untouched
  - Verify: `py -3.13 -m pytest tests/test_jev_state_v2.py -v`
  - Files: `apex_jev.py`, `tests/test_jev_state_v2.py`
  - Scope: S (1-2 files)
- [ ] Task 2: Real VIX + breadth fetchers with 5-min cache + fallback (20.0/0.5)
  - Acceptance: US→`^VIX`, NSE→`^INDIAVIX`; breadth = watchlist % above day-open; failure → fallback, never raises
  - Verify: pytest + manual `fetch_vix("us")` / `fetch_vix("india")`
  - Files: `apex_jev.py`, `tests/test_jev_state_v2.py`
  - Scope: S

### Checkpoint: Foundation
- [ ] `py_compile` clean, new tests pass, old `test_jev_gates.py` still passes

### Phase 2: Wiring (callers)
- [ ] Task 3: Wire builders into `fetch_cycle_data` + `agent_loop` (current DD, exposure, cash_pct, session WR, enriched news)
  - Acceptance: no `vix: 20.0` literals at call sites; payload contains new keys
  - Verify: manual Decision Log shows JEV entries with new context; existing tests pass
  - Files: `apex_dashboard.py`
  - Scope: M
- [ ] Task 4: Call budgeting — per-symbol only for held + top-3 by confidence, cap 10/cycle, per-symbol 300s cache
  - Acceptance: 32-symbol cycle issues ≤10 JEV calls; cache hit on repeat minute-bar
  - Verify: counter test + live log count per cycle
  - Files: `apex_dashboard.py`, `apex_jev.py`
  - Scope: M

### Checkpoint: Wiring
- [ ] Full cycle runs, JEV filter shows entries, call count verified in logs

### Phase 3: Position actions live
- [ ] Task 5: Route per-symbol `position_action` into `apply_cycle` exit/trim path (replaces system-level lookup that always returns hold/0)
  - Acceptance: held symbol with JEV exit + conf ≥ threshold → SELL/COVER logged as JEV EXIT; trim halves position
  - Verify: `pytest tests/test_jev_gates.py -v` + manual paper-trade check
  - Files: `apex_dashboard.py`
  - Scope: M (high-risk — early review)
- [ ] Task 6: Per-symbol `think_log("JEV", ...)` for news + action choices
  - Acceptance: every per-symbol JEV answer appears under JEV filter
  - Verify: manual dashboard check
  - Files: `apex_dashboard.py`
  - Scope: S

### Checkpoint: Complete
- [ ] All spec success criteria met, human reviews Decision Log output before merge

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| `^INDIAVIX` unreliable on yfinance | Med | fallback 20.0, log once at debug |
| Per-symbol calls blow budget | High | cap + cache (Task 4 before Task 5) |
| JEV EXIT over-trading | High | keep confidence thresholds, shadow-verify one session first |
| yfinance news shape change | Low | `.get()` access, try/except per symbol |

## Open Questions
- Shadow-run Task 5 for one session before live exits? (Recommended: yes)
- Include RL `(action, conf)` in JEV state now or later? (Spec: later)

## Parallelization
- Tasks 1 + 2 parallelizable (different functions, shared test file needs coordination)
- Tasks 3→4→5→6 strictly sequential (same call path)
