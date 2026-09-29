# Testing

How the Apex test suite works, and what it is actually protecting.

## Running it

```bash
pip install -r requirements-dev.txt
pytest                 # 293 cases
pytest -q tests/test_production_regressions.py   # one file
pytest -k jev                                   # by name
```

Python 3.12 is required — `pandas-ta` sets that floor. The suite runs in about
40 seconds locally once the RL stack is installed, and about 3 minutes on CI.

`pytest.ini` promotes `PytestUnhandledThreadExceptionWarning` to an error. An
exception escaping a background thread is exactly how the agent loop died
without telling anyone, so it must not be allowed to pass quietly.

## What is in it

208 test functions, 293 collected cases across 20 files. The split is roughly
half "does the code do the right thing" and half "has someone reintroduced a
defect we already found".

| File | Cases | What it protects |
|---|---:|---|
| `test_jev_gates.py` | 37 | JEV gate maths, thresholds, scaling |
| `test_input_and_auth_hardening.py` | 28 | Auth, rate limiting, value validation, security headers |
| `test_supply_chain.py` | 19 | Dependency pins and their constraints |
| `test_exit_policies.py` | 16 | EOD / trailing / SL-TP, and halt-path parity |
| `test_production_regressions.py` | 10 | Production import order, JEV liveness, config flags |
| `test_jev_state_v2.py` | 9 | JEV market-state builder |
| `test_model_registry.py` | 9 | Weight-swap locking, with plain fakes |
| `test_online_learner_race.py` | 9 | Structural proof of the copy-and-swap |
| `test_universe_single_source.py` | 5 | The trading universe is defined once |
| `test_route_inventory.py` | 5 | All 21 routes exist; mutating routes reject GET |
| `test_lock_io_separation.py` | 5 | No blocking I/O under the state lock |
| `test_silent_failures.py` | 6 | No broad `except` that swallows |
| `test_market_hours.py` | 6 | DST-correct exchange calendars |
| `test_agent_loop_resilience.py` | 5 | A failed cycle does not kill trading |
| `test_feature_contract.py` | 5 | Scaler / policy / column file agree on 29 features |
| `test_log_rotation.py` | 3 | The log file is bounded |
| `test_online_learner_runtime.py` | 8 | Copy-and-swap against a torch-shaped fake |
| `test_online_learner_real_ppo.py` | 6 | Copy-and-swap against real PPO and torch |
| `test_template_extraction.py` | 9 | The extracted SPA still serves correctly |
| `test_shadow_calibration.py` | 8 | Regime-shadow calibration |

## Two kinds of test, and why both exist

**Behavioural tests** assert what the system does. Most of the suite is this.

**Structural tests** parse the source with `ast` and assert a *shape*: that no
`with _lock:` block contains a blocking call, that no `except Exception`
handler body is a bare `pass`, that `optimizer.step()` is never applied to the
live model, that all 21 routes still exist.

These exist because the defects in this project were not the kind a behavioural
test catches. A missing route returned 404 — the tests that used it noticed, but
only because they happened to run. A broad `except: pass` changes no behaviour
until the thing it hides stops happening. Structural tests turn "someone
carelessly re-added this" into a red build.

## Tests that are validated by breaking the fix

A concurrency test that cannot fail proves nothing. `test_concurrent_inference_never_sees_a_partial_swap`
in `test_online_learner_real_ppo.py` was checked by removing the lock from
`ModelRegistry.publish()` and confirming the test fails. Do this for any new
concurrency test before trusting it.

The same applies to the parity tests in `test_exit_policies.py` — they were
written to fail while the halt path and the normal path still diverged.

## What the suite does not cover

Being explicit about this, because an unstated gap reads as coverage:

- **Supabase.** Every persistence test runs against the JSON fallback. The
  Supabase path is exercised only in production. This is why the
  lock/I-O defect could survive: it only manifests against a real network call.
- **Live market data.** `yfinance` and the NSE API are never called in tests.
  Price handling is tested with fixed inputs.
- **The JEV entry side under halt.** Exit-side parity is proven; entry-side
  parity is not.
- **torch under the non-RL path.** `apex_dashboard` imports torch lazily and
  guards it, so the serving path is tested without it.

## CI

`.github/workflows/ci.yml`, three jobs:

1. **tests** — install the pinned stack, byte-compile, run the suite, then
   assert `_JEV_AVAILABLE` under the production import order. The last step is
   the important one: the circular-import defect kept the suite green for an
   entire development cycle because the tests imported `apex_jev` first.
2. **dependency audit** — `pip-audit`, failing on any fixable advisory.
3. **image builds** — build the Dockerfile, then assert the image contains no
   `.env`, `apex.log` or `apex_dual_state.json`.

CI is currently **advisory**; branch protection is not configured. Every guard
in this suite can be bypassed by pushing straight to the deploy branch until
that is turned on.

## Adding a test for a bug

The convention used throughout this codebase:

1. Write the test. Run it. **Confirm it fails for the right reason** — a test
   that fails on a typo teaches you nothing.
2. Fix the defect.
3. Run the suite. Confirm nothing else moved.
4. If the defect was a *shape* problem (a deleted route, a reintroduced
   `except: pass`, a loosened pin), add a structural guard so the shape is
   asserted directly rather than inferred from behaviour.
