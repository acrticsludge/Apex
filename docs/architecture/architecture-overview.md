# Apex Architecture Overview

Status: current as of the `feat/jev-integration` hardening work (branch tip `200e6c0c`).

This is the map of what the system *is now*. For why it is this way, see
`docs/decisions/`. For the defects that shaped it, see
`docs/audits/jev-integration-audit.md` and its resolution table.

---

## 1. What Apex is

A paper-trading agent for the Indian (NSE) and US (NYSE) markets, with a
Flask dashboard as its only control surface. It scans a fixed universe each
cycle, scores symbols with either a trained PPO policy or rule-based logic,
overlays a TypeSafe "JEV" judgment layer for regime and risk gating, and
executes simulated orders against a Supabase-persisted ledger.

**No live order placement exists.** Every fill goes through `paper_buy` /
`paper_sell` / `paper_short` / `paper_cover` into an in-memory ledger. This
matters for risk assessment: the worst realistic failure is a bad paper
portfolio, not a bad bank account.

---

## 2. Process model

Two long-lived threads plus the request pool.

```
  gunicorn (--workers 1 --threads 8, timeout 120s)
     │
     ├── _price_updater thread ......... every 5s
     │     ├── _fetch_nse_price  (curl_cffi, TLS fingerprint for NSE)
     │     └── _fetch_us_price   (yfinance fast_info)
     │     writes into _latest_prices + _price_ts, under _price_lock (RLock)
     │
     ├── agent_loop thread ............. every check_interval_min
     │     └── _run_one_cycle()
     │           ├── price snapshot (ONCE per cycle — Thread 2 rule)
     │           ├── session rotation
     │           ├── JEV system decisions per open market
     │           ├── apply_jev_cycle_gates()  → _jev_gates (per-cycle overrides)
     │           ├── publish_jev_decisions() → _signals["jev_decisions"]
     │           ├── apply_cycle() per market
     │           │     ├── halt gate / EOD / trailing / SL-TP   (exit policies)
     │           │     ├── loss + drawdown kill-switch
     │           │     └── entry filters → paper_buy / paper_short
     │           ├── snapshot_state()  (under _lock)
     │           └── persist_state()   (OUTSIDE _lock)
     │
     └── request threads (8) ........... /api/* + SSE price stream
           every mutating route: validate → mutate under _lock → snapshot
                                 → persist outside _lock
```

**The price-snapshot rule.** The price updater writes; the agent snapshots once
at the top of a cycle; every `paper_*` reads only from that snapshot. A price
cannot change mid-decision.

**The lock discipline.** `_price_lock` (re-entrant) guards the price store.
`_lock` guards the trading ledger. **No blocking I/O ever runs under
`_lock`** — enforced by an AST test in `tests/test_lock_io_separation.py`.

---

## 3. Module layout

| Module | Lines | Owns |
|---|---:|---|
| `apex_dashboard.py` | 2,587 | Flask app, routes, the agent cycle, trade execution, exit policies, config |
| `apex_jev.py` | 621 | TypeSafe client, frozen question schema, circuit breaker, risk-gate math |
| `apex_market.py` | 59 | Exchange calendars, time zones (leaf, no project imports) |
| `apex_config.py` | 102 | Bounds/validation for every client-writable value (leaf) |
| `apex_universe.py` | 22 | The canonical 32-symbol universe (leaf, shared with the RL package) |
| `trading_agent/config.py` | 229 | RL dataset/training settings from env |
| `trading_agent/integration/rl_signal.py` | 290 | Observation build, PPO inference, regime gating |
| `trading_agent/integration/online_learner.py` | 229 | PPO fine-tuning on closed trades |
| `trading_agent/integration/model_registry.py` | 118 | Atomic weight swap between those two (leaf, torch-free) |
| `trading_agent/data/*` | ~1,000 | Fetchers, indicators, sentiment, VWAP, volume profile, sweeps |

The three leaf modules exist so that `apex_dashboard`, `apex_jev` and
`trading_agent` can share logic without importing each other. That constraint
is what makes a cycle impossible rather than merely unlikely.

### The import-order hazard

`apex_dashboard` imports `apex_jev`; `trading_agent` imports `apex_dashboard`
lazily for the signal cache. Production starts with
`gunicorn apex_dashboard:app`, so **the dashboard is imported first**. Any
import in `apex_jev` that reaches back into `apex_dashboard` for a
module-level name fails under that order.

This is not theoretical. It happened: `apex_jev` imported `cfg` from the
dashboard, the dashboard had not yet defined `cfg` at that point in its own
import, the resulting `ImportError` was swallowed by a bare
`except Exception`, and **every JEV risk gate was silently disabled in
production for the entire life of the feature** while the test suite stayed
green because the tests imported `apex_jev` first.

Two defences now exist: `apex_jev` takes its flags via `configure(cfg)` and
imports nothing from the dashboard, and CI asserts `_JEV_AVAILABLE` is true
under the production import order.

---

## 4. Data flow for one cycle

```
  yfinance / NSE API
        │
        ▼
  fetch_cycle_data()  ──────────►  analyse()  ──►  per-symbol scores
        │                              │
        │                              ├── JEV symbol decisions (cached 5 min)
        │                              └── RL observation → get_rl_signal()
        │
        ▼
  apply_cycle(market, analyses, prices, max_pos, jev_decisions)
        │
        ├── EXIT SIDE  (always, in order)
        │     eod_exit_pass   → harvest winners, force-exit the rest
        │     stop_pass       → ratchet trailing, then SL/target
        │     hold_review_pass→ RL votes an early exit on survivors
        │
        ├── RISK GATES
        │     daily loss % vs daily_loss_limit_pct
        │     drawdown  % vs max_drawdown_pct
        │     (both from effective_cfg(market) — base + this cycle's JEV gates)
        │
        └── ENTRY SIDE
              index trend · ADX · open-window · confidence · position cap
              → execute_buy / execute_short
                    │
                    ├── rcfg = effective_cfg(market)  ← JEV gates applied here
                    ├── rl_feedback_entry()  (never raises; warns on failure)
                    └── mutate mstate
        │
        ▼
  snapshot_state() ──► persist_state()   [lock released in between]
```

**Why `effective_cfg` exists.** `apply_jev_gates()` is a pure function that
returns a modified copy of the config. Its return value was originally
discarded at the call site, so no gate was ever applied. Gates now land in a
per-cycle `_jev_gates` dict that `effective_cfg(market)` overlays on the base
`cfg`. The base is never mutated — writing into it would compound the
multipliers every cycle until `risk_per_trade` decayed to zero.

---

## 5. The JEV overlay

Six frozen questions, all evaluated in parallel per TypeSafe System One:

| Question | Type | Consumed by |
|---|---|---|
| `regime` | choice (bullish/bearish/choppy/crisis) | `get_regime_scaling` |
| `trend_strength` | score 0–3 | `get_trailing_dist_mult` |
| `news_bullishness` | score | sentiment overlay |
| `portfolio_stress` | score | `get_stress_scaling` |
| `halt_new_buys` | noul 0–1 | `should_halt` — blocks new entries |
| `position_action` | choice (buy/add/hold/trim/exit) | per-symbol entry/exit gate |

Three failure modes are handled explicitly:

- **Circuit breaker.** Three consecutive failures opens it; the overlay returns
  neutral values until a manual reset or the TTL expires.
- **TTL cache.** 5 minutes per (market, symbol) key, so a 32-symbol universe
  does not mean 32 API calls per cycle.
- **Per-feature kill switches.** All six `jev_*` flags live in `cfg` and are
  pushed into `apex_jev` via `configure()`. They previously read as
  `cfg.get("jev_*", True)` against keys that did not exist, which meant every
  feature was hardwired on and impossible to disable.

---

## 6. The RL path

`RL_MODE=true` routes entries through a PPO policy; otherwise the rule-based
scorer decides.

**Observation contract.** 29 features in the order recorded in
`trading_agent/agent/model/feature_columns.json`, scaled by
`scaler.joblib`. `load_saved_feature_columns` fails closed — a missing or
mismatched contract disables the RL path rather than inferring against a
guessed column order, because a wrong order feeds the policy
plausible-looking nonsense.

**Online learning.** After 16 closed trades, `_run_update()` fires on a
background thread. It trains a **private clone** of the policy and publishes
the finished weights through `ModelRegistry.publish()`, which swaps them under
an exclusive lock. Inference holds the read lock for the duration of a forward
pass. The live policy is never stepped in place.

**JEV in the RL path.** `publish_jev_decisions()` writes
`_signals["jev_decisions"]` each cycle, which `rl_signal` reads for
`jev_regime`, `jev_trend_strength`, `jev_action` and `jev_score_delta`. The
key was previously never written, so all four were permanently at their
defaults (`bullish` / `0.0` / `hold`) and the feature was dead.

---

## 7. State and persistence

`_state` is `{india: {...}, us: {...}, sessions: [...], started_at: str}`.
Each market holds cash, positions, realised P&L, win/loss counts, peak
portfolio, drawdown, and the trade log.

Supabase is the source of truth in production (`apex_state` table, one row per
`APEX_ENV`); `apex_dual_state.json` is the local fallback. `APEX_ENV` namespaces
the row id, so a preview deployment cannot stomp production state.

**RLS is disabled** on both tables (`supabase_setup.sql`). The app
authenticates with the service-role key, which bypasses RLS anyway, so there is
no database-layer isolation behind the application. This is a real gap, not an
oversight to be discovered later — see the remaining-work plan.

---

## 8. Configuration

| Source | Scope | Mutable at runtime |
|---|---|---|
| `cfg` dict in `apex_dashboard.py` | trading risk + behaviour | yes, via `/api/config` |
| `apex_market` constants | exchange sessions | no |
| `apex_jev` env constants | TypeSafe thresholds, TTL, budget | no |
| `trading_agent.config.Settings` | RL dataset/training | no |
| env vars | secrets, capital, mode | no |

`trading_agent` previously AST-parsed `apex_dashboard.py` to copy its `cfg` and
watchlists out without importing the Flask app. It returned `{}` on any parse
failure, silently swapping in a hand-maintained copy of the watchlists. Both
now import `apex_universe` directly.

---

## 9. Testing

208 test functions, 293 collected cases, all green on the pinned stack and in
CI. Structure, not just behaviour:

| File | Guards |
|---|---|
| `test_production_regressions.py` | production import order keeps JEV live |
| `test_route_inventory.py` | all 21 routes exist; mutating routes reject GET |
| `test_lock_io_separation.py` | no blocking I/O under `_lock` (AST) |
| `test_silent_failures.py` | no broad `except` that swallows |
| `test_supply_chain.py` | exact pins; numpy inside the pandas-ta window |
| `test_feature_contract.py` | scaler / policy / column file agree on 29 |
| `test_online_learner_real_ppo.py` | copy-and-swap on real torch; fails without the lock |
| `test_agent_loop_resilience.py` | a failed cycle does not kill the trading thread |

---

## 10. Known structural debt

Recorded so it is not rediscovered as a surprise:

- `apex_dashboard.py` is 2,587 lines. `apply_cycle` is 322.
- `agent_loop` orchestrates cycle logic inline rather than through small units.
- `trading_agent/config.py` still reads ~60 env vars directly, overlapping the
  concerns `apex_config` now owns for the dashboard.
- CRLF line endings persist in 28 pre-existing files; `.gitattributes` now
  prevents new drift but nothing has been normalised.
- `trading_agent/integration/bot_bridge.py` is a standalone FastAPI service that
  is never started by the dashboard or the Dockerfile.
