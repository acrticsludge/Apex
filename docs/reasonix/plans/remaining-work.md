# Remaining Work Plan

Created 2026-09-29, after the `feat/jev-integration` hardening work landed
(tip `200e6c0c`, 293 tests green, CI 3/3). Kept current as work landed; the
counts below are as of `583d4be` plus the online-learner validation gate.

This plan covers what is left. It is ordered by risk, not by convenience.
Phases 0 and 1 require a human; everything from Phase 2 onward is
engineering work that can be planned and executed against the test suite.

---

## Where the project is

| Area | State |
|---|---|
| Correctness | 4 critical + 6 high defects found in review, all fixed with regression tests |
| Concurrency | 3 races fixed (state lock, model weights, re-entrant price lock) |
| Security | Auth fails closed; all client-writable values validated; 20 advisories cleared |
| Supply chain | 22 exact pins with constraint tests; Dependabot; pip-audit gates CI |
| CI | 3 jobs: suite, import-order assertion + audit, Docker build + secret scan |
| Tests | 421 cases green locally on the pinned stack and on CI, covering the entry decision, the Supabase path, model artifacts, the learner validation gate and execution costs |
| Deployment | **never deployed.** The changes have not run against a live market. |
| Documentation | architecture overview, 5 ADRs, runbook (this cycle) |

The code is in materially better shape than it was. The dominant risk has moved
from *correctness* to *deployment*: nothing here has yet been exercised against
real Supabase state or a live market session.

---

## Phase 0 — Before the first deploy (human, blocking)

Nothing else should start until these are done, because a mistake here is
either a lockout or an exposure.

- [ ] **Confirm the Railway secrets are set and are not the defaults.**
      `APEX_SECRET`, `APEX_USER`, `APEX_PASS` (or `APEX_PASS_HASH`). This has
      been flagged through three review cycles and never confirmed. The app now
      fails closed, so a miss locks the dashboard rather than opening it with
      `admin`. Checklist: `docs/operations/deployment-runbook.md` §1.1.
- [ ] **Confirm `SUPABASE_SERVICE_KEY` scope.** It is currently service-role,
      and RLS is off, so it is full read/write on the trading ledger. Decide
      whether that is acceptable or whether Phase 2's RLS work should come first.
- [ ] **Open the PR and get it reviewed.**
      `https://github.com/acrticsludge/Apex/pull/new/feat/jev-integration`
      Six commits, ~6k lines, no second pair of eyes yet. Given how much surgery
      went into a live trading loop, this is the highest-value remaining action.
- [ ] **Merge, then canary one full cycle unattended-but-watched.** The JEV risk
      gates go from inert to live on first run. Runbook §3.

## Phase 1 — Close the process gaps (human, ~15 min)

- [ ] **Enable branch protection** on the deploy branch: require CI to pass,
      require one approval, disallow force-push. CI currently reports; it does
      not block. Every guard added in this cycle is advisory until this is set.
- [ ] **Configure Dependabot grouping** to your taste. It is written but
      never observed; a noisy group teaches people to ignore it.
- [ ] **Free the local disk.** The RL stack needs ~4 GB for torch. Not a code
      issue, but it is why the online-learner fix was originally proven against
      a fake.

## Phase 2 — Security follow-through (engineering, ~1 day)

The review's remaining HIGH items, none of which are cosmetic.

- [ ] **Restore RLS on `apex_state` and `apex_rl_decisions`.** Currently
      `ALTER TABLE ... DISABLE ROW LEVEL SECURITY` in `supabase_setup.sql`.
      The service-role key bypasses RLS regardless, so this is defence in depth
      — but it is the difference between "a leaked key exposes the ledger" and
      "a leaked key is scoped". Requires deciding on anon/authenticated policies
      and re-testing persistence.
- [x] **Stop writing model artifacts to a tracked path.** Done in
      `trading_agent/integration/artifact_store.py`, wired into the online
      learner's save and rl_signal's load. An artifact now has an immutable
      **baseline** (the model in the image, so a fresh deploy has something to
      load) and a mutable **runtime** copy; reads prefer the runtime and fall
      back to the baseline, writes only ever touch the runtime, and a newly
      mounted volume is seeded from the baseline once and never re-seeded.
      Saves go through a temp file and `os.replace`, so a crash mid-save leaves
      the previous model rather than a truncated zip — which matters more on a
      volume, because the volume outlives the process that corrupted it. The
      `best_model_pre_online` backup now rotates instead of accumulating one
      copy per update.
      **Inert until you set `TRADING_AGENT_STORAGE_DIR`** — with it unset the
      runtime directory *is* the tracked directory, so behaviour is unchanged
      and the app logs once at boot that updates will not survive a redeploy.
      Turn it on with a Railway volume at `/data`; see
      [deployment runbook §1.4](../operations/deployment-runbook.md).
- [ ] **Rotate the committed model artifacts.** Still worth doing, and now
      decoupled from persistence: the baseline is what ships in the image, so
      rotating it out of git means building the model in CI and shipping it as
      an artifact instead. Separate question from where updates are written, and
      lower priority than it was.
- [ ] **Add an IP allowlist or tunnel** for the dashboard. It is a public
      Railway hostname today. Optional given auth is now fail-closed, but it is
      a meaningful second factor.
- [ ] **Move the Supabase service key out of the long-lived app** toward a
      narrower credential, if the platform allows it.

## Phase 3 — Structural debt (engineering, ~2 days)

Recorded here so it is not rediscovered as a surprise. None of it is a
correctness risk; all of it slows the next change.

- [x] ~~**Decompose `apply_cycle` (322 lines).**~~ **Deliberately not done —
      the plan was wrong.** The exit side was genuinely a list of independent
      triggers (SL / TP / trailing / EOD), so `run_exit_policies` removed a real
      duplication and made the ordering explicit. The entry side is one
      interlocking decision, not a list of policies: the index gate deliberately
      does *not* short-circuit so a bearish signal can still reach the short
      branch, the JEV action gate is computed once and consumed by both
      branches, and the cooldown check carries a state mutation in its `elif`.
      Extracting a policy layer would thread a dozen locals through a new
      interface to preserve behaviour exactly — more indirection at the same
      risk. `apply_cycle`'s remaining problem was never length, it was that the
      entry decision was untested; that is now fixed, so the refactor is no
      longer worth its risk. Revisit only if a second caller needs the decision.
- [ ] **Decompose `apex_dashboard.py` (2,587 lines).** The natural next
      extraction is the trade-execution layer (`execute_buy` / `execute_short` /
      `execute_sell` / `execute_cover`, ~250 lines) and the state layer
      (`load_state` / `save_state` / `_normalize_state`, ~200). Both are already
      boundary-clean. Higher value than the `apply_cycle` split, because these
      boundaries are real rather than notional.
- [ ] **Split `agent_loop`.** Session rotation, JEV gate application and
      persistence could each be a unit. The resilience wrapper is already
      separate.
- [ ] **Fold `trading_agent/config.py` into the shared config module.** It
      still reads ~60 env vars directly, overlapping what `apex_config` now
      owns for the dashboard. Lower priority: the two serve different runtimes.
- [ ] **Normalise CRLF** across the 28 pre-existing files. `.gitattributes`
      prevents new drift; nothing has been rewritten. Deliberately left as a
      standalone mechanical change so it does not pollute a behavioural diff.

## Phase 4 — Test and process depth (engineering, ongoing)

- [x] **Add a real Supabase integration test.** Done in
      `tests/test_supabase_persistence.py` (13 tests). Drives `save_state` /
      `load_state` / `save_cfg` / `load_cfg` / `_save_rl_decision` against a
      stub with postgrest's real chainable surface and its real `APIError`, so
      the `except` clauses are exercised as they would be in production. Covers
      the round trip, JSON-safety of the payload, `APEX_ENV` row namespacing, the
      config whitelist, and four failure modes (failed write falls back to JSON
      and logs; failed read falls back; missing table survivable; malformed row
      still yields both markets). The lock is asserted twice — once that
      Supabase is not called under it, once that the probe would notice if it
      were. One test is opt-in against a real project via
      `APEX_TEST_SUPABASE_URL` / `APEX_TEST_SUPABASE_KEY`.
- [x] **Add coverage for the JEV halt path's entry side.** Done in
      `tests/test_entry_policies.py` (26 tests), which covers the whole entry
      decision rather than only halt-path parity: both branches, the index
      gate's opposite polarity per direction, the RL long bypass, the ADX floor,
      the re-entry cooldown, and the daily-loss and drawdown kill-switches.
      The halt path itself returns before any entry work, which the tests
      confirm — under a halt nothing is entered at all.
- [x] **Gate the online learner on held-out validation.** Done in
      `trading_agent/integration/online_learner.py`. An update used to be
      published whenever the training loss fell, which only proves the policy
      fitted the batch harder; the module had no held-out data at all. Now the
      most recent 20% of the buffer is held out before training, the incumbent
      and the candidate are both scored on that identical slice with the same
      objective, and the update is applied only if the candidate wins by more
      than `_MIN_VALIDATION_IMPROVEMENT_PCT`. The split is walk-forward, never
      random: over a time series a random split trains on the future and
      validates on the past, which looks like progress and is not. Every branch
      fails closed, including NaN/inf, which are excluded explicitly because
      every NaN comparison is False (silently rejecting everything) and inf beats
      any finite value (silently accepting anything).
      This was a gap the persistence work introduced: updates used to evaporate
      on redeploy, so a bad one was self-cancelling. Now that they survive, an
      unvalidated one is permanent.
      Batches below `_MIN_BATCH_FOR_VALIDATION` (12) now skip rather than
      training unvalidated, so the learner fires less often early on. Skipped
      batches stay in the buffer and are used by the next update.
      **Watch `rejected_updates`** in `/api/retrain/log`. A run of rejections is
      the gate working. A persistently high count with no acceptances means the
      learner is not learning and the feature set is the likely problem.
- [x] **Model slippage, not just commission.** `commission_pct` (0.06% per side)
      was deducted correctly, but every fill happened at the quoted price. Real
      fills do not, so paper P&L was optimistic by roughly the round-trip spread
      — and the position cap, daily loss limit and drawdown kill-switch are all
      calibrated against that number.
      Now applied in `_fill_price()` as a **fill price** rather than as a fee:
      `execute_buy` and `execute_cover` fill above the quote, `execute_sell` and
      `execute_short` below. Everything downstream — position sizing, the cash
      guard, stop, target, running high/low, the recorded entry — uses the fill,
      so a long round trip pays slippage once on entry and once on exit with no
      extra bookkeeping. Default 5bps per side (`slippage_pct`), editable in the
      settings panel and bounded to [0, 0.05] by `apex_config`.
      Modelling it as a fill price rather than a fee is deliberate: it is what
      makes both sides land in P&L. See the commission asymmetry below for the
      contrast.
- [ ] **Charge entry commission to P&L.** Found while testing slippage, and
      pre-existing. Entry commission is deducted from cash when a position opens,
      but `realised_pnl` is computed from the exit side only
      (`net_proceeds - entry * qty`). So the entry cost never reaches
      `realised_pnl`, the win/loss counters, the daily loss limit or the
      drawdown kill-switch — all four read one side cheaper than the trade
      actually was. `test_entry_commission_is_missing_from_realised_pnl` pins the
      current behaviour so a fix lands as a visible change rather than a silent
      one. **Not fixed here**: it alters reported performance and the kill-switch
      thresholds, which is a decision for the operator, not a refactor.
- [ ] **Indian transaction costs are understated.** Indian equity carries STT
      (~0.1% on the buy side for delivery), stamp duty and GST. The model applies
      a single global 0.06% commission to both markets, so India is cheaper than
      it is and the US is about right. Any conclusion drawn from the India
      ledger is currently too favourable.
- [ ] **Adopt a lockfile** (`uv.lock` or `pip-compile`) so the transitive graph
      is hash-verified rather than inferred from exact pins. See ADR-005.
- [ ] **Add mutation testing** on the risk gates. A test that asserts a gate is
      applied would pass even if the gate were inverted; mutation testing is
      what catches that.
- [ ] **Alert on `Agent has failed N consecutive cycles`.** The log line exists;
      nothing watches it. This is the single highest-value observability gap
      for an unattended trading bot.

---

## Explicitly not planned

- **Live order placement.** The paper layer is the only execution path. Moving
  to live orders is a product decision, not an engineering one, and it would
  change the risk profile of everything above.
- **Multi-account / portfolio support.** One account, two markets, one ledger.
- **Backtesting on the new stack.** The existing `colab_train.ipynb` and
  `trading_agent/agent/train.py` predate the pin changes and have not been run
  under torch 2.14. That is a prerequisite for trusting any retrained model.
