# Implementation Plan: JEV (TypeSafe) Integration

## Overview
Integrate TypeSafe JEV (System One judgment model) as a structured overlay on the Apex trading system. JEV provides calibrated, typed judgments (regime, risk gates, news synthesis, position actions) that gate and size decisions without replacing the core rule-based or RL logic. Implementation follows 5 phases over 8 weeks with explicit go/no-go gates.

## Architecture Decisions
- **Overlay Pattern (ADR-0001)**: JEV never replaces core logic; only modifies config, vetoes entries, or provides confidence-weighted signals
- **Phase-Gated Rollout**: Each phase has measurable criteria; rollback via single env var toggle
- **Shadow-First**: Phase 1 logs only; validates calibration before any capital at risk
- **RL Retrain at Phase 5**: Extended observation space (+8 dims) requires full PPO retrain

## Task List

### Phase 0: Foundation (Week 1)
- [ ] **Task 01**: Create `apex_jev.py` — JEV client, question schema, state builder
- [ ] **Task 02**: Add JEV config fields to `trading_agent/config.py`
- [ ] **Task 03**: Add `TYPE_SAFE_API_KEY` to `.env.example` and document required env vars
- [ ] **Task 04**: Write unit tests for `apex_jev.py` gate functions (`tests/test_jev_gates.py`)

### Checkpoint: Foundation
- [ ] `pytest tests/test_jev_gates.py -v` — all pass
- [ ] `python -c "import apex_jev; print('import ok')"` — no import errors
- [ ] `ruff check apex_jev.py` — clean

### Phase 1: Shadow Mode (Week 1-2)
- [ ] **Task 05**: Add JEV shadow logging in `agent_loop()` — call JEV per cycle, log to `_think_buffer`, **no config changes**
- [ ] **Task 06**: Add JEV regime + decisions to `api_state()` response for dashboard visibility
- [ ] **Task 07**: Add JEV decisions to `_think_buffer` with category `JEV`
- [ ] **Task 08**: Implement regime caching (300s TTL) to reduce API calls
- [ ] **Task 09**: Add circuit breaker: 3 consecutive failures → skip JEV for cycle
- [ ] **Task 10**: Write calibration test (`tests/test_shadow_calibration.py`) — regime accuracy, Brier score

### Checkpoint: Shadow Mode Live
- [ ] `pytest tests/test_shadow_calibration.py -v` — passes with mocked data
- [ ] Run `python apex_dashboard.py` for 5 consecutive trading days — zero critical errors
- [ ] Regime accuracy >65% on live data (manual verification)
- [ ] Brier score <0.25 (manual verification)
- [ ] API error rate <5% (Grafana)

### Phase 2: Risk Gates (Week 3)
- [ ] **Task 11**: Implement `apply_jev_gates(cfg, jev)` — returns modified config dict
- [ ] **Task 12**: Pre-trade halt gate: `if halt_new_buys.noul > 0.8: return` in `apply_cycle()`
- [ ] **Task 13**: Graduated daily loss limit: stress 0-1→5%, 1-2→3%, 2-3→1.5%, 3+→0.5%
- [ ] **Task 14**: Graduated max drawdown: stress 0-1→8%, 1-2→6%, 2-3→4%, 3+→2%
- [ ] **Task 15**: Dynamic max_positions: bullish +2, bearish -2, crisis -50%
- [ ] **Task 16**: Open window adjustment: crisis 30min, bullish 5min, default 15min

### Checkpoint: Risk Gates Live
- [ ] `pytest tests/test_jev_gates.py::test_halt_gate -v` — passes
- [ ] `pytest tests/test_jev_gates.py::test_graduated_risk -v` — passes
- [ ] Paper trading 5 days: max drawdown ↓20% vs Phase 1 baseline
- [ ] Annual return >95% of baseline
- [ ] Halt trigger frequency <30% of cycles

### Phase 3: Signal Augmentation (Week 4)
- [ ] **Task 17**: Replace `compute_news_score()` with JEV `news_bullishness` in `analyse()`
- [ ] **Task 18**: Weight news by confidence: `score += round(news_score * 15 * confidence)`
- [ ] **Task 19**: Augment ADX filter: `effective_adx_min = base * (1 + (trend_strength.score-1.5)*0.2)`
- [ ] **Task 20**: Calibrate trailing stop: `dist_mult = base * (2.0 - trend_strength.score/3.0)`
- [ ] **Task 21**: Add JEV trend_strength to indicator engine feature join

### Checkpoint: Signals Augmented
- [ ] `pytest tests/test_jev_gates.py::test_news_weighting -v` — passes
- [ ] `pytest tests/test_jev_gates.py::test_adx_augmentation -v` — passes
- [ ] Paper trading 5 days: win rate ↑3% vs Phase 2 baseline
- [ ] False positive rate ↓15%
- [ ] Trade frequency >80% of baseline

### Phase 4: Position Actions (Week 5)
- [ ] **Task 22**: Position action gate: require `action=="buy" AND conf>0.65` for long entry
- [ ] **Task 23**: Short entry gate: require `action=="buy" (for short) AND conf>0.65`
- [ ] **Task 24**: Exit override: `action=="exit" AND conf>0.65` → immediate sell/cover
- [ ] **Task 25**: Trim action: `action=="trim" AND conf>0.7` → sell 50% position
- [ ] **Task 26**: Add (pyramid) action: `action=="add" AND conf>0.75` → increase position
- [ ] **Task 27**: Signal exit/cover override by JEV action

### Checkpoint: Position Actions Live
- [ ] `pytest tests/test_jev_gates.py::test_position_actions -v` — passes
- [ ] Paper trading 5 days: false exit rate <10%
- [ ] Trim actions show positive expectancy (avg P&L > 0)
- [ ] Pyramiding (`add`) shows positive expectancy

### Phase 5: RL Integration (Week 6-7)
- [ ] **Task 28**: Extend RL observation space (+8 JEV dims) in `rl_signal.py` and `config.py`
- [ ] **Task 29**: Regime-aware entropy/margin gating in `rl_signal.py`
- [ ] **Task 30**: JEV action → RL score mapping (buy=+60, add=+30, hold=0, trim=-30, exit=-60)
- [ ] **Task 31**: Extend `bot_bridge.py` schema + response metadata
- [ ] **Task 32**: Regime-conditioned online learner in `online_learner.py`
- [ ] **Task 33**: Regime-stratified validation callback in `train.py`
- [ ] **Task 34**: Regime-attributed evaluation metrics in `evaluate.py`
- [ ] **Task 35**: Extended `TradingEnv` observation space in `trading_env.py`
- [ ] **Task 36**: Curriculum sampling by regime in `train.py`
- [ ] **Task 37**: Full PPO retrain with JEV features (8 GPU-hours)

### Checkpoint: RL Retrained
- [ ] `python -m trading_agent.main --mode evaluate` — RL+JEV Sharpe > RL-only +0.15
- [ ] Regime-attributed metrics show improvement in crisis/bear
- [ ] Training stable (no divergence, loss curves healthy)
- [ ] Inference latency <100ms

### Phase 6: Production Hardening (Week 8)
- [ ] **Task 38**: Security hardening — API key handling, rate limiting, input validation
- [ ] **Task 39**: Observability — latency histogram, error counter, cost tracker in dashboard
- [ ] **Task 40**: Code review pass (security, quality, simplification)
- [ ] **Task 41**: Feature flags per phase (env vars) in `config.py`
- [ ] **Task 42**: Rollback runbook `docs/operations/jev-rollback.md`
- [ ] **Task 43**: Monitoring dashboards (Grafana) + alerts

### Checkpoint: Production Ready
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Lint clean: `ruff check .`
- [ ] Typecheck clean: `mypy .` (if configured)
- [ ] Build passes: `docker build -t apex .`
- [ ] Rollback tested: `JEV_ENABLED=false` → rule behavior verified

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| JEV API unavailable | High — gates fail open | Circuit breaker + 300s regime cache + rule defaults |
| Confidence miscalibration | High — wrong vetoes/entries | Shadow phase calibration; thresholds tunable via env |
| RL retrain divergence | High — broken policy | Checkpoint callback; resume from best; A/B test |
| Latency >500ms/cycle | Medium — missed ticks | Async calls; batch top-K; cache regime |
| Daily cost >$75 | Medium — budget overrun | Call budget (10/cycle); daily cost alert |
| Regime lag | Medium — stale gates | Cycle-start regime persists for cycle (3-15 min) |

## Open Questions
- [ ] TypeSafe API key provisioning — **BLOCKER for Task 01**
- [ ] Exact JEV model version pinning (`jev-latest` vs `jev-1.13.0`)
- [ ] Whether to run Phase 5 RL retrain on GPU cloud (Modal/Lambda) or local
- [ ] Dashboard UI changes for JEV regime badge — scope TBD

## Parallelization Opportunities
- **Tasks 01-04**: Sequential (foundation)
- **Tasks 05-09**: Sequential (shadow depends on 01)
- **Tasks 11-16**: Parallel after 05-09 (independent gates)
- **Tasks 17-21**: Parallel after 11-16 (independent signals)
- **Tasks 22-27**: Sequential (actions build on each other)
- **Tasks 28-36**: Parallel after 27 (independent RL components)
- **Task 37**: Sequential after 28-36 (retrain)
- **Tasks 38-43**: Parallel after 37 (hardening)