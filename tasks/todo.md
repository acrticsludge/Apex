# Task List: JEV (TypeSafe) Integration

## Phase 0: Foundation

- [x] **Task 01**: Create `apex_jev.py` — JEV client, question schema, state builder
  - **Acceptance criteria:**
    - [x] `apex_jev.py` imports without error
    - [x] `build_market_state()` returns valid JSON string
    - [x] `get_jev_decisions()` returns typed `JEVDecisions` dict
    - [x] `apply_jev_gates()` returns modified config dict
    - [x] All 6 JEV questions defined matching frozen schema
  - **Verification:**
    - [x] `pytest tests/test_jev_gates.py::test_import -v` passes
    - [x] `pytest tests/test_jev_gates.py::test_state_builder -v` passes
    - [x] `pytest tests/test_jev_gates.py::test_gate_functions -v` passes
  - **Dependencies:** None
  - **Files:** `apex_jev.py` (new), `tests/test_jev_gates.py` (new)

- [x] **Task 02**: Add JEV config fields to `trading_agent/config.py`
  - **Acceptance criteria:**
    - [x] All 14 JEV fields added to `Settings` dataclass
    - [x] Fields read from env vars with defaults
    - [x] `settings.jev_enabled` returns correct value
  - **Verification:**
    - [x] `python -c "from trading_agent.config import settings; print(settings.jev_enabled)"` works
    - [x] `pytest tests/test_jev_gates.py::test_config_fields -v` passes
  - **Dependencies:** Task 01
  - **Files:** `trading_agent/config.py`

- [x] **Task 03**: Add `TYPE_SAFE_API_KEY` to `.env.example` and document required env vars
  - **Acceptance criteria:**
    - [x] `.env.example` contains all 20 JEV env vars with comments
    - [x] Each var has description and default
  - **Verification:**
    - [x] `cat .env.example | grep JEV` shows all vars
  - **Dependencies:** Task 02
  - **Files:** `.env.example`

- [x] **Task 04**: Write unit tests for `apex_jev.py` gate functions
  - **Acceptance criteria:**
    - [x] `test_halt_gate()` covers noul > 0.8 / < 0.2
    - [x] `test_graduated_risk()` covers all 4 stress levels
    - [x] `test_position_actions()` covers all 5 actions + confidence gating
    - [x] `test_regime_scaling()` covers all 4 regimes + low confidence
    - [x] `test_news_weighting()` verifies confidence-weighted scoring
    - [x] `test_adx_augmentation()` verifies effective_adx_min calc
  - **Verification:**
    - [x] `pytest tests/test_jev_gates.py -v` — all 37 tests pass
  - **Dependencies:** Task 01
  - **Files:** `tests/test_jev_gates.py` (new)

### Checkpoint: Foundation Complete
- [x] `pytest tests/test_jev_gates.py -v` — all pass
- [x] `python -c "import apex_jev; print('import ok')"` — no import errors
- [x] `ruff check apex_jev.py` — clean

---

## Phase 1: Shadow Mode

- [x] **Task 05**: Add JEV shadow logging in `agent_loop()` — call JEV per cycle, log to `_think_buffer`, **no config changes**
  - **Acceptance criteria:**
    - [x] JEV called once per cycle per market (not per symbol)
    - [x] Results logged to `_think_buffer` with category `JEV`
    - [x] No config modifications in this task
    - [x] Errors caught and logged; cycle continues
  - **Verification:**
    - [x] `grep "JEV" apex.log` shows entries after run
    - [x] `python -c "from apex_dashboard import agent_loop; print('import ok')"` works
  - **Dependencies:** Task 01
  - **Files:** `apex_dashboard.py`

- [x] **Task 06**: Add JEV regime + decisions to `api_state()` response for dashboard visibility
  - **Acceptance criteria:**
    - [x] `api_state()` returns `jev_decisions` dict with all 6 questions
    - [x] Includes regime, trend_strength, news_bullishness, portfolio_stress, halt_new_buys, position_action
    - [x] No breaking changes to existing fields
  - **Verification:**
    - [x] `curl localhost:7000/api/state | jq .jev_decisions` returns valid JSON
  - **Dependencies:** Task 05
  - **Files:** `apex_dashboard.py`

- [x] **Task 07**: Add JEV decisions to `_think_buffer` with category `JEV`
  - **Acceptance criteria:**
    - [x] Each JEV question logged as separate `_think_buffer` entry
    - [x] Category = `JEV`, symbol = relevant symbol or `SYSTEM`
    - [x] Message includes choice/score/noul + confidence
  - **Verification:**
    - [x] `curl localhost:7000/api/think | jq '.[] | select(.cat=="JEV")'` shows entries
  - **Dependencies:** Task 05
  - **Files:** `apex_dashboard.py`

- [x] **Task 08**: Implement regime caching (300s TTL) to reduce API calls
  - **Acceptance criteria:**
    - [x] Regime cached per market (not per symbol)
    - [x] TTL = 300 seconds (configurable via `jev_cache_ttl_seconds`)
    - [x] Cache hit logged at debug level
    - [x] Stale cache → fresh JEV call
  - **Verification:**
    - [x] `pytest tests/test_jev_gates.py::test_regime_cache -v` passes
    - [x] API call count < 2 per cycle after first
  - **Dependencies:** Task 05
  - **Files:** `apex_jev.py`, `apex_dashboard.py`

- [x] **Task 09**: Add circuit breaker: 3 consecutive failures → skip JEV for cycle
  - **Acceptance criteria:**
    - [x] Failure counter increments on timeout/5xx/schema error
    - [x] At 3 failures: `jev_circuit_open = True` for cycle
    - [x] Next cycle: counter reset, JEV attempted again
    - [x] Circuit state logged
  - **Verification:**
    - [x] `pytest tests/test_jev_gates.py::test_circuit_breaker -v` passes
    - [x] Simulated 5xx → JEV skipped gracefully
  - **Dependencies:** Task 05
  - **Files:** `apex_jev.py`, `apex_dashboard.py`

- [x] **Task 10**: Write calibration test (`tests/test_shadow_calibration.py`)
  - **Acceptance criteria:**
    - [x] `test_regime_calibration()` computes Brier score from logged data
    - [x] `test_confidence_calibration()` verifies high-conf → high-accuracy
    - [x] `test_regime_transitions()` checks transition signal quality
    - [x] Tests run against 5-day shadow log
  - **Verification:**
    - [x] `pytest tests/test_shadow_calibration.py -v` passes with mocked data
  - **Dependencies:** Task 05, Task 07
  - **Files:** `tests/test_shadow_calibration.py` (new)

### Checkpoint: Shadow Mode Live
- [x] `pytest tests/test_shadow_calibration.py -v` — passes with mocked data
- [ ] Run `python apex_dashboard.py` for 5 consecutive trading days — zero critical errors
- [ ] Regime accuracy >65% on live data (manual: `python scripts/check_regime_accuracy.py`)
- [ ] Brier score <0.25 (manual: `python scripts/check_brier.py`)
- [ ] API error rate <5% (Grafana: `rate(jev_errors_total[5m]) < 0.05`)

---

## Phase 2: Risk Gates

- [ ] **Task 11**: Implement `apply_jev_gates(cfg, jev)` — returns modified config dict
  - **Acceptance criteria:**
    - [ ] Pure function: input cfg + jev → output cfg (no mutation)
    - [ ] Applies regime scaling (risk_mult, pos_delta, sl_mult)
    - [ ] Applies portfolio stress scaling (risk_reduction)
    - [ ] Returns new dict; original unchanged
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_apply_jev_gates -v` passes
    - [ ] `pytest tests/test_jev_gates.py::test_regime_scaling -v` passes
  - **Dependencies:** Task 04, Task 05
  - **Files:** `apex_jev.py`

- [ ] **Task 12**: Pre-trade halt gate: `if halt_new_buys.noul > 0.8: return` in `apply_cycle()`
  - **Acceptance criteria:**
    - [ ] Check at start of `apply_cycle()` before any entry logic
    - [ ] Logs `think_log("RISK", "JEV halt triggered")` when triggered
    - [ ] Respects `jev_halt_noul_threshold` config
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_halt_gate -v` passes
    - [ ] Manual: inject high halt_noul → no new entries this cycle
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 13**: Graduated daily loss limit: stress 0-1→5%, 1-2→3%, 2-3→1.5%, 3+→0.5%
  - **Acceptance criteria:**
    - [ ] Replaces static `daily_loss_limit_pct` in `apply_cycle()` line 1259
    - [ ] Uses `portfolio_stress.score` to select tier
    - [ ] Logs effective limit: `think_log("RISK", f"Daily loss limit: {limit:.1%}")`
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_graduated_daily_loss -v` passes
    - [ ] Manual: stress=2.5 → limit=1.5%
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 14**: Graduated max drawdown: stress 0-1→8%, 1-2→6%, 2-3→4%, 3+→2%
  - **Acceptance criteria:**
    - [ ] Replaces static `max_drawdown_pct` in `apply_cycle()` line 1258
    - [ ] Uses same stress tier as Task 13
    - [ ] Logs effective limit
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_graduated_drawdown -v` passes
    - [ ] Manual: stress=3.2 → limit=2%
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 15**: Dynamic max_positions: bullish +2, bearish -2, crisis -50%
  - **Acceptance criteria:**
    - [ ] Applied in `apply_cycle()` before entry loop
    - [ ] `max_pos_eff = max_pos + pos_delta` (min 1)
    - [ ] Logs effective max: `think_log("RISK", f"Max positions: {max_pos_eff}")`
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_dynamic_max_pos -v` passes
    - [ ] Manual: regime=bullish → max_pos +2
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 16**: Open window adjustment: crisis 30min, bullish 5min, default 15min
  - **Acceptance criteria:**
    - [ ] Modifies `open_filter_min` in `apply_cycle()` line 1278
    - [ ] Crisis: 30 min; Bullish: 5 min; Bearish/Choppy/Default: 15 min
    - [ ] Logs effective window
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_open_window_adjustment -v` passes
    - [ ] Manual: regime=crisis → open_filter_min=30
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

### Checkpoint: Risk Gates Live
- [ ] `pytest tests/test_jev_gates.py::test_halt_gate -v` — passes
- [ ] `pytest tests/test_jev_gates.py::test_graduated_risk -v` — passes
- [ ] Paper trading 5 days: max drawdown ↓20% vs Phase 1 baseline
- [ ] Annual return >95% of baseline
- [ ] Halt trigger frequency <30% of cycles

---

## Phase 3: Signal Augmentation

- [ ] **Task 17**: Replace `compute_news_score()` with JEV `news_bullishness` in `analyse()`
  - **Acceptance criteria:**
    - [ ] `analyse()` calls `get_jev_decisions()` for news_bullishness
    - [ ] Score formula: `score += round(news_score * 15 * confidence)`
    - [ ] Falls back to keyword scoring if JEV unavailable
    - [ ] Logs JEV news score + confidence
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_news_replacement -v` passes
    - [ ] Manual: JEV news_score=3.0, conf=0.9 → +40 score
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 18**: Weight news by confidence: `score += round(news_score * 15 * confidence)`
  - **Acceptance criteria:**
    - [ ] Implemented in Task 17 (same task)
    - [ ] Low confidence (<0.7) → minimal weight
    - [ ] High confidence (>0.9) → full weight
  - **Verification:** Covered by Task 17 test
  - **Dependencies:** Task 17
  - **Files:** `apex_dashboard.py`

- [ ] **Task 19**: Augment ADX filter: `effective_adx_min = base * (1 + (trend_strength.score-1.5)*0.2)`
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` line 1366 ADX check
    - [ ] Only applied when `trend_strength.confidence > 0.7`
    - [ ] Score 0→0.7x, 1.5→1.0x, 3→1.3x multiplier
    - [ ] Logs effective ADX min
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_adx_augmentation -v` passes
    - [ ] Manual: trend_strength=0 → adx_min * 0.7
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 20**: Calibrate trailing stop: `dist_mult = base * (2.0 - trend_strength.score/3.0)`
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` line 1152 trailing stop logic
    - [ ] Only when `trend_strength.confidence > 0.7`
    - [ ] Strong trend (3.0) → 1.0x (tighter); No trend (0) → 2.0x (wider)
    - [ ] Logs effective dist_mult
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_trailing_calibration -v` passes
    - [ ] Manual: trend_strength=3.0 → dist_mult = base * 1.0
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`

- [ ] **Task 21**: Add JEV trend_strength to indicator engine feature join
  - **Acceptance criteria:**
    - [ ] `indicator_engine.py` accepts optional `jev_trend_strength` param
    - [ ] Adds as column `jev_trend_strength` to feature DataFrame
    - [ ] Used by RL observation builder
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_indicator_join -v` passes
  - **Dependencies:** Task 11
  - **Files:** `trading_agent/data/indicator_engine.py`

### Checkpoint: Signals Augmented
- [ ] `pytest tests/test_jev_gates.py::test_news_weighting -v` — passes
- [ ] `pytest tests/test_jev_gates.py::test_adx_augmentation -v` — passes
- [ ] Paper trading 5 days: win rate ↑3% vs Phase 2 baseline
- [ ] False positive rate ↓15%
- [ ] Trade frequency >80% of baseline

---

## Phase 4: Position Actions

- [ ] **Task 22**: Position action gate: require `action=="buy" AND conf>0.65` for long entry
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` long entry block (line 1394)
    - [ ] Calls `get_position_action(jev)` helper
    - [ ] Only enters if action=="buy" AND confidence >= threshold
    - [ ] Logs JEV action + confidence
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_long_entry_gate -v` passes
    - [ ] Manual: action=buy, conf=0.6 → no entry; conf=0.7 → entry
  - **Dependencies:** Task 11
  - **Files:** `apex_dashboard.py`, `apex_jev.py` (helper)

- [ ] **Task 23**: Short entry gate: require `action=="buy" (for short) AND conf>0.65`
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` short entry block (line 1419)
    - [ ] Maps JEV "buy" → short entry (JEV doesn't distinguish long/short)
    - [ ] Same confidence threshold
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_short_entry_gate -v` passes
  - **Dependencies:** Task 22
  - **Files:** `apex_dashboard.py`

- [ ] **Task 24**: Exit override: `action=="exit" AND conf>0.65` → immediate sell/cover
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` position management (lines 1194-1237)
    - [ ] Checks JEV action before SL/TP/RL exit
    - [ ] Reason = "JEV_EXIT"
    - [ ] Logs JEV exit with confidence
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_exit_override -v` passes
    - [ ] Manual: action=exit, conf=0.7 → immediate exit
  - **Dependencies:** Task 22
  - **Files:** `apex_dashboard.py`

- [ ] **Task 25**: Trim action: `action=="trim" AND conf>0.7` → sell 50% position
  - **Acceptance criteria:**
    - [ ] New helper `execute_trim(symbol, qty//2, ...)` in paper trading layer
    - [ ] In `apply_cycle()` after exit override check
    - [ ] Updates position qty, logs partial exit
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_trim_action -v` passes
    - [ ] Manual: action=trim, conf=0.75 → qty halved
  - **Dependencies:** Task 22
  - **Files:** `apex_dashboard.py` (paper trading functions)

- [ ] **Task 26**: Add (pyramid) action: `action=="add" AND conf>0.75` → increase position
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` long entry block
    - [ ] Only if already in position (in_pos=True)
    - [ ] Calls `execute_buy()` with additional qty
    - [ ] Respects max_positions limit
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_add_action -v` passes
    - [ ] Manual: action=add, conf=0.8 → position increased
  - **Dependencies:** Task 22
  - **Files:** `apex_dashboard.py`

- [ ] **Task 27**: Signal exit/cover override by JEV action
  - **Acceptance criteria:**
    - [ ] In `apply_cycle()` signal exit block (lines 1322-1335)
    - [ ] Long: if JEV action=="exit" → override score< -30 check
    - [ ] Short: if JEV action=="exit" → override score>30 check
    - [ ] Reason = "JEV_SIGNAL_EXIT"
  - **Verification:**
    - [ ] `pytest tests/test_jev_gates.py::test_signal_exit_override -v` passes
  - **Dependencies:** Task 24
  - **Files:** `apex_dashboard.py`

### Checkpoint: Position Actions Live
- [ ] `pytest tests/test_jev_gates.py::test_position_actions -v` — passes
- [ ] Paper trading 5 days: false exit rate <10%
- [ ] Trim actions show positive expectancy (avg P&L > 0)
- [ ] Pyramiding (`add`) shows positive expectancy

---

## Phase 5: RL Integration

- [ ] **Task 28**: Extend RL observation space (+8 JEV dims) in `rl_signal.py` and `config.py`
  - **Acceptance criteria:**
    - [ ] `JEV_FEATURE_COLUMNS` added to `settings.feature_columns`
    - [ ] `_build_observation()` concatenates JEV features
    - [ ] Features normalized to [0,1]
    - [ ] Observation size increases from 22→30
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_obs_extension -v` passes
    - [ ] `python -c "from trading_agent.integration.rl_signal import _build_observation; print(_build_observation('AAPL').shape)"` → (30,)
  - **Dependencies:** Task 11
  - **Files:** `trading_agent/integration/rl_signal.py`, `trading_agent/config.py`

- [ ] **Task 29**: Regime-aware entropy/margin gating in `rl_signal.py`
  - **Acceptance criteria:**
    - [ ] Crisis regime: entropy threshold * 0.8 (stricter)
    - [ ] Bullish regime: margin threshold * 0.9 (looser)
    - [ ] Logs regime-adjusted thresholds
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_regime_gating -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/integration/rl_signal.py`

- [ ] **Task 30**: JEV action → RL score mapping (buy=+60, add=+30, hold=0, trim=-30, exit=-60)
  - **Acceptance criteria:**
    - [ ] In `get_rl_signal()` score calculation (line 202)
    - [ ] Maps JEV position_action to score delta
    - [ ] Combined with existing norm_prob + trend_align
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_action_score_mapping -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/integration/rl_signal.py`

- [ ] **Task 31**: Extend `bot_bridge.py` schema + response metadata
  - **Acceptance criteria:**
    - [ ] `PredictionRequest` adds optional `jev_features: list[float]`
    - [ ] `/predict` response includes `jev_regime_probs`
    - [ ] `/status` includes `jev_version`, `jev_last_call_latency_ms`
    - [ ] Backward compatible (jev_features optional)
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_bridge_schema -v` passes
    - [ ] `curl -X POST localhost:8000/predict -d '{"observation": [...]}'` works
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/integration/bot_bridge.py`

- [ ] **Task 32**: Regime-conditioned online learner in `online_learner.py`
  - **Acceptance criteria:**
    - [ ] `record_entry()` stores regime at entry
    - [ ] `record_exit()` weights reward by inverse portfolio_stress
    - [ ] Buffer stratified by regime for updates
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_online_learner -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/integration/online_learner.py`

- [ ] **Task 33**: Regime-stratified validation callback in `train.py`
  - **Acceptance criteria:**
    - [ ] `ValidationSharpeEvalCallback` computes metrics per regime
    - [ ] Selection score includes regime-weighted component
    - [ ] Best model chosen by composite across regimes
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_stratified_validation -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/agent/train.py`

- [ ] **Task 34**: Regime-attributed evaluation metrics in `evaluate.py`
  - **Acceptance criteria:**
    - [ ] `evaluate_model_on_frames()` returns per-regime metrics
    - [ ] Win rate, Sharpe, DD, trade count per regime
    - [ ] Summary includes regime comparison table
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_regime_metrics -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/agent/evaluate.py`

- [ ] **Task 35**: Extended `TradingEnv` observation space in `trading_env.py`
  - **Acceptance criteria:**
    - [ ] `observation_space` shape = (30,) instead of (22,)
    - [ ] `_get_observation()` returns 30-dim vector
    - [ ] Compatible with existing checkpoints (graceful handling)
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_env_obs_space -v` passes
  - **Dependencies:** Task 28
  - **Files:** `trading_agent/environment/trading_env.py`

- [ ] **Task 36**: Curriculum sampling by regime in `train.py`
  - **Acceptance criteria:**
    - [ ] `_sample_episode_bounds()` oversamples crisis/bearish
    - [ ] Target: 25% crisis, 25% bearish, 25% bullish, 25% choppy
    - [ ] Logs regime distribution per epoch
  - **Verification:**
    - [ ] `pytest tests/test_rl_integration.py::test_curriculum_sampling -v` passes
  - **Dependencies:** Task 28, Task 33
  - **Files:** `trading_agent/agent/train.py`

- [ ] **Task 37**: Full PPO retrain with JEV features (8 GPU-hours)
  - **Acceptance criteria:**
    - [ ] Training completes 500k timesteps without divergence
    - [ ] Best validation Sharpe > baseline + 0.15
    - [ ] Regime-attributed metrics all improved vs baseline
    - [ ] Model saved to `best_model.zip`
  - **Verification:**
    - [ ] `python -m trading_agent.main --mode evaluate` — RL+JEV Sharpe > RL-only +0.15
    - [ ] Training curves in TensorBoard healthy
  - **Dependencies:** Task 28-36
  - **Files:** `trading_agent/agent/train.py` (run)

### Checkpoint: RL Retrained
- [ ] `python -m trading_agent.main --mode evaluate` — RL+JEV Sharpe > RL-only +0.15
- [ ] Regime-attributed metrics show improvement in crisis/bear
- [ ] Training stable (no divergence)
- [ ] Inference latency <100ms

---

## Phase 6: Production Hardening

- [ ] **Task 38**: Security hardening — API key handling, rate limiting, input validation
  - **Acceptance criteria:**
    - [ ] API key never logged (filter in logging config)
    - [ ] Rate limit: max 10 calls/cycle enforced
    - [ ] JEV response schema validated (pydantic model)
    - [ ] No PII in state sent to JEV
  - **Verification:**
    - [ ] `pytest tests/test_security.py -v` passes
    - [ ] `grep -r "TYPE_SAFE_API_KEY" *.py` → only in config/env
  - **Dependencies:** Task 01
  - **Files:** `apex_jev.py`, `apex_dashboard.py`

- [ ] **Task 39**: Observability — latency histogram, error counter, cost tracker in dashboard
  - **Acceptance criteria:**
    - [ ] `jev_latency_seconds` histogram (prometheus client)
    - [ ] `jev_errors_total` counter
    - [ ] `jev_cost_usd_total` counter
    - [ ] Exposed at `/metrics` endpoint
  - **Verification:**
    - [ ] `curl localhost:7000/metrics | grep jev` shows metrics
    - [ ] Grafana dashboard panels render
  - **Dependencies:** Task 38
  - **Files:** `apex_dashboard.py`, `apex_jev.py`

- [ ] **Task 40**: Code review pass (security, quality, simplification)
  - **Acceptance criteria:**
    - [ ] No security findings (bandit/secrets scan)
    - [ ] No quality issues (ruff, mypy clean)
    - [ ] Simplification: no duplicate logic, clear separation
    - [ ] All functions have docstrings + type hints
  - **Verification:**
    - [ ] `ruff check .` — clean
    - [ ] `mypy .` — clean (if configured)
    - [ ] `bandit -r .` — no high findings
  - **Dependencies:** All previous tasks
  - **Files:** All modified

- [ ] **Task 41**: Feature flags per phase (env vars) in `config.py`
  - **Acceptance criteria:**
    - [ ] `JEV_REGIME_ENABLED`, `JEV_TREND_ENABLED`, `JEV_NEWS_ENABLED`, `JEV_RISK_ENABLED`, `JEV_ACTION_ENABLED`
    - [ ] Each gate checks its flag before executing
    - [ ] Default: all false except regime/trend/news
  - **Verification:**
    - [ ] `pytest tests/test_feature_flags.py -v` passes
    - [ ] `JEV_RISK_ENABLED=false` → risk gates skipped
  - **Dependencies:** Task 02
  - **Files:** `trading_agent/config.py`, `apex_dashboard.py`, `apex_jev.py`

- [ ] **Task 42**: Rollback runbook `docs/operations/jev-rollback.md`
  - **Acceptance criteria:**
    - [ ] Instant rollback: `export JEV_ENABLED=false`
    - [ ] Selective: per-phase flags
    - [ ] Verification steps for fallback behavior
    - [ ] Contact info for TypeSafe support
  - **Verification:**
    - [ ] `cat docs/operations/jev-rollback.md` — complete
    - [ ] Manual test: toggle flag → verify rule behavior
  - **Dependencies:** Task 41
  - **Files:** `docs/operations/jev-rollback.md` (new)

- [ ] **Task 43**: Monitoring dashboards (Grafana) + alerts
  - **Acceptance criteria:**
    - [ ] Panel: JEV Latency p50/p95/p99
    - [ ] Panel: JEV Error Rate
    - [ ] Panel: JEV Daily Cost
    - [ ] Panel: Regime Distribution
    - [ ] Alert: p95 latency >500ms
    - [ ] Alert: error rate >5%
    - [ ] Alert: daily cost >$75
  - **Verification:**
    - [ ] Grafana dashboard `apex-jev` exists
    - [ ] Alert rules fire in test
  - **Dependencies:** Task 39
  - **Files:** `grafana/dashboards/apex-jev.json` (new), `grafana/alerts/jev.yml` (new)

### Checkpoint: Production Ready
- [ ] `pytest tests/ -v` — all pass
- [ ] `ruff check .` — clean
- [ ] `mypy .` — clean (if configured)
- [ ] `docker build -t apex .` — passes
- [ ] Rollback tested: `JEV_ENABLED=false` → rule behavior verified

---

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