# JEV (TypeSafe) Integration Specification

**Project**: Apex Trading Dashboard  
**Date**: 2026-09-22  
**Status**: **DEFINE** → **ARCHITECT** → **PLAN** (Ready for Implementation)  
**Author**: AI Assistant  
**Lifecycle Stage**: Post-DEFINE, Pre-IMPLEMENT  

---

## 0. ORIENT / DISCOVER — Context Summary

### Repository State
- **Primary**: `apex_dashboard.py` — Flask + paper trading dashboard (production)
- **Secondary**: `trading_agent/` — RL system (PPO + Gymnasium + Stable-Baselines3, training/serving)
- **State**: `apex_dual_state.json` + Supabase persistence
- **Data**: yfinance (price), Finnhub (news), pandas-ta (indicators)

### Existing Decision Points (from code audit)
| System | Decision Points | Current Mechanism |
|--------|-----------------|-------------------|
| Dashboard | Regime detection | Index % change only |
| Dashboard | Risk limits | Static thresholds (5% daily, 8% DD) |
| Dashboard | Entry gates | Confidence threshold + ADX + index filter |
| Dashboard | Exit logic | SL/TP/trailing/RL-entropy/EOD |
| Dashboard | News sentiment | Keyword matching (VADER-lite) |
| RL | Observation | 22 technical features |
| RL | Reward | Sharpe + excess return - penalties |
| RL | Training | Uniform across all regimes |

### Gap Analysis
| Capability | Current | Needed |
|------------|---------|--------|
| Regime-aware risk | ❌ | ✅ |
| Contextual news synthesis | ❌ | ✅ |
| Calibrated confidence | ❌ | ✅ |
| Unified position actions | ❌ | ✅ |
| Graduated risk scaling | ❌ | ✅ |

---

## 1. DEFINE — Requirements Specification

### 1.1 Problem Statement
The Apex trading system makes decisions using **hard-coded thresholds** and **single-indicator rules** that don't adapt to market regime, portfolio stress, or news context. This leads to:
- False entries in choppy markets
- Missed exits during regime shifts
- Over/under-sizing in bull/bear/crisis
- News interpreted without symbol/sector context

### 1.2 Users
- **Primary**: Apex dashboard operator (human-in-the-loop)
- **Secondary**: RL agent (automated policy)
- **Tertiary**: Dashboard UI (visualization)

### 1.3 Goals
| Goal | Metric | Target |
|------|--------|--------|
| Reduce max drawdown | Peak-to-trough | -25% vs baseline |
| Improve win rate | Winning trades / total | +3-5% |
| Reduce false positives | Losing buys / total buys | -15% |
| Maintain trade frequency | Trades/day | >80% of baseline |
| Crisis alpha | Return in drawdown >10% | Flat or positive |

### 1.4 Non-Goals
- ❌ Replace rule engine entirely
- ❌ Replace RL policy entirely
- ❌ Make JEV the sole decision maker
- ❌ Add latency >500ms per cycle
- ❌ Increase daily API cost >$75

### 1.5 Functional Requirements

#### FR-1: Market Regime Classification
- **Input**: Symbol + indicators + news + portfolio + market context
- **Output**: `Choice{bullish, bearish, choppy, crisis}` with probabilities + confidence
- **Latency**: <300ms per symbol
- **Cache**: 300s TTL per regime (market-wide)

#### FR-2: Portfolio Risk Gate
- **Input**: Portfolio state + market context
- **Output**: `Noul(halt_new_buys)` + `Score(portfolio_stress: 0-3)`
- **Action**: Veto new entries when `halt_new_buys > 0.8`

#### FR-3: News Sentiment Synthesis
- **Input**: Recent headlines + symbol + recency
- **Output**: `Score(news_bullishness: 0-4)` with confidence
- **Replaces**: `compute_news_score()` keyword matching

#### FR-4: Trend Strength Assessment
- **Input**: Technical indicators + regime + multi-timeframe
- **Output**: `Score(trend_strength: 0-3)` with confidence
- **Augments**: ADX filter, trailing stop calibration

#### FR-5: Position Action Recommendation
- **Input**: Full analysis + position state + regime
- **Output**: `Choice{buy, add, hold, trim, exit}` with confidence
- **Gates**: Only act when confidence > 0.65

#### FR-6: RL Observation Extension
- **Input**: JEV outputs
- **Output**: 8-dim feature vector appended to PPO observation
- **Requires**: Full retrain

### 1.6 Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-1 | Regime accuracy >65% next-day direction | Backtest on 2022-2024 |
| AC-2 | Confidence calibrated (reliability diagram) | Brier score <0.25 |
| AC-3 | Drawdown reduction >20% in shadow | Phase 2 metrics |
| AC-4 | Win rate improvement >3% | Phase 3 metrics |
| AC-5 | API latency p95 <500ms | Load test |
| AC-6 | Zero critical failures in 2-week shadow | Phase 1 pass |

### 1.7 Constraints
| Constraint | Value |
|------------|-------|
| Max JEV calls/cycle | 10 (rate limit) |
| Max daily cost | $75 |
| Fallback behavior | Rule-based defaults |
| Rollback time | <30s (env var toggle) |
| Python version | 3.10+ |

---

## 2. ARCHITECT — Technical Design

### 2.1 Architecture Decision: Overlay Pattern
**Decision**: JEV as **overlay/gating layer** — never replaces core logic  
**Rationale**: Preserves working rule/RL systems; additive value; easy rollback  
**ADR**: `docs/decisions/0001-jev-overlay-pattern.md`

### 2.2 System Boundaries
```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  apex_dashboard.py (Flask + agent_loop + apply_cycle)           │
│  └── JEV Integration Module (apex_jev.py)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │  RULE ENGINE │ │  RL AGENT   │ │  DASHBOARD  │
   │  (existing)  │ │  (existing) │ │  (existing) │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          ▼               ▼               ▼
   Gate thresholds   Extend obs        Display
   Veto entries      Retrain PPO       Regime/confidence
   Scale position
```

### 2.3 Data Flow

#### Per-Cycle (Dashboard)
```
agent_loop()
  │
  ├─▶ _refresh_index_trend() ──▶ [JEV: regime + trend_strength + portfolio_stress + halt_new_buys]
  │                                    │
  │                                    ▼
  │                          apply_jev_gates(cfg, jev)  ──▶ Modified cfg for cycle
  │                                    │
  ├─▶ fetch_cycle_data() ──────────────┼──▶ For each symbol:
  │          │                          │
  │          ├─▶ RL mode: _get_rl_signal() [+ JEV features in obs]
  │          │
  │          └─▶ Rule mode: analyse() [+ JEV news_bullishness]
  │                                    │
  ▼                                    ▼
apply_cycle()
  │
  ├─▶ Pre-trade: if halt_new_buys > 0.8 → return
  │
  ├─▶ For each symbol:
  │     ├─▶ Run JEV: position_action
  │     ├─▶ Long entry: require action=="buy" AND conf>0.65
  │     ├─▶ Short entry: require action=="buy" (for short) AND conf>0.65
  │     ├─▶ Exit: if action=="exit" AND conf>0.65 → immediate
  │     ├─▶ Trim: if action=="trim" AND conf>0.7 → sell 50%
  │     └─▶ Add: if action=="add" AND conf>0.75 → pyramid
  │
  └─▶ Post-cycle: log JEV decisions for calibration
```

#### RL Training Pipeline
```
prepare_datasets()
  │
  ├─▶ For each ticker: build features (indicators + sentiment)
  │                          │
  │                          └─▶ [JEV: Add regime labels to each timestep]
  │
  ├─▶ TrainingEnv: observation_space extended +8 dims
  │
  ├─▶ ValidationSharpeEvalCallback: regime-stratified metrics
  │
  └─▶ run_training(): curriculum sampling (oversample crisis/bear)
```

### 2.4 Interface Contracts

#### `apex_jev.py` — Public API
```python
# Types
class JEVRegime(TypedDict):
    choice: Literal["bullish", "bearish", "choppy", "crisis"]
    probabilities: dict[str, float]
    confidence: float

class JEVScore(TypedDict):
    score: float  # 0 to max_level
    probabilities: dict[int, float]
    confidence: float
    legend: dict[int, str]

class JEVNoul(TypedDict):
    noul: float  # 0-1 probability of True
    confidence: float

class JEVAction(TypedDict):
    choice: Literal["buy", "add", "hold", "trim", "exit"]
    probabilities: dict[str, float]
    confidence: float

class JEVDecisions(TypedDict):
    regime: JEVRegime
    trend_strength: JEVScore
    news_bullishness: JEVScore
    portfolio_stress: JEVScore
    halt_new_buys: JEVNoul
    position_action: JEVAction

# Functions
def build_market_state(
    symbol: str,
    indicators: dict,
    news: list[dict],
    portfolio: dict,
    market: dict
) -> str:  # JSON string for JEV state
    ...

def get_jev_decisions(state_json: str) -> JEVDecisions:
    """Calls TypeSafe API, returns typed decisions."""
    ...

def apply_jev_gates(cfg: dict, jev: JEVDecisions) -> dict:
    """Returns modified config dict for this cycle."""
    ...

def should_halt(jev: JEVDecisions) -> bool:
    return jev["halt_new_buys"]["noul"] > cfg["jev_halt_noul_threshold"]

def get_position_action(jev: JEVDecisions) -> tuple[str, float]:
    """Returns (action, confidence) if confidence > threshold, else (None, 0)."""
    action = jev["position_action"]["choice"]
    conf = jev["position_action"]["confidence"]
    if conf >= cfg["jev_action_confidence_threshold"]:
        return action, conf
    return None, 0.0

def get_regime_scaling(jev: JEVDecisions) -> dict:
    """Returns risk/position scaling factors based on regime."""
    regime = jev["regime"]["choice"]
    conf = jev["regime"]["confidence"]
    if conf < cfg["jev_regime_confidence_threshold"]:
        return {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0}
    scaling = {
        "bullish":  {"risk_mult": 1.0, "pos_delta": +2, "sl_mult": 1.0},
        "bearish":  {"risk_mult": 0.8, "pos_delta": -2, "sl_mult": 1.2},
        "choppy":   {"risk_mult": 0.7, "pos_delta": -1, "sl_mult": 1.1},
        "crisis":   {"risk_mult": 0.5, "pos_delta": -4, "sl_mult": 1.5},
    }
    return scaling.get(regime, {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0})
```

#### `rl_signal.py` — Extended Observation
```python
# New feature columns (added to settings.feature_columns)
JEV_FEATURE_COLUMNS = [
    "jev_regime_bullish",
    "jev_regime_bearish", 
    "jev_regime_choppy",
    "jev_regime_crisis",
    "jev_trend_strength",
    "jev_news_bullishness",
    "jev_portfolio_stress",
    "jev_halt_noul",
]

# In _build_observation():
def _build_observation(ticker: str) -> tuple[np.ndarray, float, float] | None:
    # ... existing code ...
    
    # Fetch JEV decisions for this ticker
    jev = _get_jev_for_ticker(ticker)  # cached per cycle
    if jev:
        jev_features = np.array([
            jev["regime"]["probabilities"]["bullish"],
            jev["regime"]["probabilities"]["bearish"],
            jev["regime"]["probabilities"]["choppy"],
            jev["regime"]["probabilities"]["crisis"],
            jev["trend_strength"]["score"] / 3.0,
            jev["news_bullishness"]["score"] / 4.0,
            jev["portfolio_stress"]["score"] / 3.0,
            jev["halt_new_buys"]["noul"],
        ], dtype=np.float32)
        obs = np.concatenate([obs, jev_features])
    
    return obs, raw_atr, trend_5d_pct
```

### 2.5 Configuration Schema
```python
# In trading_agent/config.py Settings class
jev_enabled: bool = True
jev_regime_enabled: bool = True
jev_trend_enabled: bool = True
jev_news_enabled: bool = True
jev_risk_enabled: bool = True
jev_action_enabled: bool = True

jev_regime_confidence_threshold: float = 0.65
jev_action_confidence_threshold: float = 0.65
jev_trend_confidence_threshold: float = 0.70
jev_news_confidence_threshold: float = 0.70
jev_halt_noul_threshold: float = 0.80

jev_crisis_risk_multiplier: float = 0.5
jev_crisis_sl_multiplier: float = 1.5
jev_bullish_position_boost: int = 2
jev_bearish_position_cut: int = 2
jev_stress_risk_reduction: float = 0.7

jev_max_calls_per_cycle: int = 10
jev_cache_ttl_seconds: int = 300
```

### 2.6 Error Handling & Fallbacks
| Failure Mode | Detection | Fallback |
|--------------|-----------|----------|
| JEV API timeout | `requests.Timeout` | Use cached regime (300s); default others to neutral |
| JEV API 5xx | HTTP status >= 500 | Same as timeout |
| JEV invalid response | Schema validation fail | Log error; use rule-based defaults |
| Confidence below threshold | `conf < threshold` | Ignore JEV output; use rule-based |
| Rate limit exceeded | 429 response | Skip JEV for this cycle; use cache |

---

## 3. PLAN — Implementation Tasks

### 3.1 Task Breakdown

| Task ID | Phase | Title | Files | Est. Hours | Dependencies |
|---------|-------|-------|-------|------------|--------------|
| **T-01** | Setup | Create `apex_jev.py` with client, questions, state builder | New file | 4 | — |
| **T-02** | Setup | Add JEV config fields to `trading_agent/config.py` | `config.py` | 1 | T-01 |
| **T-03** | Setup | Add `TYPE_SAFE_API_KEY` to `.env.example` | `.env.example` | 0.5 | — |
| **T-04** | Phase 1 | Shadow mode: log JEV decisions in `agent_loop()` | `apex_dashboard.py` | 3 | T-01 |
| **T-05** | Phase 1 | Add JEV regime to `api_state()` response | `apex_dashboard.py` | 1 | T-04 |
| **T-06** | Phase 1 | Add JEV decisions to `_think_buffer` | `apex_dashboard.py` | 1 | T-04 |
| **T-07** | Phase 2 | Implement `apply_jev_gates()` — risk scaling | `apex_dashboard.py` | 3 | T-01 |
| **T-08** | Phase 2 | Pre-trade halt gate (`halt_new_buys`) | `apex_dashboard.py` | 2 | T-07 |
| **T-09** | Phase 2 | Graduated daily loss / drawdown limits | `apex_dashboard.py` | 2 | T-07 |
| **T-10** | Phase 2 | Dynamic max_positions by regime | `apex_dashboard.py` | 1 | T-07 |
| **T-11** | Phase 3 | Replace `compute_news_score()` with JEV news | `apex_dashboard.py` | 3 | T-01 |
| **T-12** | Phase 3 | Augment ADX filter with JEV trend_strength | `apex_dashboard.py` | 2 | T-01 |
| **T-13** | Phase 3 | Calibrate trailing stop with JEV trend | `apex_dashboard.py` | 2 | T-01 |
| **T-14** | Phase 3 | Open window adjustment by regime | `apex_dashboard.py` | 1 | T-01 |
| **T-15** | Phase 4 | Position action gate (buy/add/hold/trim/exit) | `apex_dashboard.py` | 4 | T-01 |
| **T-16** | Phase 4 | Pyramiding (`add`) and partial profit (`trim`) | `apex_dashboard.py` | 3 | T-15 |
| **T-17** | Phase 4 | Signal exit/cover override by JEV action | `apex_dashboard.py` | 2 | T-15 |
| **T-18** | Phase 5 | Extend RL observation space (+8 dims) | `rl_signal.py`, `config.py` | 2 | T-01 |
| **T-19** | Phase 5 | Regime-aware entropy/margin gating | `rl_signal.py` | 2 | T-18 |
| **T-20** | Phase 5 | JEV action → RL score mapping | `rl_signal.py` | 2 | T-18 |
| **T-21** | Phase 5 | Extend `bot_bridge.py` schema + response | `bot_bridge.py` | 2 | T-18 |
| **T-22** | Phase 5 | Regime-conditioned online learner | `online_learner.py` | 2 | T-18 |
| **T-23** | Phase 5 | Regime-stratified validation callback | `train.py` | 3 | T-18 |
| **T-24** | Phase 5 | Regime-attributed evaluation metrics | `evaluate.py` | 2 | T-18 |
| **T-25** | Phase 5 | Extended `TradingEnv` observation space | `trading_env.py` | 2 | T-18 |
| **T-26** | Phase 5 | Curriculum sampling by regime | `train.py` | 2 | T-23 |
| **T-27** | Phase 5 | Full PPO retrain with JEV features | `train.py` (run) | 8 GPU-hrs | T-18..T-26 |
| **T-28** | Verify | Unit tests for all gate functions | `tests/test_jev_*.py` | 4 | T-01..T-17 |
| **T-29** | Verify | Integration test: shadow mode calibration | `tests/test_shadow_calibration.py` | 3 | T-04..T-06 |
| **T-30** | Verify | Backtest: regime accuracy on 2022-2024 | Notebook | 4 | T-04 |
| **T-31** | Harden | Security: API key handling, rate limiting | `apex_jev.py` | 2 | T-01 |
| **T-32** | Harden | Observability: latency, error, cost metrics | `apex_dashboard.py` | 2 | T-04 |
| **T-33** | Review | Code review (security, quality, simplification) | All changed | 3 | T-01..T-27 |
| **T-34** | Prod | Feature flags per phase (env vars) | `config.py` | 1 | T-02 |
| **T-35** | Prod | Rollback runbook | `docs/operations/jev-rollback.md` | 1 | — |
| **T-36** | Ship | Phase 1 deploy (shadow) | — | 1 | T-28..T-32 |
| **T-37** | Ship | Phase 2 deploy (risk gates) | — | 1 | Phase 1 pass |
| **T-38** | Ship | Phase 3 deploy (signals) | — | 1 | Phase 2 pass |
| **T-39** | Ship | Phase 4 deploy (actions) | — | 1 | Phase 3 pass |
| **T-40** | Ship | Phase 5 deploy (RL) | — | 1 | Phase 4 pass |

### 3.2 Dependency Graph
```
T-01 → T-02, T-03
         ↓
T-04 → T-05, T-06
         ↓
T-07 → T-08, T-09, T-10
         ↓
T-11, T-12, T-13, T-14 (parallel)
         ↓
T-15 → T-16, T-17
         ↓
T-18 → T-19, T-20, T-21, T-22, T-23, T-24, T-25, T-26
         ↓
T-27 (retrain)
         ↓
T-28, T-29, T-30 (parallel)
         ↓
T-31, T-32, T-33 (parallel)
         ↓
T-34, T-35
         ↓
T-36 → T-37 → T-38 → T-39 → T-40 (sequential phases)
```

### 3.3 Milestones

| Milestone | Target | Criteria |
|-----------|--------|----------|
| **M1: Shadow Live** | Week 1 | T-01..T-06, T-28..T-32 complete; JEV logging for 5 days |
| **M2: Calibration Verified** | Week 2 | T-29, T-30 pass; regime accuracy >65%, Brier <0.25 |
| **M3: Risk Gates Live** | Week 3 | T-07..T-10, T-36→T-37; drawdown reduction confirmed |
| **M4: Signals Augmented** | Week 4 | T-11..T-14, T-38; win rate +3% confirmed |
| **M5: Actions Live** | Week 5 | T-15..T-17, T-39; false exits <10% |
| **M6: RL Retrained** | Week 7 | T-18..T-27 complete; RL+JEV Sharpe > RL-only +0.15 |
| **M7: Full Production** | Week 8 | T-40; all phases green |

### 3.4 Resource Requirements
| Resource | Quantity | Notes |
|----------|----------|-------|
| TypeSafe API | 1 key | Production tier |
| GPU (retrain) | 1× A100 40GB | ~8 hours for 500k steps |
| Daily API budget | $75 | Monitor via dashboard |
| Engineering | 1 FTE | 8 weeks |

---

## 4. IMPLEMENT — Implementation Guidelines

### 4.1 Code Standards
- **Type hints**: Full typing on all new functions
- **Error handling**: Explicit try/except with structured logging
- **No global state**: Pass config explicitly; use dependency injection
- **Async**: JEV calls in thread pool (non-blocking for dashboard)
- **Testing**: Unit test every gate function; integration test per phase

### 4.2 File Changes Summary

#### New Files
```
Apex/
├── apex_jev.py                    # JEV client + questions + state builder
├── apex_jev_integration.py        # Gate functions + helpers
└── tests/
    ├── test_jev_gates.py
    ├── test_jev_integration.py
    └── test_shadow_calibration.py
```

#### Modified Files
| File | Changes |
|------|---------|
| `apex_dashboard.py` | Import JEV; shadow logging; risk gates; signal augmentation; action gates |
| `trading_agent/config.py` | JEV config fields |
| `trading_agent/integration/rl_signal.py` | Extended observation; regime gating; JEV score mapping |
| `trading_agent/integration/bot_bridge.py` | Extended schema; response metadata |
| `trading_agent/integration/online_learner.py` | Regime conditioning; stress-weighted reward |
| `trading_agent/agent/train.py` | Regime-stratified validation; curriculum sampling |
| `trading_agent/agent/evaluate.py` | Regime-attributed metrics |
| `trading_agent/environment/trading_env.py` | Extended observation space; regime sampling |
| `trading_agent/data/indicator_engine.py` | JEV trend feature join |
| `trading_agent/data/sentiment_engine.py` | VADER replacement with JEV |

### 4.3 Incremental Implementation Rule
Each task:
1. Write failing test
2. Implement minimal change
3. Run targeted verification
4. Commit atomic change
5. No accumulated unverified diff

---

## 5. VERIFY — Verification Plan

### 5.1 Verification Levels

| Level | Scope | Command | Gate |
|-------|-------|---------|------|
| **Unit** | Individual gate functions | `pytest tests/test_jev_gates.py -v` | All pass |
| **Integration** | Shadow mode calibration | `pytest tests/test_shadow_calibration.py -v` | Regime acc >65% |
| **Backtest** | Historical regime accuracy | `python scripts/backtest_regime.py` | Brier <0.25 |
| **RL Eval** | Retrained model metrics | `python -m trading_agent.main --mode evaluate` | Sharpe +0.15 |
| **Load** | API latency under load | `locust -f tests/load_test.py` | p95 <500ms |
| **E2E** | Full cycle with JEV | `python apex_dashboard.py` (shadow) | No crashes 24h |

### 5.2 Calibration Metrics (Phase 1 Shadow)
```python
# tests/test_shadow_calibration.py
def test_regime_calibration():
    """Reliability diagram: predicted prob vs actual frequency."""
    # Bin predictions by regime probability
    # For each bin: actual frequency of next-day regime match
    # Brier score = mean((pred - actual)^2)
    assert brier_score < 0.25

def test_confidence_calibration():
    """High confidence → high accuracy."""
    high_conf = df[df['confidence'] > 0.8]
    assert high_conf['accuracy'] > 0.85

def test_regime_transitions():
    """Regime changes predict market turns."""
    # crisis→bearish should precede bottom
    # bullish→choppy should precede top
    assert transition_signal_quality > 0.6
```

### 5.3 Phase Gates

| Phase | Gate | Metric | Threshold |
|-------|------|--------|-----------|
| **Shadow** | Go/No-Go | Regime accuracy | >65% |
| | | Brier score | <0.25 |
| | | API error rate | <5% |
| **Risk Gates** | Go/No-Go | Max drawdown | ↓20% vs baseline |
| | | Annual return | >95% of baseline |
| | | Halt frequency | <30% of cycles |
| **Signals** | Go/No-Go | Win rate | ↑3% |
| | | False positive rate | ↓15% |
| | | Trade frequency | >80% baseline |
| **Actions** | Go/No-Go | False exit rate | <10% |
| | | Trim alpha | >0 (positive) |
| | | Pyramiding Sharpe | >1.0 |
| **RL** | Go/No-Go | RL+JEV Sharpe | > RL-only +0.15 |
| | | Training stability | No divergence |
| | | Inference latency | <100ms |

---

## 6. HARDEN — Security & Reliability

### 6.1 Security Checklist
- [ ] API key in env var only (never logged)
- [ ] Request/response sanitization (no PII in state)
- [ ] Rate limiting enforced (10 calls/cycle max)
- [ ] Circuit breaker on 3 consecutive failures
- [ ] Input validation on JEV response schema
- [ ] No secrets in `_think_buffer` or logs

### 6.2 Reliability Checklist
- [ ] Graceful degradation: JEV failure → rule defaults
- [ ] Cache regime (300s TTL) to survive API blips
- [ ] Idempotent gates: re-running cycle safe
- [ ] Observability: latency histogram, error counter, cost tracker
- [ ] Alerting: p95 latency >1s, error rate >5%, daily cost >$75

### 6.3 Performance Budget
| Metric | Budget | Measurement |
|--------|--------|-------------|
| JEV API latency (p50) | <200ms | Histogram |
| JEV API latency (p95) | <500ms | Histogram |
| Cycle overhead | <2s | Per-cycle timer |
| Memory overhead | <50MB | Process RSS |
| Daily API cost | <$75 | Cost tracker |

---

## 7. REVIEW — Code Review Checklist

### 7.1 Review Dimensions
| Dimension | Checklist |
|-----------|-----------|
| **Correctness** | Gate logic matches spec; edge cases handled (no position, no cash, etc.) |
| **Security** | No key leakage; input validation; rate limits |
| **Reliability** | Fallbacks work; circuit breaker; idempotency |
| **Maintainability** | Clear separation: JEV client vs integration vs gates |
| **Performance** | Caching used; async where needed; no N+1 calls |

### 7.2 Simplification Targets
- [ ] Remove duplicate regime logic between dashboard and RL
- [ ] Consolidate `apply_jev_gates()` into single config transform
- [ ] Eliminate `apex_jev_integration.py` if gates inline cleanly
- [ ] Single source of truth for JEV question schema

---

## 8. PRODUCTIONIZE — Deployment Checklist

### 8.1 Pre-Deploy (per phase)
- [ ] Tests pass (unit + integration + backtest)
- [ ] Lint/typecheck clean (`ruff`, `mypy`)
- [ ] Build passes (`docker build`)
- [ ] Feature flag OFF by default
- [ ] Rollback tested (toggle flag → verify rule behavior)

### 8.2 Deploy Sequence
```bash
# Phase 1: Shadow
export JEV_ENABLED=true
export JEV_REGIME_ENABLED=true
export JEV_TREND_ENABLED=true
export JEV_NEWS_ENABLED=true
export JEV_RISK_ENABLED=false
export JEV_ACTION_ENABLED=false
# Deploy → monitor 5 days

# Phase 2: Risk Gates
export JEV_RISK_ENABLED=true
# Deploy → monitor 5 days

# Phase 3: Signals
export JEV_NEWS_ENABLED=true  # already on
export JEV_TREND_ENABLED=true  # already on
# Deploy → monitor 5 days

# Phase 4: Actions
export JEV_ACTION_ENABLED=true
# Deploy → monitor 5 days

# Phase 5: RL
# Retrain PPO with JEV features
# Deploy new model + JEV features enabled
# A/B test: 50% traffic to RL+JEV
```

### 8.3 Monitoring Dashboard
| Panel | Query | Alert |
|-------|-------|-------|
| JEV Latency | `histogram_quantile(0.95, jev_latency_seconds)` | >500ms |
| JEV Errors | `rate(jev_errors_total[5m])` | >0.05/s |
| JEV Cost | `sum(increase(jev_cost_usd_total[1h]))` | >$75/day |
| Regime Distribution | `sum by (regime) (jev_regime_total)` | Shift >2σ |
| Confidence Calibration | Weekly reliability diagram job | Brier >0.25 |

---

## 9. SHIP — Release Criteria

### 9.1 Phase 1 (Shadow) — **READY TO SHIP** When:
- [ ] `apex_jev.py` implemented and tested
- [ ] Shadow logging active for 5 consecutive trading days
- [ ] Regime accuracy >65% on live data
- [ ] Brier score <0.25
- [ ] Zero critical errors

### 9.2 Phase 2 (Risk Gates) — **READY TO SHIP** When:
- [ ] Phase 1 criteria met
- [ ] Risk gates reduce drawdown >20% in paper
- [ ] No missed major moves (return >95% baseline)
- [ ] Halt trigger frequency <30%

### 9.3 Phase 3 (Signals) — **READY TO SHIP** When:
- [ ] Phase 2 criteria met
- [ ] Win rate +3% vs shadow baseline
- [ ] False positive rate -15%
- [ ] News sentiment correlation with outcomes >0.3

### 9.4 Phase 4 (Actions) — **READY TO SHIP** When:
- [ ] Phase 3 criteria met
- [ ] Trim/exit actions improve avg holding P&L
- [ ] Pyramiding (`add`) shows positive expectancy
- [ ] False exit rate <10%

### 9.5 Phase 5 (RL) — **READY TO SHIP** When:
- [ ] Phase 4 criteria met
- [ ] Retrained PPO+JEV Sharpe > PPO-only +0.15
- [ ] Regime-attributed metrics show improvement in crisis/bear
- [ ] A/B test (50/50) shows statistical significance (p<0.05)

---

## 10. REPORT — Final Deliverables

| Artifact | Location | Owner |
|----------|----------|-------|
| **Specification** | `docs/reasonix/specs/jev-integration-spec.md` | AI Assistant |
| **Audit** | `docs/audits/jev-integration-audit.md` | AI Assistant |
| **ADR** | `docs/decisions/0001-jev-overlay-pattern.md` | AI Assistant |
| **Implementation** | `apex_jev.py`, `apex_jev_integration.py` | Engineer |
| **Tests** | `tests/test_jev_*.py` | Engineer |
| **Rollback Runbook** | `docs/operations/jev-rollback.md` | Engineer |
| **Monitoring** | Grafana dashboards + alerts | Ops |
| **Calibration Report** | `docs/audits/jev-calibration-<date>.md` | Quant |
| **Phase Reports** | `docs/reasonix/plans/jev-phase-<n>-report.md` | PM |

---

## Appendix A: JEV Question Schema (Frozen for Implementation)

```json
{
  "model": "jev-latest",
  "state": "{symbol, price, indicators{}, news[], portfolio{}, market{}}",
  "questions": {
    "regime": {
      "type": "choice",
      "instructions": "What is the current market regime for this symbol?",
      "criteria": {
        "bullish": "Uptrend, higher highs/lows, positive breadth, VIX < 15",
        "bearish": "Downtrend, lower highs/lows, negative breadth, VIX > 25",
        "choppy": "Range-bound, mixed signals, ADX < 20, whipsaws",
        "crisis": "Sharp drawdown > 3% in 5d, VIX > 30, panic volume"
      }
    },
    "trend_strength": {
      "type": "score",
      "instructions": "How strong is the current trend?",
      "criteria": [
        "No trend (ADX < 15, EMAs flat/converging, volume declining)",
        "Weak trend (ADX 15-25, EMAs slightly separated, mixed volume)",
        "Strong trend (ADX 25-40, EMAs well separated, confirming volume)",
        "Explosive trend (ADX > 40, parabolic, extreme volume)"
      ]
    },
    "news_bullishness": {
      "type": "score",
      "instructions": "How bullish is recent news for this symbol?",
      "criteria": [
        "Very bearish (miss, guidance cut, fraud, bankruptcy risk)",
        "Bearish (negative sentiment, sector headwinds, downgrades)",
        "Neutral (mixed/no material news, routine filings)",
        "Bullish (beats, upgrades, positive catalysts, buybacks)",
        "Very bullish (transformational M&A, breakthrough, major deal)"
      ]
    },
    "portfolio_stress": {
      "type": "score",
      "instructions": "How stressed is the portfolio?",
      "criteria": [
        "Calm (DD < 2%, cash > 30%, winners > losers, low correlation)",
        "Cautious (DD 2-5%, cash 15-30%, some correlated losers)",
        "Stressed (DD 5-10%, cash < 15%, multiple losers, high correlation)",
        "Critical (DD > 10%, risk limits hit, forced selling risk)"
      ]
    },
    "halt_new_buys": {
      "type": "noul",
      "instructions": "Should new position entry be halted given current portfolio risk?"
    },
    "position_action": {
      "type": "choice",
      "instructions": "What action should be taken on this position?",
      "criteria": {
        "buy": "No position; strong setup; favorable R:R; regime supportive",
        "add": "Existing long; trend confirming; pullback to support; pyramid",
        "hold": "Thesis intact; no catalyst; wait for target/SL",
        "trim": "Extended move; take partial profits; reduce risk; trail rest",
        "exit": "Thesis broken; SL hit; regime change; news shock; EOD"
      }
    }
  }
}
```

---

## Appendix B: Environment Variables (Complete)

```bash
# .env.production
TYPE_SAFE_API_KEY=sk-...

# Feature Flags
JEV_ENABLED=true
JEV_REGIME_ENABLED=true
JEV_TREND_ENABLED=true
JEV_NEWS_ENABLED=true
JEV_RISK_ENABLED=true
JEV_ACTION_ENABLED=true

# Thresholds
JEV_REGIME_CONFIDENCE_THRESHOLD=0.65
JEV_ACTION_CONFIDENCE_THRESHOLD=0.65
JEV_TREND_CONFIDENCE_THRESHOLD=0.70
JEV_NEWS_CONFIDENCE_THRESHOLD=0.70
JEV_HALT_NOUl_THRESHOLD=0.80

# Risk Scaling
JEV_CRISIS_RISK_MULTIPLIER=0.5
JEV_CRISIS_SL_MULTIPLIER=1.5
JEV_BULLISH_POSITION_BOOST=2
JEV_BEARISH_POSITION_CUT=2
JEV_STRESS_RISK_REDUCTION=0.7

# Operations
JEV_MAX_CALLS_PER_CYCLE=10
JEV_CACHE_TTL_SECONDS=300
```

---

## Appendix C: Rollback Procedure

```bash
# Instant rollback (any phase)
export JEV_ENABLED=false
# Or selective:
export JEV_RISK_ENABLED=false
export JEV_ACTION_ENABLED=false

# Verify fallback behavior:
# 1. Dashboard starts without JEV import errors
# 2. agent_loop() runs with rule-based config only
# 3. RL inference works without JEV features (if model not retrained)
# 4. Dashboard UI shows "JEV: DISABLED" badge
```

---

**Specification Status**: **APPROVED FOR IMPLEMENTATION**  
**Next Action**: Begin Task T-01 (Create `apex_jev.py`)  
**Blockers**: TypeSafe API key provisioning