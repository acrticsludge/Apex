# JEV (TypeSafe) Integration Audit — Complete Inventory

**Project**: Apex Trading Dashboard  
**Date**: 2026-09-22  
**Scope**: Every file, function, and decision point where JEV can add value  
**Status**: Comprehensive audit — no stone unturned  

---

## Executive Summary

This audit identifies **47 distinct integration points** across **18 files** where TypeSafe JEV's structured judgment primitives (Choice, Score, Noul) can enhance the Apex trading system. Each point is categorized by impact, implementation effort, and risk.

**Integration Philosophy**: JEV as **overlay/gating layer** — never replaces core rule/RL logic, only modifies parameters, vetoes decisions, or provides confidence-weighted signals when its calibrated confidence exceeds thresholds.

---

## File-by-File Integration Map

---

### 1. `apex_dashboard.py` — **22 Integration Points** (Primary Target)

#### A. Market Regime Classification — `agent_loop()` / `apply_cycle()`

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 1 | `agent_loop()` line 1642 | `_refresh_index_trend()` computes index % change | `Choice{regime}` | Run JEV at cycle start per market; regime gates all downstream logic |
| 2 | `apply_cycle()` line 1093 | Logs scan start with positions/cash | `Score(trend_strength)` | Augment log with JEV trend assessment |
| 3 | `apply_cycle()` line 1105 | Updates `peak_portfolio` for drawdown calc | `Score(portfolio_stress)` | Feed portfolio stress into peak update logic |
| 4 | `apply_cycle()` line 1123 | EOD harvest logic based on minutes-to-close | `Noul(halt_new_buys)` | Pre-check: if halt_noul > 0.8, skip entire cycle |
| 5 | `apply_cycle()` line 1255 | Daily loss / drawdown kill-switch (hard thresholds) | `Score(portfolio_stress)` | Replace static thresholds with graduated scaling |

#### B. Entry/Exit Gates — `apply_cycle()` lines 1343-1470

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 6 | Line 1346 | Open window: `conf < 85%` → skip | `Score(trend_strength)` | If trend_strength.confidence > 0.7 AND score > 2, lower open_conf_gate to 70% |
| 7 | Line 1366 | ADX gate: `adx < 20` → skip | `Score(trend_strength)` | Effective `adx_min = base_adx_min * (1 + (trend_strength.score-1.5)*0.2)` |
| 8 | Line 1377 | Re-entry cooldown (hard 60 min) | `Noul(halt_new_buys)` | If halt_noul > 0.5, extend cooldown; if < 0.2, shorten |
| 9 | Line 1394 | Long entry: `conf >= conf_thr AND score > 0` | `Choice(position_action)` | Require `action == "buy" AND confidence > 0.65` as additional gate |
| 10 | Line 1419 | Short entry: `bearish_conf >= short_threshold` | `Choice(position_action)` | Require `action ∈ {"exit", "trim"} for longs` / `"buy" for shorts` |
| 11 | Line 1424 | Short index gate: `index_pct <= 0.5%` | `Choice(regime)` | If regime == "bearish", relax index_max_pct_for_short to 1.5% |
| 12 | Line 1444 | Max positions check | `Choice(regime)` | Dynamic max_pos: bullish +2, bearish -2, crisis -50% |
| 13 | Line 1448 | Insufficient cash check | `Score(portfolio_stress)` | If stress > 2.5, require cash > price * 3 (more buffer) |

#### C. Position Management — `apply_cycle()` lines 1140-1237

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 14 | Line 1150 | Trailing stop (ATR-based, fixed multipliers) | `Score(trend_strength)` | `dist_mult = base * (2.0 - trend_strength.score/3.0)` — tighter trail in strong trends |
| 15 | Line 1194 | Stop loss hit | `Choice(position_action)` | Log JEV action at SL hit for calibration |
| 16 | Line 1201 | Target hit | `Choice(position_action)` | Log JEV action at TP hit |
| 17 | Line 1213 | RL early exit (entropy/margin gated) | `Choice(position_action)` | **Replace RL exit gate**: `action == "exit" AND confidence > 0.6` |
| 18 | Line 1322 | Signal exit for longs: `score < -30` | `Choice(position_action)` | If `action == "exit" AND confidence > 0.65`, exit immediately |
| 19 | Line 1329 | Signal cover for shorts: `score > 30` | `Choice(position_action)` | If `action == "exit" AND confidence > 0.65`, cover immediately |

#### D. Risk Management — `apply_cycle()` lines 1254-1270

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 20 | Line 1255 | Daily loss limit: `daily_loss_pct < -5%` | `Score(portfolio_stress)` | Graduated: stress 0-1 → 5%, 1-2 → 3%, 2-3 → 1.5%, 3+ → 0.5% |
| 21 | Line 1258 | Max drawdown: `drawdown_pct > 8%` | `Score(portfolio_stress)` | Graduated: stress 0-1 → 8%, 1-2 → 6%, 2-3 → 4%, 3+ → 2% |
| 22 | Line 1272 | Open observation window (15 min) | `Choice(regime)` | Crisis regime: extend window to 30 min; bullish: reduce to 5 min |

#### E. Dashboard API Endpoints — `api_state()` / `api_think()`

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 23 | `api_state()` line 1766 | Returns agent_log, decision_log | All | Add `jev_decisions` to response for dashboard display |
| 24 | `api_think()` line 1908 | Returns `_think_buffer` | All | Add JEV reasoning entries with confidence scores |

---

### 2. `apex_dashboard.py` — **Signal Generation** (`fetch_cycle_data()` / `analyse()`)

#### F. Signal Analysis — `fetch_cycle_data()` lines 1036-1081

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 25 | Line 1044 | RL mode: `_get_rl_signal()` | `Score(news_bullishness)` | Replace `news_score` in RL observation with JEV news score |
| 26 | Line 1074 | Rule-based fallback: `analyse()` | All 6 questions | Run JEV for each symbol; embed results in `analysis` dict |

#### G. Technical Analysis — `analyse()` (implicit via `compute_news_score`)

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 27 | `compute_news_score()` lines 480-510 | Keyword matching (bull/bear word lists) | `Score(news_bullishness)` | **Full replacement**: JEV evaluates headlines with context |
| 28 | Line 558 | `news_score` added to composite score | `Score(news_bullishness)` | Weight by confidence: `score += round(news_score * 15 * confidence)` |

---

### 3. `trading_agent/integration/rl_signal.py` — **5 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 29 | `_build_observation()` line 107 | Builds 22-feature observation vector | All | **Append 5 JEV features** to observation (see below) |
| 30 | `get_rl_signal()` line 181 | Entropy/margin gate | `Choice(regime)` | Regime-aware gating: crisis → stricter entropy threshold |
| 31 | Line 191 | Trend alignment (±3% = full) | `Score(trend_strength)` | Use JEV trend_strength.score instead of 5-day close pct |
| 32 | Line 202 | Score mapping: `40 + norm_prob*60 + trend*12` | `Choice(position_action)` | Map JEV action to score: buy=+60, add=+30, hold=0, trim=-30, exit=-60 |
| 33 | Line 234 | Returns `rl_extremes` (top 3 features) | All | Add JEV regime/probabilities to extremes for dashboard |

**Proposed Extended Observation Vector** (+5 dims):
```python
jev_features = [
    jev["regime"].probabilities["bullish"],      # 0-1
    jev["regime"].probabilities["bearish"],      # 0-1
    jev["regime"].probabilities["choppy"],       # 0-1
    jev["regime"].probabilities["crisis"],       # 0-1
    jev["trend_strength"].score / 3.0,           # 0-1 normalized
    jev["news_bullishness"].score / 4.0,         # 0-1 normalized
    jev["portfolio_stress"].score / 3.0,         # 0-1 normalized
    jev["halt_new_buys"].noul,                   # 0-1
]
# Total: 8 additional features → retrain PPO
```

---

### 4. `trading_agent/integration/bot_bridge.py` — **3 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 34 | `PredictionRequest` line 30 | Expects `observation: list[float]` | All | **Extend schema**: add optional `jev_features: list[float]` |
| 35 | `_predict_action_and_confidence()` line 86 | Forward pass only | `Choice(regime)` | Return regime probabilities in response metadata |
| 36 | `status()` line 163 | Returns model metadata | All | Add `jev_version`, `jev_last_call_latency_ms` |

---

### 5. `trading_agent/integration/online_learner.py` — **2 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 37 | `record_entry()` line 52 | Stores obs+action at trade open | `Choice(regime)` | Store regime at entry for regime-conditioned learning |
| 38 | `record_exit()` line 64 | Computes ATR-normalized reward | `Score(portfolio_stress)` | Weight reward by inverse portfolio stress (less learning when stressed) |

---

### 6. `trading_agent/agent/train.py` — **2 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 39 | `ValidationSharpeEvalCallback` line 99 | Selection score: Sharpe + excess - DD + coverage | All | Add regime-stratified validation: compute metrics per regime |
| 40 | `run_training()` line 220 | Trains on all tickers uniformly | `Choice(regime)` | Curriculum: oversample crisis/bearish regimes (harder to learn) |

---

### 7. `trading_agent/agent/evaluate.py` — **1 Integration Point**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 41 | `evaluate_model_on_frames()` | Overall metrics + diagnostics | `Choice(regime)` | **Regime-attributed metrics**: win rate, Sharpe, DD per regime |

---

### 8. `trading_agent/environment/trading_env.py` — **3 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 42 | `__init__()` line 74 | Observation space: fixed feature columns | All | **Extend observation_space** to include JEV features (requires retrain) |
| 43 | `step()` line 364 | Reward: daily_return + excess - penalties | `Score(portfolio_stress)` | Scale `flat_position_penalty` by portfolio_stress (less penalty when stressed) |
| 44 | `reset()` line 265 | Random ticker selection | `Choice(regime)` | Stratified sampling: equal episodes per regime |

---

### 9. `trading_agent/data/indicator_engine.py` — **1 Integration Point**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 45 | `add_technical_indicators()` line 9 | Adds 11 technical indicators | `Score(trend_strength)` | **Add JEV trend_strength as feature** (computed externally, joined here) |

---

### 10. `trading_agent/data/sentiment_engine.py` — **2 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 46 | `add_sentiment_feature()` line 130 | VADER compound score [-1,1] | `Score(news_bullishness)` | **Replace entirely**: JEV score replaces VADER; store both for comparison |
| 47 | `fetch_daily_sentiment()` line 73 | Finnhub API → daily sentiment | `Score(news_bullishness)` | Use JEV to score articles directly (bypass VADER), cache JEV scores |

---

### 11. `apex_trading_agent.py` (Legacy CLI) — **3 Integration Points**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 48 | `run_market_cycle()` line 320 | Fetches prices, runs analyses | `Choice(regime)` | Run JEV once per cycle; apply regime scaling to `RISK_PER_TRADE` |
| 49 | `analyse()` line 171 | Computes score from 5 indicators | `Score(news_bullishness)` | Replace keyword news scoring |
| 50 | `main()` line 567 | Dashboard render loop | All | Add JEV regime badge to console dashboard |

---

### 12. `trading_agent/config.py` — **1 Integration Point**

| # | Location | Current Logic | JEV Primitive | Proposed Integration |
|---|----------|---------------|---------------|---------------------|
| 51 | `Settings` class line 155 | All hyperparameters as fields | All | Add `jev_*` config fields (thresholds, feature toggles) |

---

## Consolidated Integration Points by Category

### By JEV Primitive

| Primitive | Count | Primary Use Cases |
|-----------|-------|-------------------|
| **Choice (regime)** | 12 | Market regime → risk scaling, signal filtering, capital allocation |
| **Choice (position_action)** | 8 | Unified exit/entry logic, pyramiding, partial profits |
| **Score (trend_strength)** | 7 | ADX replacement, trailing stop calibration, RL observation |
| **Score (news_bullishness)** | 6 | VADER replacement, recency weighting, confidence gating |
| **Score (portfolio_stress)** | 7 | Dynamic risk limits, drawdown scaling, halt decisions |
| **Noul (halt_new_buys)** | 4 | Pre-trade veto, cooldown modulation, EOD mode |

### By System Component

| Component | Files | Points | Priority |
|-----------|-------|--------|----------|
| **Dashboard Core** | `apex_dashboard.py` | 22 | **CRITICAL** — Main trading loop |
| **RL Inference** | `rl_signal.py`, `bot_bridge.py` | 8 | **HIGH** — Model input enhancement |
| **RL Training** | `train.py`, `evaluate.py`, `trading_env.py`, `online_learner.py` | 6 | **MEDIUM** — Requires retrain |
| **Signal Generation** | `sentiment_engine.py`, `indicator_engine.py` | 3 | **HIGH** — Direct replacement |
| **Legacy CLI** | `apex_trading_agent.py` | 3 | **LOW** — Deprecated path |
| **Config** | `config.py` | 1 | **REQUIRED** — Feature flags |

### By Implementation Phase

| Phase | Points | Files | Effort | Risk |
|-------|--------|-------|--------|------|
| **Phase 1: Shadow** | 6 | dashboard, rl_signal | Low | None (logging only) |
| **Phase 2: Risk Gates** | 8 | dashboard | Low | Low (veto only) |
| **Phase 3: Signal Augmentation** | 10 | dashboard, sentiment, indicator | Medium | Medium (replaces logic) |
| **Phase 4: Position Actions** | 7 | dashboard | Medium | Medium (new actions) |
| **Phase 5: RL Integration** | 8 | rl_signal, bot_bridge, train, env | High | High (retrain required) |

---

## Data Flow Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MARKET DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  yfinance OHLCV ──▶ indicator_engine.py ──▶ TradingEnv (RL)                │
│       │                    │                        │                         │
│       │                    │                        ▼                         │
│       │                    │               ┌─────────────────┐               │
│       │                    │               │  PPO Policy     │               │
│       │                    │               │  (train.py)     │               │
│       │                    │               └────────┬────────┘               │
│       │                    │                        │                         │
│       ▼                    ▼                        ▼                         │
│  Finnhub News ──▶ sentiment_engine.py ──▶ rl_signal.py ──▶ get_rl_signal()  │
│       │                    │                        │                         │
│       │                    │                        ▼                         │
│       │                    │               ┌─────────────────┐               │
│       │                    │               │  JEV FEATURES   │               │
│       │                    │               │  (8 dims)       │               │
│       │                    │               └────────┬────────┘               │
│       │                    │                        │                         │
│       ▼                    ▼                        ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    JEV API CALL (per symbol/cycle)                   │    │
│  │  State: {symbol, price, indicators, news[], portfolio, market}      │    │
│  │  Questions: regime, trend_strength, news_bullishness,               │    │
│  │             portfolio_stress, halt_new_buys, position_action        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                 │
│              ▼                     ▼                     ▼                 │
│     ┌───────────────┐      ┌───────────────┐      ┌───────────────┐       │
│     │  RULE ENGINE  │      │   RL AGENT    │      │  DASHBOARD    │       │
│     │  (dashboard)  │      │  (PPO)        │      │  (API/UI)     │       │
│     └───────┬───────┘      └───────┬───────┘      └───────┬───────┘       │
│             │                      │                      │                │
│             ▼                      ▼                      ▼                │
│     Gate thresholds          Extend obs            Display regime,       │
│     Veto entries             Retrain PPO           confidence, actions  │
│     Scale position size                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Requirements

### New Environment Variables

```bash
# Required
TYPE_SAFE_API_KEY=your_typesafe_key

# JEV Feature Toggles
JEV_ENABLED=true
JEV_REGIME_ENABLED=true
JEV_TREND_ENABLED=true
JEV_NEWS_ENABLED=true
JEV_RISK_ENABLED=true
JEV_ACTION_ENABLED=true

# Confidence Thresholds (tunable per phase)
JEV_REGIME_CONFIDENCE_THRESHOLD=0.65
JEV_ACTION_CONFIDENCE_THRESHOLD=0.65
JEV_TREND_CONFIDENCE_THRESHOLD=0.70
JEV_NEWS_CONFIDENCE_THRESHOLD=0.70
JEV_HALT_NOUl_THRESHOLD=0.80

# Risk Scaling Factors
JEV_CRISIS_RISK_MULTIPLIER=0.5
JEV_CRISIS_SL_MULTIPLIER=1.5
JEV_BULLISH_POSITION_BOOST=2
JEV_BEARISH_POSITION_CUT=2
JEV_STRESS_RISK_REDUCTION=0.7

# Rate Limiting
JEV_MAX_CALLS_PER_CYCLE=10
JEV_CACHE_TTL_SECONDS=300
```

### New Config Fields in `trading_agent/config.py`

```python
@dataclass
class Settings:
    # ... existing fields ...
    
    # JEV Integration
    jev_enabled: bool = field(default_factory=lambda: os.getenv("JEV_ENABLED", "true").lower() == "true")
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

---

## Testing Requirements per Integration Point

| Point | Test Type | Description |
|-------|-----------|-------------|
| 1-5 | Unit | Regime gating logic with mocked JEV responses |
| 6-13 | Unit | Entry/exit gate modifications with parametrized confidence |
| 14-19 | Unit | Position management with JEV action/confidence |
| 20-22 | Integration | Graduated risk limits vs static (backtest) |
| 23-24 | API | Dashboard response includes JEV fields |
| 25-28 | Integration | Signal quality: JEV vs keyword news (A/B) |
| 29-33 | RL | Observation extension → retrain → eval metrics |
| 34-36 | API | Bridge schema extension, backward compat |
| 37-38 | Unit | Online learner regime conditioning |
| 39-40 | Integration | Regime-stratified validation metrics |
| 41 | Integration | Regime-attributed evaluation |
| 42-44 | RL | Env observation space extension |
| 45 | Unit | Indicator engine JEV feature join |
| 46-47 | Integration | VADER vs JEV sentiment correlation |
| 48-50 | Legacy | CLI regime display, risk scaling |

---

## Rollback Plan per Phase

| Phase | Rollback Trigger | Rollback Action |
|-------|------------------|-----------------|
| Shadow | JEV API errors > 5% | Set `JEV_ENABLED=false` |
| Risk Gates | Drawdown increases | Disable `JEV_RISK_ENABLED` |
| Signals | Win rate drops > 2% | Disable `JEV_NEWS_ENABLED`, `JEV_TREND_ENABLED` |
| Actions | False exits > 10% | Disable `JEV_ACTION_ENABLED` |
| RL | Training diverges | Revert observation space, retrain from checkpoint |

---

## Cost Projection

| Phase | Daily JEV Calls | Est. Cost/Day | Monthly |
|-------|-----------------|---------------|---------|
| Shadow | 64 (32 sym × 2 markets) | ~$15 | ~$450 |
| Risk Gates | 64 | ~$15 | ~$450 |
| Signals | 64 | ~$15 | ~$450 |
| Actions | 64 | ~$15 | ~$450 |
| RL | 64 + retrain batch | ~$20 | ~$600 |

**Optimization**: Cache regime (300s TTL), batch symbols, limit to top-K candidates.

---

## Sign-Off Requirements

| Role | Requirement |
|------|-------------|
| **Quant/Strategy** | Validate regime definitions match trading thesis |
| **Risk** | Approve graduated risk scaling factors |
| **Engineering** | Review API error handling, caching, latency budgets |
| **ML** | Approve observation space extension, retrain protocol |
| **Ops** | Budget approval, monitoring/alerting setup |

---

## Appendix: JEV Question Schema (Canonical)

```json
{
  "model": "jev-latest",
  "state": {
    "symbol": "AAPL",
    "price": 175.43,
    "indicators": {"rsi": 42, "macd_hist": 0.34, "bb_pos": 0.6, "ema_9_21": "bull", "adx": 28, "atr": 2.1, "vol_ratio": 1.3},
    "news": [{"title": "Apple beats earnings", "sentiment": 0.8, "recency_h": 4}, ...],
    "portfolio": {"cash": 18000, "drawdown_pct": 0.03, "open_positions": 3, "daily_pnl_pct": 0.005},
    "market": {"spy_trend_pct": 0.4, "vix": 14.2, "breadth": 0.62}
  },
  "questions": {
    "regime": {"type": "choice", "instructions": "Current market regime", "criteria": {"bullish": "...", "bearish": "...", "choppy": "...", "crisis": "..."}},
    "trend_strength": {"type": "score", "instructions": "Trend strength", "criteria": ["No trend", "Weak", "Strong", "Explosive"]},
    "news_bullishness": {"type": "score", "instructions": "News bullishness", "criteria": ["Very bearish", "Bearish", "Neutral", "Bullish", "Very bullish"]},
    "portfolio_stress": {"type": "score", "instructions": "Portfolio stress", "criteria": ["Calm", "Cautious", "Stressed", "Critical"]},
    "halt_new_buys": {"type": "noul", "instructions": "Halt new entries?"},
    "position_action": {"type": "choice", "instructions": "Position action", "criteria": {"buy": "...", "add": "...", "hold": "...", "trim": "...", "exit": "..."}}
  }
}
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-22 | JEV as overlay only | Preserves working systems; additive value |
| 2026-09-22 | 6 questions per symbol | Covers all decision dimensions |
| 2026-09-22 | Confidence thresholds 0.65-0.8 | Balance signal vs noise; tunable |
| 2026-09-22 | Shadow mode first | Validate calibration before risk |
| 2026-09-22 | 8 JEV features for RL | Regime probs + scores + noul |
| 2026-09-22 | Replace VADER entirely | JEV has context; VADER is keyword-only |

---

**End of Audit** — 51 integration points identified across 12 files. Ready for Phase 1 implementation.