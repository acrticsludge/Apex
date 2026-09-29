# Spec: JEV State Expansion (v2)

## Objective
Give JEV (TypeSafe SystemOne) the full context it needs to make effective
`regime / trend_strength / news_bullishness / portfolio_stress / halt_new_buys / position_action`
choices in `apex_dashboard.py` + `apex_jev.py`. Today it decides half-blind
(hardcoded VIX/breadth, missing position context, stale drawdown, title-only news).

Success = every JEV choice is grounded in real portfolio + market + position
state, logged to the Decision Log, within the existing 10-calls/cycle budget.

## ASSUMPTIONS
1. JEV API (`https://api.typesafe.ai/v1/systemone`, `JEV_MODEL`) stays as-is.
2. `MAX_CALLS_PER_CYCLE=10`, `CACHE_TTL_SECONDS=300` stay unless spec says otherwise.
3. yfinance remains price/news source; Finnhub stays training-only (US-only free plan).
4. No new paid data vendor in this spec.

## Current gaps (evidence)
| # | Gap | Location |
|---|-----|----------|
| G1 | `vix: 20.0`, `breadth: 0.5` hardcoded | `apex_dashboard.py:1136-1140, 1868-1872` |
| G2 | `bb_pos`, `ema_9_21` never sent (default 0.5/neutral) | `apex_dashboard.py:1109-1116` vs `apex_jev.py:171-175` |
| G3 | No per-position context for `position_action` (entry, P&L%, SL/TP gap, holding time, min-to-close) | `apply_cycle` exit path uses system-level decisions only |
| G4 | `drawdown_pct` = `max_drawdown` (historical), `daily_pnl` ignores unrealised | `apex_dashboard.py:1129-1134, 1880-1885` |
| G5 | News = title + recency only, no publisher/type/summary | `apex_dashboard.py:1117-1125` |
| G6 | System call never asks `position_action` → `get_position_action()` returns hold/conf 0 | `apex_jev.py:388-393` + `_parse_choice(None)` default |
| G7 | Per-symbol `_call_jev_api` bypasses `MAX_CALLS_PER_CYCLE` (32 symbols = 32 calls) | `apex_dashboard.py:1148` |
| G8 | Per-symbol JEV result never `think_log`ged | `fetch_cycle_data` after line 1148 |

## Proposed state v2 (backward-compatible, additive only)

### Indicators (extend dict, keep old keys)
```python
indicators = {
    "price": live,
    "rsi": ..., "macd_hist": ..., "adx": ..., "atr": ...,
    "atr_pct": atr / price,              # NEW — normalises across ₹/$ names
    "vol_ratio": ...,
    "bb_pos": (price - bl) / max(bu - bl, 1e-9),  # NEW [0,1]
    "ema_gap_pct": (e9 - e21) / e21 * 100,        # NEW — replaces "neutral"
}
```

### Market context (real values, no hardcodes)
```python
market_context = {
    "spy_trend_pct": _index_trend[market],  # existing
    "vix": fetch_vix(market),               # NEW — ^VIX for US, ^INDIAVIX for NSE, cached 5m
    "breadth": calc_breadth(watchlist),     # NEW — % watchlist above day-open, [-1,1]
}
```

### Portfolio context (current, not historical)
```python
portfolio_ctx = {
    "cash": ..., "cash_pct": cash / portfolio_value,
    "drawdown_pct": current_dd_pct,    # (peak - pv)/peak, not max_drawdown
    "max_drawdown": ...,               # keep for stress history
    "open_positions": ..., "exposure_pct": invested / pv,
    "daily_pnl_pct": (realised + unrealised) / session_start_cash,
    "session_wr": wins / max(wins+losses, 1),
}
```

### Position context (NEW — only for held symbols, for `position_action`)
```python
position_ctx = {
    "side": pos["side"], "qty": pos["qty"],
    "entry": pos["entry"], "current": price,
    "pnl_pct": ..., "sl_dist_atr": (price - sl) / atr,
    "tp_dist_atr": (tp - price) / atr,
    "holding_min": (now - entered_at).min,
    "min_to_close": minutes_to_close(market_key),
    "eod_flag": mtc <= harvest_min,
}
```

### News (enrich, still top-5)
`{title, publisher, type, recency_h}` from `yf.Ticker.news`
(no summary — yfinance doesn't provide it).

## Rules
* System call per market per cycle: `regime, trend_strength, portfolio_stress, halt_new_buys`.
* Per-symbol call ONLY for: held positions (need `position_action`) + top-3 unscored
  by confidence. Cap total JEV calls/cycle at `MAX_CALLS_PER_CYCLE`.
* Per-symbol cache key `(symbol, minute-bar-ts)`, TTL 300s.
* Every JEV answer → `think_log("JEV", "<question>=<choice> conf=<.2f>", sym)`.
* Pure additive: old callers without new keys still work (defaults in `build_market_state`).

## Commands
```bash
py -3.13 -m py_compile apex_dashboard.py apex_jev.py
py -3.13 -m pytest tests/test_jev_gates.py -v
py -3.13 apex_dashboard.py
```

## Project Structure
```
apex_dashboard.py            → indicator/market/portfolio/position builders, call budgeting
apex_jev.py                  → build_market_state v2, fetch_vix/calc_breadth helpers
docs/reasonix/specs/         → this spec
tests/test_jev_gates.py      → gate unit tests
tests/test_jev_state_v2.py   → NEW state-builder tests
```

## Code Style
Small pure builders, no I/O inside state builders except cached fetchers:
```python
def build_position_context(pos: dict, price: float, atr: float, mtc: float | None) -> dict:
    ...
```

## Testing Strategy
* `pytest tests/test_jev_gates.py` — existing gates still pass.
* NEW `tests/test_jev_state_v2.py`: bb_pos in [0,1], ema gap sign, current-DD vs max-DD,
  position_ctx maths, VIX/breadth fallback when fetch fails, call-budget cap.
* Manual: start agent, filter Decision Log = JEV, confirm regime/trend/stress/halt +
  per-position action entries each cycle.

## Boundaries
* Always: additive fields only, redact API keys from logs, closed-bar values only.
* Ask first: new data vendor, changing JEV question text, raising MAX_CALLS.
* Never: commit `.env`, log `?token=`, trade on incomplete-bar indicators.

## Success Criteria
* [ ] No `vix: 20.0` / `breadth: 0.5` literals in dashboard call sites.
* [ ] `position_action` on held symbols returns real choice (not default hold/0).
* [ ] JEV calls/cycle ≤ 10 with 16+16 watchlist.
* [ ] JEV filter in Decision Log shows every JEV choice.
* [ ] Tests above pass.

## Open Questions
1. NSE VIX source: `^INDIAVIX` via yfinance reliable enough, or fallback 20?
2. Breadth: watchlist-only, or Nifty50/S&P500 advance-decline from index?
3. Include RL `(action, conf, entropy)` in JEV state for fusion, or keep separate?
