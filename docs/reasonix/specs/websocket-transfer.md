# Spec: Transfer to Fully Websocket

## Objective
Replace polling with push: live prices into the agent and live state into the
browser over websockets, cutting the 5s REST fan-out (32 symbols/cycle) and the
30s browser poll. Target: tick-driven updates, lower yfinance/NSE ban risk.

## ASSUMPTIONS
1. Paper trading only; no order-routing WS needed.
2. Free-tier data acceptable where noted; paid SIP only if user opts in.
3. gunicorn single-worker deploy stays (constrains WS server choice).
4. Fallback to current polling must always work (WS is additive first).

## What "free" actually covers (verified 2026-09-28)
| Source | Free WS | Catch |
|--------|---------|-------|
| Finnhub WS | 50 symbols, US-only | NSE (`.NS`) needs paid All-In-One (~$3500/mo); same 403 as REST |
| Alpaca WS Basic (paper incl.) | IEX feed, 30 symbols, 1 conn | US-only, IEX not full SIP; full SIP = $99/mo Plus; our 16 US fit in 30 |
| yfinance | No websocket exists | stays REST |
| NSE (unofficial API) | No free WS | stays `curl_cffi` polling |
| Browser ← server | SSE (current) / SocketIO — free, just code | gunicorn needs eventlet/gevent worker for SocketIO |

Conclusion: 100% websocket is NOT free. Free covers US leg (Alpaca IEX) +
browser push. NSE leg stays polling.

## Proposed architecture (phased)
```
Phase A (free, no new keys):  browser SSE stays; NSE+US polling stays.
Phase B (free, 2 new keys):   US 16 → Alpaca WS `wss://stream.data.alpaca.markets/v2/iex`
                              (trades/bars), NSE 16 → polling as today.
Phase C (optional):           browser SSE → SocketIO rooms (prices/state/think);
                              needs worker change + `flask-socketio` dep.
```

## New env (Phase B only)
```
ALPACA_KEY_ID=
ALPACA_SECRET_KEY=
ALPACA_FEED=iex        # iex (free) | sip (paid Plus)
```

## Rules
* WS thread writes `_latest_prices` under existing `_price_lock`; agent loop
  keeps snapshotting — no change to strategy locking.
* Stale-tick kill: symbol without tick for 60s → mark stale, fall back to REST once.
* Reconnect with backoff (5s→60s), resubscribe on reconnect; log to Decision Log (RISK).
* After-hours: WS test stream `v2/test` for connectivity checks.
* Secrets via env only; never log keys or full WS URLs with creds.

## Commands
```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 -m py_compile apex_dashboard.py
py -3.13 apex_dashboard.py
```

## Project Structure
```
apex_dashboard.py        → ws client thread, fallback router, (Phase C) SocketIO emit
requirements.txt         → + websocket-client (B) / flask-socketio (C)
.env.example             → + ALPACA_* keys
tests/test_ws_feed.py    → NEW: reconnect, stale fallback, lock discipline
```

## Code Style
One small client module pattern inside dashboard (matches `_price_updater` style):
thread + queue + `_price_lock`, no strategy logic in the socket callback.

## Testing Strategy
* NEW `tests/test_ws_feed.py`: auth-fail fallback, reconnect resubscribe, stale-symbol
  REST fallback, no-lock-deadlock (timeout acquire in tests).
* Manual: kill network 30s → reconnect + resubscribe visible in logs; prices resume.
* Soak: one full US session, compare WS ticks vs REST spot-checks.

## Boundaries
* Always: fallback to polling works with WS keys blank; redact secrets in logs.
* Ask first: adding `flask-socketio`/worker change (C), paid SIP feed.
* Never: block agent loop on WS recv; commit keys; drop NSE polling until proven.

## Success Criteria
* [ ] US symbols tick via WS with keys set; blank keys → polling, zero errors.
* [ ] NSE unchanged and green throughout.
* [ ] Stale (>60s) symbol auto-falls-back, recovers without restart.
* [ ] Browser still live (SSE) during WS outage.

## Open Questions
1. Create Alpaca paper account for `ALPACA_KEY_ID/SECRET`? (Blocks Phase B)
2. Is IEX-only acceptable for signals, or is SIP ($99/mo) required?
3. Phase C SocketIO, or is current SSE good enough for browser?
