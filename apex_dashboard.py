"""
Apex Dashboard — Local Web Management Interface
================================================
Run:   pip install flask yfinance pandas numpy
       python apex_dashboard.py
Open:  http://localhost:7000

Controls everything end-to-end:
  • Start / Stop / Pause the trading agent
  • Live positions with manual Sell buttons
  • Signal board for both markets
  • Settings panel (risk, SL, target, intervals)
  • Full trade log
"""

import threading
import json
import os
import sys
from math import isfinite as _isfinite

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── RL mode: agent drives all buy/sell decisions; user only sets capital ──────
RL_MODE = os.getenv("RL_MODE", "true").lower() == "true"

# ── Deployment namespace — set APEX_ENV per Railway deployment so each has
#    its own Supabase rows and never stomps the other deployment's state. ──────
_APEX_ENV      = os.getenv("APEX_ENV", "main")
_STATE_ID      = f"{_APEX_ENV}-singleton"
_CONFIG_ID     = f"{_APEX_ENV}-config"

try:
    from trading_agent.integration.rl_signal import get_rl_signal as _get_rl_signal
    _RL_AVAILABLE = True
except Exception as _rl_import_err:  # noqa: BLE001
    _RL_AVAILABLE = False
    _get_rl_signal = None  # type: ignore[assignment]

# ── JEV (TypeSafe) integration ──────────────────────────────────────────────────
def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


_JEV_ENABLED = _env_flag("JEV_ENABLED")

# apex_jev must never import this module (it did, and that back-edge raised
# ImportError under the production import order). configure() pushes the flags
# in instead, so a failure here is a genuine subsystem fault, not a cycle.
try:
    import apex_jev as _jev
    _JEV_AVAILABLE = True
except Exception as _jev_import_err:  # noqa: BLE001
    _JEV_AVAILABLE = False
    _jev = None  # type: ignore[assignment]
    # Surfaced in /api/jev/status: a silently disabled risk layer is worse than
    # an absent one, so the reason is never swallowed.
    _JEV_IMPORT_ERROR = repr(_jev_import_err)
else:
    _JEV_IMPORT_ERROR = None

import time
import warnings
import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from flask import (
    Flask, jsonify, request, render_template, session, redirect, Response,
    stream_with_context,
)
from werkzeug.exceptions import HTTPException
import yfinance as yf
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

# ─── SUPABASE CLIENT ──────────────────────────────────────────────────────────
try:
    from supabase import create_client as _sb_create
    _SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    _SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
    _sb = _sb_create(_SUPABASE_URL, _SUPABASE_KEY) if (_SUPABASE_URL and _SUPABASE_KEY) else None
except ImportError:
    _sb = None

# ─── CONFIG (mutable at runtime via /api/config) ─────────────────────────────

cfg = {
    "india_capital":        int(os.getenv("INDIA_CAPITAL", "180000")),
    "us_capital":           int(os.getenv("US_CAPITAL", "18000")),
    "rl_mode":              RL_MODE,
    "india_max_positions":       16,
    "us_max_positions":          16,
    "risk_per_trade":         0.02,   # % of session_start_cash risked per trade (ATR-normalised)
    "max_position_pct":       0.20,   # hard cap: no single position > 20% of cash
    "confidence_threshold":     62,
    "stop_loss_pct":          0.03,   # fallback fixed SL (used when ATR unavailable)
    "target_pct":            0.045,   # fallback fixed TP
    "check_interval_min":        3,
    "idle_interval_min":        15,
    "eod_harvest_min":          30,   # sell profitable positions this many min before close
    "eod_exit_min":             15,   # force-exit everything this many min before close
    "state_file":  "apex_dual_state.json",
    # ── Risk controls ────────────────────────────────────────────────────────
    "daily_loss_limit_pct":   0.05,   # halt new buys if day's realised loss > 5% of start cash
    "max_drawdown_pct":       0.08,   # halt new buys if portfolio drops > 8% from peak
    # ── ATR exits ────────────────────────────────────────────────────────────
    "use_atr_exits":          True,
    "atr_sl_mult":             2.0,   # SL = entry - ATR × 2.0
    "atr_tp_mult":             3.0,   # TP = entry + ATR × 3.0
    # ── Trailing stop ────────────────────────────────────────────────────────
    "trailing_stop_enabled":  True,
    "trailing_activation_mult": 1.0,  # activate once price rises by 1× ATR from entry
    "trailing_dist_mult":      1.5,   # trail at 1.5× ATR below the running high
    # ── RL exit ──────────────────────────────────────────────────────────────
    "rl_exit_confidence":       55,   # min RL confidence to trigger early exit
    # ── Signal filters ───────────────────────────────────────────────────────
    "adx_min":                  20,   # only trade when ADX confirms a trend
    "index_min_pct":          -0.5,   # block new longs if index is down > 0.5% on the day
    "open_filter_min":          15,   # observation window after market open (minutes)
    "open_filter_confidence":   85,   # allow trades during window only if confidence >= 85%
    # ── Trade hygiene ────────────────────────────────────────────────────────
    "cooldown_after_sl_min":    60,   # re-entry blocked for 60 min after a stop-loss
    "commission_pct":        0.0006,  # 0.06% per side (buy + sell)
    # Slippage: fills do not happen at the quote. 5bps per side, so a round trip
    # costs ~10bps. Without this, paper P&L is optimistic and the position cap,
    # daily loss limit and drawdown kill-switch are all calibrated against a
    # number the market will not reproduce. Editable in the settings panel.
    "slippage_pct":          0.0005,
    # ── Short selling ─────────────────────────────────────────────────────────
    "short_selling_enabled":       True,
    "short_confidence_threshold":    75,  # bearish confidence needed to short (0–100)
    "index_max_pct_for_short":      0.5,  # block shorts if index UP more than +0.5%
    "settings_enabled":            True,  # False = full liberty: bypass threshold/ADX/index/cap filters
    # ── JEV (TypeSafe) feature flags ──────────────────────────────────────────
    # These were previously read as cfg.get("jev_*", True), but no such key ever
    # existed, so every JEV feature was hardwired on and impossible to disable.
    "jev_enabled":            _JEV_ENABLED,
    "jev_risk_enabled":       _JEV_ENABLED,
    "jev_action_enabled":     _JEV_ENABLED,
    "jev_trend_enabled":      _JEV_ENABLED,
    "jev_news_enabled":       _JEV_ENABLED,
    "jev_regime_enabled":     _JEV_ENABLED,
}

# Hand the flag set to apex_jev now that cfg exists. Done here rather than at the
# import site so the import check can stay early without needing cfg.
if _JEV_AVAILABLE:
    _jev.configure(cfg)

from apex_universe import INDIA_WATCHLIST, US_WATCHLIST  # canonical, shared with trading_agent
from apex_market import EDT, IST  # canonical calendar; see MARKET HOURS below
from apex_config import (  # ledger/config guards live in a leaf module
    CONFIG_BOUNDS,
    POSITION_EDIT_SPEC,
    STATE_EDIT_SPEC,
    clean_numeric,
    validate_config_payload,
)

# Time zones come from apex_market (imported above). They must not be redefined
# here: a frozen -4 offset for US Eastern made the app believe the close was
# 21:00 UTC year round, firing EOD exits an hour late for ~5 months a year.

# ─── AGENT LOGGER ─────────────────────────────────────────────────────────────

_log_buffer: deque = deque(maxlen=500)

class _BufHandler(logging.Handler):
    def emit(self, record):
        _log_buffer.append({
            "ts":    datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        })

apex_log = logging.getLogger("apex")
apex_log.setLevel(logging.DEBUG)
if not apex_log.handlers:
    # Rotating, not a plain FileHandler: the price updater ticks every 5 s and the
    # cycle logs per position, which is ~17k lines/day growing without bound on a
    # Railway volume.
    _fh = RotatingFileHandler(
        "apex.log", maxBytes=int(os.getenv("LOG_MAX_BYTES", 5_000_000)),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", 3)), encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    apex_log.addHandler(_fh)
    apex_log.addHandler(_BufHandler())

if not _JEV_AVAILABLE:
    apex_log.error(
        "JEV subsystem unavailable — every JEV risk gate is DISABLED. Cause: %s", _JEV_IMPORT_ERROR
    )
else:
    apex_log.info("JEV subsystem loaded — risk gates active")

# ─── DECISION / THINKING LOG ──────────────────────────────────────────────────
# Separate from apex_log — plain-language narration of every bot decision.
# Categories: CYCLE · SCAN · FILTER · ENTRY · EXIT · RISK · INDEX

_think_buffer: deque = deque(maxlen=400)

def think_log(cat: str, msg: str, sym: str = "") -> None:
    """Append a reasoning entry. Thread-safe: deque.append is atomic in CPython."""
    _think_buffer.append({
        "ts":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cat": cat,
        "sym": sym,
        "msg": msg,
    })

# ─── SHARED STATE ─────────────────────────────────────────────────────────────

_lock    = threading.Lock()
_state   = {}                       # trading state (loaded from Supabase on startup)
_signals = {                        # latest scan results
    "india": [], "us": [],
    "india_prices": {}, "us_prices": {},
    "jev_decisions": {},            # per-market JEV decisions, read by rl_signal
}
_agent   = {
    "running": False, "paused": False,
    "status":  "idle", "last_update": None, "next_check": None,
    "india_open": False, "us_open": False,
}

# ── Per-cycle JEV risk-gate overrides ─────────────────────────────────────────
# apply_jev_gates() is a pure function: it returns a modified copy of the base
# config. Its return value used to be discarded, so no gate was ever applied.
# Gating writes here instead, per market, and is cleared at the top of every
# cycle — writing into `cfg` would compound the multipliers every cycle until
# risk_per_trade decayed to zero.
_jev_gates: dict = {}

JEV_GATED_KEYS = (
    "risk_per_trade",
    "stop_loss_pct",
    "india_max_positions",
    "us_max_positions",
    "daily_loss_limit_pct",
    "max_drawdown_pct",
    "open_filter_min",
)


def effective_cfg(market_key: str) -> dict:
    """Base config overlaid with this cycle's JEV gates for one market."""
    overrides = _jev_gates.get(market_key)
    if not overrides:
        return cfg
    merged = dict(cfg)
    merged.update(overrides)
    return merged


def apply_jev_cycle_gates(jev_decisions_per_market: dict) -> None:
    """Fold each market's JEV decisions into the per-cycle gate overrides."""
    _jev_gates.clear()
    if not _JEV_AVAILABLE:
        return
    for market_key, decisions in (jev_decisions_per_market or {}).items():
        if not decisions:
            continue
        try:
            gated = _jev.apply_jev_gates(cfg, decisions)
        except Exception as e:  # noqa: BLE001
            apex_log.warning("[JEV] gate application failed for %s: %s", market_key, e)
            continue
        overrides = {k: gated[k] for k in JEV_GATED_KEYS if k in gated and gated[k] != cfg.get(k)}
        if overrides:
            _jev_gates[market_key] = overrides
            apex_log.info(
                "[JEV] %s gates applied: %s",
                market_key.upper(),
                ", ".join(f"{k}={v}" for k, v in overrides.items()),
            )


def publish_jev_decisions(jev_decisions_per_market: dict) -> None:
    """Expose decisions to the RL path, which reads _signals['jev_decisions'].

    Nothing wrote this key before, so jev_trend_strength was always None,
    jev_regime always 'bullish' and jev_score_delta always 0 inside rl_signal.
    """
    with _lock:
        _signals["jev_decisions"] = dict(jev_decisions_per_market or {})

# ── Shared price store (Thread 1 writes, Thread 2 snapshots) ──────────────────
_latest_prices: dict = {}
# Wall-clock of the last successful quote per symbol. Without this a symbol
# whose feed dies kept trading on its last good price indefinitely, because a
# failed fetch and "no quote" looked identical.
_price_ts: dict = {}
# RLock, not Lock: _record_prices() and price_is_fresh() take the lock themselves,
# so a caller already holding it would deadlock the price thread and every
# request thread. Re-entrancy is the safer contract here.
_price_lock = threading.RLock()

# A quote older than this is not trusted for entries, exits or SL/TP checks.
PRICE_MAX_AGE_SECONDS = int(os.getenv("PRICE_MAX_AGE_SECONDS", "120"))


def _record_prices(fresh: dict) -> None:
    """Merge fresh quotes into the shared store, stamping their arrival time.

    Caller must not hold _price_lock.
    """
    if not fresh:
        return
    now = time.time()
    with _price_lock:
        _latest_prices.update(fresh)
        for sym in fresh:
            _price_ts[sym] = now


def price_is_fresh(sym: str, max_age: float | None = None) -> bool:
    """True when we hold a quote for sym that is recent enough to trade on."""
    limit = PRICE_MAX_AGE_SECONDS if max_age is None else max_age
    with _price_lock:
        if sym not in _latest_prices:
            return False
        ts = _price_ts.get(sym)
    return ts is not None and (time.time() - ts) <= limit


def stale_price_symbols(max_age: float | None = None) -> list:
    """Symbols we hold a quote for that are now too old to act on."""
    limit = PRICE_MAX_AGE_SECONDS if max_age is None else max_age
    now = time.time()
    with _price_lock:
        return [s for s in _latest_prices if (now - _price_ts.get(s, 0.0)) > limit]

# ─── MARKET HOURS ─────────────────────────────────────────────────────────────
# The calendar lives in apex_market (a leaf module, unit-tested in isolation).
# Re-exported here so every existing call site keeps working unchanged.
from apex_market import (  # noqa: E402
    EDT,
    IST,
    is_india_open,
    is_us_open,
    minutes_since_open,
    minutes_to_close,
    session_date,
)

# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_data(symbol: str):
    try:
        df = yf.download(symbol, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.rename(columns=str.lower).dropna()
    except Exception:
        return None

def get_price(symbol: str):
    try:
        return float(yf.Ticker(symbol).fast_info.last_price)
    except Exception:
        df = fetch_data(symbol)
        return float(df["close"].iloc[-1]) if df is not None else None

def fetch_prices(symbols: list) -> dict:
    try:
        data = yf.download(" ".join(symbols), period="1d", interval="1m",
                           progress=False, auto_adjust=True, group_by="ticker")
        result = {}
        for sym in symbols:
            try:
                if sym in data.columns.get_level_values(0):
                    result[sym] = float(data[sym]["Close"].dropna().iloc[-1])
                else:
                    result[sym] = float(data["Close"].dropna().iloc[-1])
            except Exception as e:
                apex_log.debug("[PRICE] batch extract failed for %s: %s", sym, e)
        return result
    except Exception:
        return {}

# ─── THREAD 1: PRICE UPDATER (every 10 s) ────────────────────────────────────

_nse_session      = None
_nse_session_ts   = 0.0
_NSE_SESSION_TTL  = 300  # refresh cookies every 5 min

# SSE subscribers — each is a Queue that gets fresh price dicts pushed into it
_price_subscribers: list = []
_price_sub_lock = threading.Lock()

def _get_nse_session():
    """Return a curl_cffi session with fresh NSE cookies."""
    global _nse_session, _nse_session_ts
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    now = time.time()
    if _nse_session is None or (now - _nse_session_ts) > _NSE_SESSION_TTL:
        try:
            s = curl_requests.Session(impersonate="chrome120")
            s.get("https://www.nseindia.com", timeout=10)
            _nse_session    = s
            _nse_session_ts = now
        except Exception as e:
            apex_log.debug(f"[NSE] session init failed: {e}")
            _nse_session = None
    return _nse_session

def _fetch_nse_price(sym: str) -> tuple[str, float | None]:
    """Fetch live NSE price; returns (yf_symbol, price_or_None)."""
    nse_sym = sym.replace(".NS", "").replace(".BO", "")
    s = _get_nse_session()
    if s is not None:
        try:
            r = s.get(
                f"https://www.nseindia.com/api/quote-equity?symbol={nse_sym}",
                timeout=8,
            )
            if r.status_code == 200 and r.text:
                p = r.json().get("priceInfo", {}).get("lastPrice")
                if p:
                    return sym, float(p)
        except Exception:
            global _nse_session
            _nse_session = None
    # fallback
    try:
        fb = yf.Ticker(sym).fast_info.last_price
        if fb and fb > 0:
            return sym, float(fb)
    except Exception as e:
        apex_log.debug("[PRICE] yfinance fallback failed for %s: %s", sym, e)
    return sym, None

def _fetch_us_price(sym: str) -> tuple[str, float | None]:
    try:
        p = yf.Ticker(sym).fast_info.last_price
        if p and p > 0:
            return sym, float(p)
    except Exception as e:
        apex_log.debug("[PRICE] yfinance fetch failed for %s: %s", sym, e)
    return sym, None

def _price_updater():
    """Fetches all symbols in parallel every 5 s, pushes updates via SSE."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="price")
    while True:
        futures = (
            [pool.submit(_fetch_nse_price, s) for s in INDIA_WATCHLIST] +
            [pool.submit(_fetch_us_price,  s) for s in US_WATCHLIST]
        )
        fresh = {}
        for f in as_completed(futures):
            sym, price = f.result()
            if price:
                fresh[sym] = price
        if fresh:
            _record_prices(fresh)
            # push to SSE subscribers
            with _price_sub_lock:
                dead = []
                for q in _price_subscribers:
                    try:
                        q.put_nowait(dict(fresh))
                    except Exception:
                        dead.append(q)
                for q in dead:
                    _price_subscribers.remove(q)
            apex_log.info(
                f"[PRICE] tick — {len(fresh)}/{len(INDIA_WATCHLIST + US_WATCHLIST)} symbols refreshed"
            )
            aged = stale_price_symbols()
            if aged:
                apex_log.warning(
                    f"[PRICE] {len(aged)} symbols are holding quotes older than "
                    f"{PRICE_MAX_AGE_SECONDS}s and will not be traded: {', '.join(sorted(aged)[:8])}"
                )
        time.sleep(5)

# ─── INDEX TREND FILTER ───────────────────────────────────────────────────────

_index_trend: dict = {"india": 0.0, "us": 0.0}   # intraday % change from day open

def _refresh_index_trend():
    """Fetch today's intraday % change for Nifty50 (^NSEI) and S&P500 (^GSPC).
    Called once per agent cycle — no lock needed (floats are atomic in CPython)."""
    for sym, key in [("^NSEI", "india"), ("^GSPC", "us")]:
        try:
            df = yf.download(sym, period="1d", interval="1m",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                first = float(close.dropna().iloc[0])
                last  = float(close.dropna().iloc[-1])
                if first > 0:
                    _index_trend[key] = round((last - first) / first * 100, 3)
        except Exception as e:
            apex_log.debug(f"[INDEX] {key} refresh failed: {e}")
    think_log("INDEX",
              f"India (^NSEI) {_index_trend['india']:+.2f}%   "
              f"US (^GSPC) {_index_trend['us']:+.2f}%")

# ─── INDICATORS ───────────────────────────────────────────────────────────────

def calc_rsi(close, p=14) -> float:
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return float((100 - 100 / (1 + g / l)).iloc[-1])

def calc_macd(close):
    e12  = close.ewm(span=12, adjust=False).mean()
    e26  = close.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    return float(line.iloc[-1]), float(sig.iloc[-1]), float((line - sig).iloc[-1])

def calc_bb(close, p=20):
    m = close.rolling(p).mean()
    s = close.rolling(p).std()
    return float((m + 2*s).iloc[-1]), float(m.iloc[-1]), float((m - 2*s).iloc[-1])

def calc_ema(close, p) -> float:
    return float(close.ewm(span=p, adjust=False).mean().iloc[-1])

def calc_vol_ratio(volume, p=20) -> float:
    avg = volume.rolling(p).mean().iloc[-1]
    return float(volume.iloc[-1] / avg) if avg > 0 else 1.0

def calc_atr(df, p=14) -> float:
    """Average True Range on OHLCV dataframe. Uses high/low/close columns."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(p).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 0.0

def calc_adx(df, p=14) -> float:
    """Average Directional Index. 0–100; values > 20 indicate a tradeable trend."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index
    )
    minus_dm  = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index
    )
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14    = tr.rolling(p).mean()
    plus_di  = 100 * plus_dm.rolling(p).mean()  / atr14
    minus_di = 100 * minus_dm.rolling(p).mean() / atr14
    denom    = (plus_di + minus_di).replace(0, np.nan)
    dx       = (100 * (plus_di - minus_di).abs() / denom).fillna(0)
    adx      = dx.rolling(p).mean()
    val      = adx.iloc[-1]
    return float(val) if not np.isnan(val) else 0.0

def compute_hist_bias(df) -> tuple:
    """
    Count how many of the last 4 completed trading days closed above their open.
    Returns (bull_days: int, total_days: int, bias_score: float [-1.0, +1.0]).
    Returns (0, 0, 0.0) when fewer than 2 completed days are available.
    """
    try:
        idx   = df.index
        dates = idx.normalize() if hasattr(idx, "normalize") else pd.DatetimeIndex(idx).normalize()
        unique_days = sorted(dates.unique())
        if len(unique_days) >= 1:
            unique_days = unique_days[:-1]          # drop today (may be incomplete)
        completed_days = unique_days[-4:]           # last 4 completed days
        total_days     = len(completed_days)
        if total_days < 2:
            return (0, 0, 0.0)
        bull_days = 0
        for day in completed_days:
            day_bars = df[dates == day]
            if len(day_bars) < 2:
                continue
            if float(day_bars["close"].iloc[-1]) > float(day_bars["open"].iloc[0]):
                bull_days += 1
        bias_score = round((bull_days / total_days) * 2 - 1, 4)
        return (bull_days, total_days, bias_score)
    except Exception:
        return (0, 0, 0.0)

# ─── NEWS SENTIMENT ───────────────────────────────────────────────────────────

_NEWS_CACHE: dict = {}   # {symbol: (fetched_at, news_score, article_count)}
_NEWS_TTL   = 900        # re-fetch every 15 minutes

_BULL_KWORDS = {
    "beat", "beats", "record", "growth", "profit", "surge", "soars", "rally",
    "upgrade", "outperform", "strong", "positive", "revenue", "bullish",
    "raises", "raised", "exceeds", "wins", "deal", "partnership", "approve",
    "approved", "launches", "expands", "boost", "jump", "jumps", "rises",
    "rise", "high", "buy", "acquisition", "dividend", "buyback",
}

_BEAR_KWORDS = {
    "miss", "misses", "loss", "losses", "decline", "falls", "drops", "tumbles",
    "downgrade", "underperform", "weak", "negative", "below", "disappoints",
    "cuts", "cut", "warn", "warning", "bearish", "lawsuit", "investigation",
    "recall", "fraud", "default", "halt", "suspend", "probe", "fine",
    "penalty", "resign", "bankruptcy", "layoff", "layoffs", "selloff",
}

def compute_news_score(symbol: str) -> tuple:
    """
    Fetch recent yfinance headlines for symbol and score them via keyword matching.
    Returns (news_score: float [-1.0, +1.0], article_count: int).
    Scores are cached for 15 minutes per symbol to avoid repeated fetches.
    Returns (0.0, 0) on any error or when no articles are found.
    """
    now = time.time()
    cached = _NEWS_CACHE.get(symbol)
    if cached and (now - cached[0]) < _NEWS_TTL:
        return (cached[1], cached[2])
    try:
        news   = yf.Ticker(symbol).news or []
        recent = [n for n in news
                  if now - float(n.get("providerPublishTime", 0)) < 86400]
        if not recent:
            _NEWS_CACHE[symbol] = (now, 0.0, 0)
            return (0.0, 0)
        bull = bear = 0
        for item in recent:
            words = set((item.get("title") or "").lower().split())
            bull += len(words & _BULL_KWORDS)
            bear += len(words & _BEAR_KWORDS)
        total = bull + bear
        raw   = round((bull - bear) / total, 4) if total > 0 else 0.0
        score = max(-1.0, min(1.0, raw))
        _NEWS_CACHE[symbol] = (now, score, len(recent))
        return (score, len(recent))
    except Exception:
        _NEWS_CACHE[symbol] = (now, 0.0, 0)
        return (0.0, 0)

# ─── SIGNAL ENGINE ────────────────────────────────────────────────────────────

def analyse(symbol: str, df, live_price: float, jev_decisions: dict | None = None) -> dict:
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)
    price  = live_price or float(close.iloc[-1])
    sigs   = {}
    score  = 0

    r = calc_rsi(close)
    sigs["RSI"] = {"value": f"{r:.1f}",
                   "signal": ("OVERSOLD BUY" if r < 30 else
                               "OVERBOUGHT SELL" if r > 70 else
                               "Mildly bullish" if r < 50 else "Mildly bearish")}
    score += 25 if r < 30 else -25 if r > 70 else 8 if r < 50 else -8

    bu, bm, bl = calc_bb(close)
    sigs["BB"] = {"value": f"U:{bu:.0f} L:{bl:.0f}",
                  "signal": ("Below band BUY" if price < bl else
                              "Above band SELL" if price > bu else "Inside band HOLD")}
    score += 20 if price < bl else -20 if price > bu else 0

    ml, ms, mh = calc_macd(close)
    sigs["MACD"] = {"value": f"H:{mh:.2f}",
                    "signal": ("Bullish BUY"  if mh > 0 and ml > ms else
                                "Bearish SELL" if mh < 0 and ml < ms else "Neutral HOLD")}
    score += 20 if (mh > 0 and ml > ms) else -20 if (mh < 0 and ml < ms) else 0

    e9, e21 = calc_ema(close, 9), calc_ema(close, 21)
    sigs["EMA"] = {"value": f"9:{e9:.0f} 21:{e21:.0f}",
                   "signal": "Bull BUY" if e9 > e21 else "Bear SELL"}
    score += 20 if e9 > e21 else -20

    vr = calc_vol_ratio(volume)
    sigs["Vol"] = {"value": f"{vr:.2f}x",
                   "signal": ("High confirms" if vr > 1.5 else
                               "Low volume"   if vr < 0.7 else "Normal")}
    score += 15 if vr > 1.5 else -5 if vr < 0.7 else 0

    atr = calc_atr(df)
    adx = calc_adx(df)

    hist_win, hist_total, hist_bias = compute_hist_bias(df)
    if hist_total >= 2:
        score += round(hist_bias * 10)

    # Task 17-18: JEV news_bullishness replaces compute_news_score
    if jev_decisions and "news_bullishness" in jev_decisions:
        news = jev_decisions["news_bullishness"]
        news_score = (news["score"] - 2.0) / 2.0  # Normalize from 0-4 to -1 to +1
        news_conf = news["confidence"]
        weight = _jev.get_news_weight(jev_decisions) if _JEV_AVAILABLE else 1.0
        if news_conf > 0 and news_score != 0:
            score += round(news_score * 15 * weight)
    else:
        # Fallback to keyword-based scoring
        news_score, news_count = compute_news_score(symbol)
        if news_count > 0:
            score += round(news_score * 15)
    
    score = max(-100, min(100, score))

    return {
        "symbol":          symbol,
        "price":           price,
        "score":           score,
        "confidence":      round(min(100, max(0, (score + 100) / 2)), 1),
        "signals":         sigs,
        "rsi":             r,
        "bb_upper":        bu,
        "bb_lower":        bl,
        "atr":             round(atr, 4),
        "adx":             round(adx, 1),
        "hist_win_days":   hist_win,
        "hist_total_days": hist_total,
        "hist_bias":       hist_bias,
        "news_score":      news_score if 'news_score' in locals() else 0.0,
        "news_count":      news_count if 'news_count' in locals() else 0,
    }

# ─── STATE MANAGEMENT ─────────────────────────────────────────────────────────

def _empty_mstate(capital: float, session_date: str = None) -> dict:
    return {
        "cash":               float(capital),
        "positions":          {},
        "realised_pnl":       0.0,
        "wins":               0,
        "losses":             0,
        "peak_portfolio":     float(capital),
        "max_drawdown":       0.0,
        "trade_log":          [],
        "session_date":       session_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "session_start_cash": float(capital),
        "trading_halted":     False,
        "cooldown_until":     {},
    }

def _normalize_state(st: dict) -> dict:
    """Backfill keys that were added after a state was first persisted."""
    st.setdefault("sessions", [])
    for mkt in ("india", "us"):
        if mkt not in st:
            continue
        if "session_date" not in st[mkt]:
            tz = IST if mkt == "india" else EDT
            st[mkt]["session_date"] = datetime.now(tz).strftime("%Y-%m-%d")
        if "session_start_cash" not in st[mkt]:
            cap = cfg["india_capital"] if mkt == "india" else cfg["us_capital"]
            st[mkt]["session_start_cash"] = float(cap)
        st[mkt].setdefault("trading_halted", False)
        st[mkt].setdefault("cooldown_until", {})
        # ── Carry-over: if cash was reset to env-var capital, restore from last session ──
        env_cap = float(cfg["india_capital"] if mkt == "india" else cfg["us_capital"])
        if abs(st[mkt].get("cash", env_cap) - env_cap) < 0.01 and not st[mkt].get("positions"):
            mkt_sessions = [s for s in st.get("sessions", []) if s.get("market") == mkt]
            if mkt_sessions:
                last_end = sorted(mkt_sessions, key=lambda s: s.get("date", ""))[-1].get("end_cash")
                if last_end and float(last_end) > 0 and abs(float(last_end) - env_cap) > 0.01:
                    st[mkt]["cash"] = float(last_end)
                    st[mkt]["session_start_cash"] = float(last_end)
        # ── Migrate positions to short-selling schema ─────────────────────────
        for pos in st[mkt].get("positions", {}).values():
            pos.setdefault("side", "long")           # all pre-existing positions are longs
            pos.setdefault("running_high", pos.get("entry", 0))  # safety (already present)
            if pos.get("side") == "short":
                pos.setdefault("running_low", pos.get("entry", 0))

    return st

def load_state() -> dict:
    if _sb:
        try:
            resp = _sb.table("apex_state").select("data").eq("id", _STATE_ID).execute()
            if resp.data:
                apex_log.info("State loaded from Supabase")
                return _normalize_state(resp.data[0]["data"])
        except Exception as e:
            apex_log.warning(f"Supabase load error (falling back to JSON): {e}")
    # JSON fallback (local dev / Supabase unavailable)
    sf = cfg["state_file"]
    if os.path.exists(sf):
        try:
            with open(sf) as f:
                apex_log.info(f"State loaded from {sf}")
                return _normalize_state(json.load(f))
        except Exception as e:
            apex_log.warning(f"JSON load error: {e}")
    apex_log.info("Starting with fresh state")
    return {
        "india":      _empty_mstate(cfg["india_capital"], datetime.now(IST).strftime("%Y-%m-%d")),
        "us":         _empty_mstate(cfg["us_capital"],    datetime.now(EDT).strftime("%Y-%m-%d")),
        "sessions":   [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

def save_state(st: dict):
    if _sb:
        try:
            _sb.table("apex_state").upsert({
                "id":         _STATE_ID,
                "data":       json.loads(json.dumps(st, default=str)),
                "updated_at": datetime.now().isoformat(),
            }).execute()
            return
        except Exception as e:
            apex_log.error(f"Supabase save error (falling back to JSON): {e}")
    # JSON fallback
    try:
        with open(cfg["state_file"], "w") as f:
            json.dump(st, f, indent=2, default=str)
    except Exception as e:
        apex_log.error(f"JSON save error: {e}")

def _settings_active() -> bool:
    return cfg.get("settings_enabled", True)


def snapshot_state() -> dict:
    """Deep-copy the trading state for a writer thread.

    Callers hold _lock. The copy is what gets handed to the blocking Supabase
    upsert, so the I/O never runs under the lock and never observes a torn
    mutation.
    """
    return json.loads(json.dumps(_state, default=str))


def persist_state(snap: dict) -> None:
    """Write a snapshot to Supabase/JSON. MUST be called with _lock released."""
    save_state(snap)

# ── Keys the user can change via the Settings panel ───────────────────────────
_CFG_PERSIST_KEYS = (
    "risk_per_trade", "confidence_threshold", "stop_loss_pct", "target_pct",
    "check_interval_min", "idle_interval_min",
    "india_max_positions", "us_max_positions",
    "eod_harvest_min", "eod_exit_min",
    "rl_exit_confidence",
    "settings_enabled",
)

def load_cfg():
    """Restore user-adjusted settings from Supabase (id='config' row)."""
    if not _sb:
        return
    try:
        resp = _sb.table("apex_state").select("data").eq("id", _CONFIG_ID).execute()
        if resp.data:
            saved = resp.data[0]["data"]
            for k in _CFG_PERSIST_KEYS:
                if k in saved:
                    cfg[k] = saved[k]
            apex_log.info(f"Config restored from Supabase: {[k for k in _CFG_PERSIST_KEYS if k in saved]}")
    except Exception as e:
        apex_log.warning(f"Config load error: {e}")

def save_cfg():
    """Persist user-adjustable settings to Supabase (id='config' row)."""
    if not _sb:
        return
    try:
        _sb.table("apex_state").upsert({
            "id":         _CONFIG_ID,
            "data":       {k: cfg[k] for k in _CFG_PERSIST_KEYS},
            "updated_at": datetime.now().isoformat(),
        }).execute()
    except Exception as e:
        apex_log.error(f"Config save error: {e}")

def _save_rl_decision(symbol: str, market: str, sig: dict):
    if not _sb:
        return
    try:
        probs = sig.get("rl_probs", [0.0, 0.0, 0.0])
        _sb.table("apex_rl_decisions").insert({
            "symbol":    symbol,
            "market":    market,
            "action":    int(sig.get("rl_action", 0)),
            "prob_hold": float(probs[0]),
            "prob_buy":  float(probs[1]),
            "prob_sell": float(probs[2]),
            "confidence": float(sig.get("confidence", 0)),
            "price":     float(sig.get("price", 0)),
            "reasoning": {
                "entropy":  sig.get("rl_entropy"),
                "margin":   sig.get("rl_margin"),
                "extremes": sig.get("rl_extremes", []),
            },
        }).execute()
    except Exception as e:
        apex_log.warning(f"RL decision log failed: {e}")

# ─── SESSION MANAGEMENT ───────────────────────────────────────────────────────

def _close_session(market_key: str, prices: dict):
    """Archive the current session and reset market state for the next day.
    Must be called with _lock held. prices should be the latest price snapshot."""
    tz      = IST if market_key == "india" else EDT
    today   = datetime.now(tz).strftime("%Y-%m-%d")
    mstate  = _state[market_key]
    capital = cfg[f"{market_key}_capital"]
    date    = mstate.get("session_date", today)
    sym     = "₹" if market_key == "india" else "$"

    # Force-close any remaining open positions (safety net — EOD should have cleared them)
    for s in list(mstate["positions"].keys()):
        price = prices.get(s) or mstate["positions"][s]["entry"]
        if mstate["positions"][s].get("side", "long") == "long":
            execute_sell(s, price, "SESSION END", mstate)
        else:
            execute_cover(s, price, "SESSION END", mstate)

    start_c = mstate.get("session_start_cash", float(capital))
    net_pnl = mstate["realised_pnl"]
    n_trades = len(mstate["trade_log"])

    record = {
        "id":                  f"{market_key}_{date}",
        "market":              market_key,
        "date":                date,
        "start_cash":          round(start_c, 2),
        "end_cash":            round(mstate["cash"], 2),
        "end_portfolio":       round(portfolio_value(mstate, prices), 2),
        "net_pnl":             round(net_pnl, 2),
        "pnl_pct":             round(net_pnl / start_c * 100, 2) if start_c > 0 else 0.0,
        "wins":                mstate["wins"],
        "losses":              mstate["losses"],
        "n_trades":            n_trades,
        "trades":              list(mstate["trade_log"]),
        "archived_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    sessions = _state.setdefault("sessions", [])
    # Upsert: replace if same id (handles re-archiving same day)
    idx = next((i for i, s in enumerate(sessions) if s["id"] == record["id"]), None)
    if idx is not None:
        sessions[idx] = record
    else:
        sessions.append(record)
    if len(sessions) > 365:
        _state["sessions"] = sessions[-365:]

    tot = record["wins"] + record["losses"]
    wr  = f"{record['wins']/tot*100:.0f}%" if tot else "—"
    apex_log.info(
        f"[SESSION] {market_key.upper()} {date} archived — "
        f"P&L={sym}{net_pnl:+.0f} ({record['pnl_pct']:+.1f}%)  "
        f"{record['wins']}W/{record['losses']}L  WR={wr}  {n_trades} trades"
    )

    # Reset for next session — carry forward end-of-session cash
    carry_cash = round(mstate["cash"], 2)
    _state[market_key] = _empty_mstate(carry_cash, today)
    apex_log.info(
        f"[SESSION] {market_key.upper()} reset — carry_cash={sym}{carry_cash:.0f}  ({today})"
    )


def _check_session_rotation(prices: dict):
    """If the calendar date has changed since the active session was started,
    archive the old session and open a fresh one. Call with _lock held.

    Tolerates a market that has not been loaded: this runs on the agent thread,
    where a KeyError would kill the thread and stop trading silently.
    """
    for market_key in ("india", "us"):
        mstate = _state.get(market_key)
        if mstate is None:
            apex_log.warning(
                "[SESSION] No %s state loaded — cannot check session rollover", market_key
            )
            continue
        tz        = IST if market_key == "india" else EDT
        today     = datetime.now(tz).strftime("%Y-%m-%d")
        sess_date = mstate.get("session_date", today)
        if sess_date == today:
            continue
        # Date rolled over
        has_activity = (
            mstate.get("wins", 0) + mstate.get("losses", 0) > 0
            or len(mstate.get("trade_log", [])) > 0
        )
        if has_activity:
            apex_log.info(
                f"[SESSION] New day for {market_key.upper()} "
                f"({sess_date} → {today}) — archiving old session"
            )
            _close_session(market_key, prices)
        else:
            _state[market_key]["session_date"] = today
            apex_log.debug(
                f"[SESSION] {market_key.upper()} session date updated to {today} (no prior activity)"
            )


# ─── TRADE EXECUTION ──────────────────────────────────────────────────────────

def _fill_price(price: float, direction: str) -> float:
    """The price an order actually fills at, after slippage.

    A buy pays up; a sell receives less. Recording the fill rather than the
    quote is what makes slippage propagate: entry, stop, target, running
    high/low and the cash guard all derive from what was really paid or
    received, so a long round trip pays slippage once on entry and once on
    exit, which is what a market would charge.

    This is the only place slippage is applied. Four call sites each computing
    their own adjustment is how one of them eventually ends up inverted, which
    would make the cost a rebate.

    A negative or non-finite setting is ignored rather than clamped: a negative
    value would make every trade profitable, and silently clamping hides the
    mistake. The cost can also not be allowed to drive a fill to zero or below,
    which would corrupt every downstream calculation.
    """
    try:
        slip = float(cfg.get("slippage_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        return price
    if not _isfinite(slip) or slip <= 0:
        return price
    # Bound to below 100% so a bad setting cannot invert or zero the fill.
    slip = min(slip, 0.5)
    adj = price * (1.0 + slip) if direction == "buy" else price * (1.0 - slip)
    return adj if adj > 0 else price

def log_trade(mstate: dict, msg: str, kind: str):
    mstate["trade_log"].append({
        "time":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": msg,
        "kind":    kind,
    })
    if len(mstate["trade_log"]) > 200:
        mstate["trade_log"] = mstate["trade_log"][-200:]

def execute_buy(symbol: str, price: float, mstate: dict, atr: float = None, rcfg: dict | None = None):
    # rcfg is the effective config (base + this cycle's JEV gates). Defaulting to
    # cfg keeps ad-hoc/manual calls working.
    rcfg = rcfg if rcfg is not None else cfg
    # Fill price, not quote: a buy pays up. Everything below — sizing, the cash
    # guard, the stop, the target, the recorded entry — uses this price, so
    # slippage is charged once here and once again on the matching sell.
    price = _fill_price(price, "buy")
    # ── Position sizing: ATR-normalised risk, capped at max_position_pct ─────
    start_cash   = mstate.get("session_start_cash", price * 10)
    risk_dollars = start_cash * rcfg["risk_per_trade"]
    atr_sl_mult  = cfg["atr_sl_mult"]

    if atr and atr > 0 and cfg.get("use_atr_exits", True):
        sl_dist  = atr * atr_sl_mult
        tp_dist  = atr * cfg["atr_tp_mult"]
        # Floor/cap: keep SL/TP within 0.5×–2× of the fixed-pct fallback
        sl_floor = price * rcfg["stop_loss_pct"] * 0.5
        sl_cap   = price * rcfg["stop_loss_pct"] * 2.0
        tp_floor = price * cfg["target_pct"]    * 0.5
        tp_cap   = price * cfg["target_pct"]    * 2.0
        if cfg.get("settings_enabled", True):
            sl_dist = max(sl_floor, min(sl_cap, sl_dist))
            tp_dist = max(tp_floor, min(tp_cap, tp_dist))
        # Risk-adjusted qty: how many shares until 1 SL hit = risk_dollars
        qty = max(1, int(risk_dollars / sl_dist)) if sl_dist > 0 else 1
    else:
        sl_dist = price * rcfg["stop_loss_pct"]
        tp_dist = price * cfg["target_pct"]
        qty     = max(1, int((mstate["cash"] * rcfg["risk_per_trade"]) / price))

    # Hard cap: single position may not exceed max_position_pct of available cash
    max_qty = max(1, int(mstate["cash"] * cfg["max_position_pct"] / price))
    qty     = min(qty, max_qty)

    cost = qty * price
    if cost > mstate["cash"]:
        return None

    # Commission on the buy side
    commission = cost * cfg.get("commission_pct", 0.0)
    if mstate["cash"] < cost + commission:
        return None

    mstate["cash"] -= cost + commission

    sl  = round(price - sl_dist, 2)
    tgt = round(price + tp_dist, 2)

    mstate["positions"][symbol] = {
        "side":         "long",
        "qty":          qty,
        "entry":        price,
        "stop_loss":    sl,
        "target":       tgt,
        "atr":          atr or 0.0,
        "running_high": price,
        "entered_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_trade(mstate,
              f"BUY {qty} {symbol} @ {price:.2f}  SL:{sl:.2f}  T:{tgt:.2f}  "
              f"Cost:{cost:.0f}  ATR:{atr:.3f}" if atr else
              f"BUY {qty} {symbol} @ {price:.2f}  SL:{sl:.2f}  T:{tgt:.2f}  Cost:{cost:.0f}",
              "BUY")
    apex_log.info(
        f"BUY  {qty}x {symbol} @ {price:.2f}  cost={cost:.0f}  "
        f"SL={sl:.2f}  T={tgt:.2f}  ATR={f'{atr:.3f}' if atr else 'n/a'}"
    )
    return qty

def execute_short(symbol: str, price: float, mstate: dict, atr: float = None, rcfg: dict | None = None):
    """Open a short position: sell-to-open, profit when price falls."""
    rcfg = rcfg if rcfg is not None else cfg
    # Sell-to-open fills below the quote, and the matching cover fills above it.
    price = _fill_price(price, "sell")
    start_cash   = mstate.get("session_start_cash", price * 10)
    risk_dollars = start_cash * rcfg["risk_per_trade"]
    atr_sl_mult  = cfg["atr_sl_mult"]

    if atr and atr > 0 and cfg.get("use_atr_exits", True):
        sl_dist = atr * atr_sl_mult
        tp_dist = atr * cfg["atr_tp_mult"]
        sl_floor = price * rcfg["stop_loss_pct"] * 0.5
        sl_cap   = price * rcfg["stop_loss_pct"] * 2.0
        tp_floor = price * cfg["target_pct"]    * 0.5
        tp_cap   = price * cfg["target_pct"]    * 2.0
        if cfg.get("settings_enabled", True):
            sl_dist = max(sl_floor, min(sl_cap, sl_dist))
            tp_dist = max(tp_floor, min(tp_cap, tp_dist))
        qty = max(1, int(risk_dollars / sl_dist)) if sl_dist > 0 else 1
    else:
        sl_dist = price * rcfg["stop_loss_pct"]
        tp_dist = price * cfg["target_pct"]
        qty     = max(1, int((start_cash * rcfg["risk_per_trade"]) / price))

    max_qty = max(1, int(mstate["cash"] * cfg["max_position_pct"] / price))
    qty     = min(qty, max_qty)

    # Only commission deducted on short open; PnL settled on cover
    commission = qty * price * cfg.get("commission_pct", 0.0)
    if mstate["cash"] < commission + price:   # keep at least 1-share worth as buffer
        return None

    mstate["cash"] -= commission

    sl  = round(price + sl_dist, 2)                       # SL ABOVE entry
    tgt = round(max(price - tp_dist, 0.01), 2)            # TP BELOW entry

    mstate["positions"][symbol] = {
        "side":        "short",
        "qty":         qty,
        "entry":       price,
        "stop_loss":   sl,
        "target":      tgt,
        "atr":         atr or 0.0,
        "running_low": price,
        "entered_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log_trade(mstate,
              f"SHORT {qty} {symbol} @ {price:.2f}  SL:{sl:.2f}  T:{tgt:.2f}  "
              + (f"ATR:{atr:.3f}" if atr else ""),
              "SHORT")
    apex_log.info(
        f"SHORT {qty}x {symbol} @ {price:.2f}  SL={sl:.2f}  T={tgt:.2f}  "
        f"ATR={f'{atr:.3f}' if atr else 'n/a'}"
    )
    return qty

# ─── ONLINE-LEARNER FEEDBACK ─────────────────────────────────────────────────
# These used to be inline `try/except: pass` blocks in the trade paths. A failure
# left the learner's _pending buffer empty, so record_exit became a no-op and the
# online learner never trained — with nothing logged anywhere. Failures are now
# visible, and never propagate into the trade path.


def rl_feedback_entry(sym: str, action: int, price: float, atr: float = 0.0) -> None:
    """Feed an opened trade to the online learner. Never raises."""
    try:
        from trading_agent.integration.rl_signal import get_cached_obs as _get_obs
        from trading_agent.integration.online_learner import record_entry as _rl_entry
        obs = _get_obs(sym)
        if obs is None:
            return
        _rl_entry(sym, obs, action, price, atr)
    except Exception as e:  # noqa: BLE001
        apex_log.warning("RL record_entry failed for %s: %s — learner will not see this entry", sym, e)
        think_log("RL", f"WARNING: learner entry feedback failed for {sym}: {e}", sym)


def rl_feedback_exit(symbol: str, price: float, pnl: float, atr: float = 0.0, side: str = "long") -> None:
    """Feed a closed trade to the online learner. Never raises."""
    try:
        from trading_agent.integration.online_learner import record_exit as _rl_exit
        _rl_exit(symbol, price, pnl, atr, side=side)
    except Exception as e:  # noqa: BLE001
        apex_log.warning("RL record_exit failed for %s: %s — learner will not see this exit", symbol, e)
        think_log("RL", f"WARNING: learner exit feedback failed for {symbol}: {e}", symbol)


def execute_sell(symbol: str, price: float, reason: str, mstate: dict):
    pos = mstate["positions"].get(symbol)
    if not pos:
        return
    # A sell receives less than the quote. Charged against the recorded entry,
    # which was itself a slipped fill, so a flat round trip costs 2x slippage.
    price = _fill_price(price, "sell")
    proceeds   = pos["qty"] * price
    commission = proceeds * cfg.get("commission_pct", 0.0)
    net_proceeds = proceeds - commission
    pnl = net_proceeds - pos["entry"] * pos["qty"]
    mstate["cash"]         += net_proceeds
    mstate["realised_pnl"] += pnl
    if pnl >= 0:
        mstate["wins"]   += 1
    else:
        mstate["losses"] += 1
    sign = "+" if pnl >= 0 else ""
    log_trade(mstate,
              f"SELL {pos['qty']} {symbol} @ {price:.2f}  P&L:{sign}{pnl:.0f}  ({reason})",
              "SELL" if pnl >= 0 else "LOSS")
    apex_log.info(f"SELL {pos['qty']}x {symbol} @ {price:.2f}  P&L={sign}{pnl:.0f}  reason={reason}")
    del mstate["positions"][symbol]

    # Online learning: feed closed long trade back to PPO
    rl_feedback_exit(symbol, price, pnl, pos.get("atr", 0.0), side="long")

    # Set re-entry cooldown after a stop-loss
    if reason == "STOP LOSS":
        cooldown_min = cfg.get("cooldown_after_sl_min", 60)
        until = (
            datetime.now(timezone.utc) + timedelta(minutes=cooldown_min)
        ).isoformat(timespec="seconds")
        mstate.setdefault("cooldown_until", {})[symbol] = until
        apex_log.info(f"[COOLDOWN] {symbol} blocked for {cooldown_min}m (stop-loss triggered)")

def execute_cover(symbol: str, price: float, reason: str, mstate: dict):
    """Close a short position (buy-to-cover). Profit when price fell below entry."""
    pos = mstate["positions"].get(symbol)
    if not pos or pos.get("side") != "short":
        return
    # Buy-to-cover fills above the quote.
    price = _fill_price(price, "buy")
    gross_pnl  = (pos["entry"] - price) * pos["qty"]    # positive when price fell
    commission = price * pos["qty"] * cfg.get("commission_pct", 0.0)
    net_pnl    = gross_pnl - commission
    mstate["cash"]         += net_pnl
    mstate["realised_pnl"] += net_pnl
    if net_pnl >= 0:
        mstate["wins"]   += 1
    else:
        mstate["losses"] += 1
    sign = "+" if net_pnl >= 0 else ""
    log_trade(mstate,
              f"COVER {pos['qty']} {symbol} @ {price:.2f}  P&L:{sign}{net_pnl:.0f}  ({reason})",
              "SELL" if net_pnl >= 0 else "LOSS")
    apex_log.info(f"COVER {pos['qty']}x {symbol} @ {price:.2f}  P&L={sign}{net_pnl:.0f}  reason={reason}")
    del mstate["positions"][symbol]

    # Online learning: feed closed short trade back to PPO
    rl_feedback_exit(symbol, price, net_pnl, pos.get("atr", 0.0), side="short")

    if reason == "STOP LOSS":
        cooldown_min = cfg.get("cooldown_after_sl_min", 60)
        until = (
            datetime.now(timezone.utc) + timedelta(minutes=cooldown_min)
        ).isoformat(timespec="seconds")
        mstate.setdefault("cooldown_until", {})[symbol] = until
        apex_log.info(f"[COOLDOWN] {symbol} blocked for {cooldown_min}m (short stop-loss triggered)")

# ─── MARKET CYCLE ─────────────────────────────────────────────────────────────

# ── Per-symbol JEV cache + budget (v2) ─────────────────────────────────────
# {(symbol): (fetched_at, decisions)} — avoids 32 API calls per cycle.
_jev_sym_cache: dict = {}
_JEV_SYM_TTL = 300
_JEV_SYM_MAX_PER_CYCLE = 10


def fetch_cycle_data(watchlist: list, prices: dict, market_key: str = "india") -> tuple:
    """Phase 1: fetch signals (slow, no lock needed).
    In RL mode uses PPO inference on daily bars; falls back to rule-based on failure."""
    analyses = []
    _pending_fb: list = []  # rule-fallback bars for budgeted two-pass JEV
    # Shared per-cycle market context for JEV (v2): computed once, reused per symbol.
    _vix_v2 = _jev.fetch_vix(market_key) if _JEV_AVAILABLE else 20.0
    try:
        with _lock:
            _prev_scores = [a.get("score", 0) for a in (_signals.get(market_key, []) or [])]
    except Exception:
        _prev_scores = []
    _breadth_v2 = _jev.calc_breadth(_prev_scores) if _JEV_AVAILABLE else 0.5
    for symbol in watchlist:
        if not _agent["running"]:
            break
        live = prices.get(symbol)
        if cfg.get("rl_mode") and _RL_AVAILABLE:
            if live is None:
                df = fetch_data(symbol)
                live = float(df["close"].iloc[-1]) if df is not None else None
            if live:
                try:
                    result = _get_rl_signal(symbol, live)
                    if result is not None:
                        action_name = {1: "BUY", 2: "SELL", 0: "HOLD"}.get(result.get("rl_action", 0), "HOLD")
                        probs       = result.get("rl_probs", [0, 0, 0])
                        extremes    = result.get("rl_extremes", [])
                        ext_str     = ", ".join(f"{e['feature']}={e['value']:.2f}" for e in extremes)
                        trend_tag = (
                            f"trend={result.get('trend_5d_pct', 0):+.1f}% "
                            f"({'✓' if result.get('trend_aligned') else '✗'})"
                        )
                        think_log(
                            "RL",
                            f"{action_name} | H:{probs[0]:.0%} B:{probs[1]:.0%} S:{probs[2]:.0%} | "
                            f"conf={result.get('confidence', 0):.0f}% margin={result.get('rl_margin', 0):.2f} | "
                            f"{trend_tag} | extremes: {ext_str or 'none'}",
                            symbol,
                        )
                        _save_rl_decision(symbol, market_key, result)
                        analyses.append(result)
                        time.sleep(0.1)
                        continue
                except Exception as exc:
                    apex_log.warning("RL signal error for %s: %s — falling back", symbol, exc)
        # Rule-based fallback (pass 1: collect bars; JEV budgeted in pass 2)
        df = fetch_data(symbol)
        if df is None:
            continue
        live = live or float(df["close"].iloc[-1])
        _pending_fb.append((symbol, df, live))
        time.sleep(0.2)

    # Pass 1: rule-only analyse for ranking (fast, local, no API).
    _scored_fb: dict = {}
    _order_fb: list = []
    for _sym, _df, _live in _pending_fb:
        try:
            _scored_fb[_sym] = {"df": _df, "live": _live,
                                "base": analyse(_sym, _df, _live, None)}
            _order_fb.append(_sym)
        except Exception:
            continue

    # Pass 2: budgeted per-symbol JEV — held positions first, then top-3
    # non-held by confidence. Hard cap keeps us within MAX_CALLS_PER_CYCLE.
    try:
        with _lock:
            _held_fb = set(_state.get(market_key, {}).get("positions", {}).keys())
    except Exception:
        _held_fb = set()
    _held_sel = [s for s in _order_fb if s in _held_fb][:_JEV_SYM_MAX_PER_CYCLE]
    _room = max(_JEV_SYM_MAX_PER_CYCLE - len(_held_sel), 0)
    _top3 = sorted(
        (s for s in _order_fb if s not in _held_fb),
        key=lambda s: _scored_fb[s]["base"].get("confidence", 0),
        reverse=True,
    )[:min(3, _room)]
    _jev_selected = set(_held_sel + _top3)

    for _sym in _order_fb:
        _e = _scored_fb[_sym]
        if _sym in _jev_selected:
            _jd = _get_jev_symbol_decisions(
                _sym, _e["df"], _e["live"], market_key, prices,
                _vix_v2, _breadth_v2)
            if _jd is not None:
                try:
                    _a2 = analyse(_sym, _e["df"], _e["live"], _jd)
                    _a2["jev_decisions"] = _jd  # T5: per-symbol action routing
                    analyses.append(_a2)
                    continue
                except Exception as e:
                    apex_log.debug("[JEV] symbol enrichment failed for %s: %s", _sym, e)
        analyses.append(_e["base"])

    wl_prices = {s: prices[s] for s in watchlist if s in prices}
    return analyses, wl_prices
        
# ── Per-symbol JEV cache + budget (v2) ─────────────────────────────────────
def _get_jev_symbol_decisions(symbol: str, df, live: float,
                              market_key: str, prices: dict,
                              vix: float = 20.0, breadth: float = 0.5) -> dict | None:
    """Budgeted per-symbol JEV (news_bullishness + position_action).

    Returns cached decisions when fresh, else calls the API once and
    think_logs both choices. Never raises — None on any failure/disabled.
    """
    if not (_JEV_AVAILABLE and cfg.get("jev_enabled", True)
            and cfg.get("jev_news_enabled", True)):
        return None
    now = time.time()
    try:
        cached = _jev_sym_cache.get(symbol)
        if cached and (now - cached[0]) < _JEV_SYM_TTL:
            return cached[1]
    except Exception as e:
        apex_log.debug("[JEV] symbol cache read failed for %s: %s", symbol, e)
    try:
        # Build symbol-specific state for JEV (v2: full indicator + news context)
        close_s = df["close"].astype(float)
        _rsi_v = calc_rsi(close_s) if len(df) > 14 else 50.0
        _mh_v = calc_macd(close_s)[2] if len(df) > 26 else 0.0
        _bb_u, _bb_m, _bb_l = calc_bb(close_s) if len(df) >= 20 else (live, live, live)
        _e9_v = calc_ema(close_s, 9) if len(df) > 9 else live
        _e21_v = calc_ema(close_s, 21) if len(df) > 21 else live
        _atr_v = calc_atr(df) if len(df) > 14 else 0.0
        indicators = {
            "price": live,
            "rsi": _rsi_v,
            "macd_hist": _mh_v,
            "bb_pos": round((live - _bb_l) / max(_bb_u - _bb_l, 1e-9), 4),
            "ema_gap_pct": round((_e9_v - _e21_v) / max(abs(_e21_v), 1e-9) * 100, 4),
            "adx": calc_adx(df) if len(df) > 14 else 20.0,
            "atr": _atr_v,
            "atr_pct": round(_atr_v / max(live, 1e-9), 6),
            "vol_ratio": calc_vol_ratio(df["volume"].astype(float)) if len(df) > 20 else 1.0,
        }
        # Fetch news for this symbol
        news = []
        try:
            yf_news = yf.Ticker(symbol).news or []
            recent = [n for n in yf_news if time.time() - float(n.get("providerPublishTime", 0)) < 86400]
            for item in recent[:5]:
                news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "type": item.get("type", ""),
                    "recency_h": (time.time() - float(item.get("providerPublishTime", 0))) / 3600,
                })
        except Exception as e:
            apex_log.debug("[NEWS] item parse failed: %s", e)

        with _lock:
            mstate = _state.get(market_key, {})
            _pv_v2 = portfolio_value(mstate, prices)
            _peak_v2 = mstate.get("peak_portfolio", _pv_v2)
            _unrl_v2 = unrealised_pnl(mstate, prices)
            _start_v2 = max(mstate.get("session_start_cash", 1.0), 1.0)
            _w_v2, _l_v2 = mstate.get("wins", 0), mstate.get("losses", 0)
            portfolio_ctx = {
                "cash": mstate.get("cash", 0.0),
                "cash_pct": round(mstate.get("cash", 0.0) / max(_pv_v2, 1e-9), 4),
                "drawdown_pct": round((_peak_v2 - _pv_v2) / _peak_v2 * 100, 3) if _peak_v2 > 0 else 0.0,
                "open_positions": len(mstate.get("positions", {})),
                "exposure_pct": round((_pv_v2 - mstate.get("cash", 0.0)) / max(_pv_v2, 1e-9), 4),
                "daily_pnl_pct": (mstate.get("realised_pnl", 0.0) + _unrl_v2) / _start_v2,
                "session_wr": round(_w_v2 / max(_w_v2 + _l_v2, 1), 4),
            }

        market_context = {
            "spy_trend_pct": _index_trend.get(market_key, 0.0),
            "vix": vix,
            "breadth": breadth,
        }

        # Get per-symbol JEV decisions (news_bullishness, position_action)
        symbol_state = _jev.build_market_state(symbol, indicators, news, portfolio_ctx, market_context)
        symbol_questions = {
            "news_bullishness": _jev.JEV_QUESTIONS["news_bullishness"],
            "position_action": _jev.JEV_QUESTIONS["position_action"],
        }
        symbol_result = _jev._call_jev_api(symbol_state, symbol_questions)
        if symbol_result:
            _jev_sym_cache[symbol] = (now, symbol_result)
            try:
                _nb = symbol_result.get("news_bullishness", {})
                _pa = symbol_result.get("position_action", {})
                think_log("JEV",
                          f"news={_nb.get('score', '?')} conf={_nb.get('confidence', 0):.2f}  "
                          f"action={_pa.get('choice', '?')} conf={_pa.get('confidence', 0):.2f}",
                          symbol)
            except Exception as e:
                apex_log.debug("[JEV] narration failed for %s: %s", symbol, e)
            return symbol_result
    except (requests.RequestException, ValueError, KeyError) as e:
        apex_log.debug(f"[JEV] Per-symbol decisions failed for {symbol}: {e}")
    return None

# ─── EXIT POLICIES ───────────────────────────────────────────────────────────
# These were duplicated inside apply_cycle: once in the JEV-halt early return and
# again on the normal path. The copies had already drifted — only the halt path
# applied the JEV trailing multiplier, only the main path counted sells — so the
# same position was managed two different ways depending on whether a risk gate
# happened to be open. One implementation, one behaviour, both paths.
#
# All three mutate mstate in place and are called with _lock held.


def eod_exit_pass(mstate: dict, prices: dict, mtc) -> int:
    """End-of-day liquidation: harvest winners in the harvest window, then force
    everything out inside the force-exit window. Returns the count closed."""
    closed = 0
    if mtc is None:
        return closed
    harvest_min = cfg["eod_harvest_min"]
    exit_min    = cfg["eod_exit_min"]
    for sym in list(mstate["positions"].keys()):
        pos = mstate["positions"].get(sym)
        if not pos:
            continue
        price = prices.get(sym) or get_price(sym)
        if price is None:
            continue
        is_short = pos.get("side", "long") == "short"
        pnl = ((pos["entry"] - price) * pos["qty"] if is_short
               else (price - pos["entry"]) * pos["qty"])

        if exit_min < mtc <= harvest_min and pnl > 0:
            think_log("EXIT", f"{sym} EOD HARVEST: {mtc:.0f}m to close  P&L={pnl:+.0f}", sym)
            if is_short:
                paper_cover(sym, pos["qty"], prices, mstate, f"EOD PROFIT ({mtc:.0f}m to close)")
            else:
                paper_sell(sym, pos["qty"], prices, mstate, f"EOD PROFIT ({mtc:.0f}m to close)")
            apex_log.info(f"EOD profit harvest: {sym}  P&L={pnl:+.0f}  {mtc:.0f}m left")
            closed += 1

        elif mtc <= exit_min:
            tag = "EOD EXIT" if pnl >= 0 else "EOD CUT LOSS"
            think_log("EXIT", f"{sym} EOD FORCE EXIT ({tag}): {mtc:.0f}m to close  P&L={pnl:+.0f}", sym)
            if is_short:
                paper_cover(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
            else:
                paper_sell(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
            apex_log.info(f"EOD force exit: {sym}  P&L={pnl:+.0f}  {mtc:.0f}m left")
            closed += 1
    return closed


def _trailing_dist(jev_decisions) -> float:
    """Trailing distance multiplier, widened by a confident low-trend JEV read."""
    base = cfg.get("trailing_dist_mult", 1.5)
    if jev_decisions and _JEV_AVAILABLE and cfg.get("jev_trend_enabled", True):
        return _jev.get_trailing_dist_mult(base, jev_decisions)
    return base


def stop_pass(mstate: dict, prices: dict, jev_decisions=None) -> int:
    """Ratchet trailing stops, then close on stop-loss or target. Count closed.

    Trailing runs before the SL/TP check for each symbol so a freshly ratcheted
    stop is honoured in the same cycle.
    """
    closed = 0
    trailing_on = cfg.get("trailing_stop_enabled", True)
    act_mult = cfg.get("trailing_activation_mult", 1.0)
    dist_mult = _trailing_dist(jev_decisions)

    for sym in list(mstate["positions"].keys()):
        pos = mstate["positions"].get(sym)
        if pos is None or prices.get(sym) is None:
            continue
        price = prices[sym]
        is_short = pos.get("side", "long") == "short"

        # ── Trailing stop — mirrored for longs vs shorts ──────────────────────
        if trailing_on and pos.get("atr", 0) > 0:
            if not is_short:
                # LONG: activate 1×ATR above entry, SL trails up
                activation = pos["entry"] + pos["atr"] * act_mult
                if price >= activation:
                    pos["running_high"] = max(pos.get("running_high", price), price)
                    new_sl = pos["running_high"] - pos["atr"] * dist_mult
                    if new_sl > pos["stop_loss"]:        # only ever tightens
                        old_sl = pos["stop_loss"]
                        pos["stop_loss"] = round(new_sl, 2)
                        think_log("EXIT",
                                  f"{sym} TRAIL ↑ SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                                  f"running_high={pos['running_high']:.2f}", sym)
                        apex_log.debug(
                            f"[TRAIL] {sym}  SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                            f"high={pos['running_high']:.2f}"
                        )
            else:
                # SHORT: activate 1×ATR below entry, SL trails down
                activation = pos["entry"] - pos["atr"] * act_mult
                if price <= activation:
                    pos["running_low"] = min(pos.get("running_low", price), price)
                    new_sl = pos["running_low"] + pos["atr"] * dist_mult
                    if new_sl < pos["stop_loss"]:        # only ever tightens
                        old_sl = pos["stop_loss"]
                        pos["stop_loss"] = round(new_sl, 2)
                        think_log("EXIT",
                                  f"{sym} TRAIL ↓ SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                                  f"running_low={pos['running_low']:.2f}", sym)
                        apex_log.debug(
                            f"[TRAIL-SHORT] {sym}  SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                            f"low={pos['running_low']:.2f}"
                        )

        # ── SL / target — inverted for shorts ─────────────────────────────────
        if not is_short:
            sl_hit = price <= pos["stop_loss"]
            tp_hit = price >= pos["target"]
        else:
            sl_hit = price >= pos["stop_loss"]   # short: price rising = loss
            tp_hit = price <= pos["target"]       # short: price falling = profit

        if sl_hit:
            think_log("EXIT", f"{sym} STOP LOSS hit @ {price:.2f}  SL was {pos['stop_loss']:.2f}", sym)
            if is_short:
                paper_cover(sym, pos["qty"], prices, mstate, "STOP LOSS")
            else:
                paper_sell(sym, pos["qty"], prices, mstate, "STOP LOSS")
            closed += 1
        elif tp_hit:
            think_log("EXIT", f"{sym} TARGET HIT @ {price:.2f}  T was {pos['target']:.2f}", sym)
            if is_short:
                paper_cover(sym, pos["qty"], prices, mstate, "TARGET HIT")
            else:
                paper_sell(sym, pos["qty"], prices, mstate, "TARGET HIT")
            closed += 1
    return closed


def hold_review_pass(mstate: dict, prices: dict, market_key: str, _dc: dict) -> int:
    """Let the RL agent vote an early exit on positions that survived stop/target.

    Split out of the stop pass so the exit side of a cycle reads as an ordered
    list of policies rather than one 500-line function. Returns the count closed.
    """
    closed = 0
    for sym in list(mstate["positions"].keys()):
        pos = mstate["positions"].get(sym)
        if pos is None or prices.get(sym) is None:
            continue
        price = prices[sym]
        is_short = pos.get("side", "long") == "short"
        pnl_pct = ((pos["entry"] - price) / pos["entry"] * 100 if is_short
                   else (price - pos["entry"]) / pos["entry"] * 100)

        rl_exited = False
        if cfg.get("rl_mode") and _RL_AVAILABLE:
            try:
                rl_result = _get_rl_signal(sym, price)
                if rl_result is not None:
                    rl_action = rl_result.get("rl_action", 0)
                    rl_conf = rl_result.get("confidence", 0)
                    rl_exit_thr = cfg.get("rl_exit_confidence", 55)
                    should_exit = (
                        (not is_short and rl_action == 2 and rl_conf >= rl_exit_thr) or
                        (is_short and rl_action == 1 and rl_conf >= rl_exit_thr)
                    )
                    if should_exit:
                        think_log("EXIT",
                                  f"{sym} RL EXIT  conf={rl_conf:.0f}%  "
                                  f"P&L={pnl_pct:+.1f}%  "
                                  f"action={'SELL' if rl_action == 2 else 'BUY-TO-COVER'}", sym)
                        if is_short:
                            paper_cover(sym, pos["qty"], prices, mstate, "RL EXIT")
                        else:
                            paper_sell(sym, pos["qty"], prices, mstate, "RL EXIT")
                        _dc["sell"] += 1
                        closed += 1
                        rl_exited = True
            except Exception as e:
                # Falling through here leaves the position open, so this must be
                # visible rather than silently ignored.
                apex_log.warning("RL early-exit check failed for %s: %s", sym, e)

        if not rl_exited:
            apex_log.debug(
                f"[{market_key.upper()}] SL-CHECK HOLD  {sym}  "
                f"price={price:.2f}  P&L={pnl_pct:+.1f}%  "
                f"SL={pos['stop_loss']:.2f}  T={pos['target']:.2f}"
            )
    return closed


def run_exit_policies(mstate: dict, prices: dict, mtc, jev_decisions=None) -> int:
    """The whole exit side of a cycle. Identical whether or not JEV halted entries."""
    return eod_exit_pass(mstate, prices, mtc) + stop_pass(mstate, prices, jev_decisions)


def apply_cycle(market_key: str, analyses: list, prices: dict, max_pos: int, jev_decisions_per_market: dict | None = None):
    """Phase 2: apply decisions to state (must be called with _lock held)."""
    mstate       = _state[market_key]
    mtc          = minutes_to_close(market_key)
    harvest_min  = cfg["eod_harvest_min"]
    exit_min     = cfg["eod_exit_min"]
    _dc          = {"buy": 0, "sell": 0, "hold": 0, "watch": 0}
    _strict      = _settings_active()
    conf_thr     = cfg["confidence_threshold"] if _strict else 50
    max_pos_eff  = max_pos if _strict else 999
    # Base config overlaid with this cycle's JEV risk gates. The gated keys
    # (risk_per_trade, stop_loss_pct, daily/max loss limits, open window) must
    # read from here, not from cfg, or the gates have no effect on execution.
    rcfg         = effective_cfg(market_key)
    apex_log.info(
        f"[{market_key.upper()}] think loop ▶  "
        f"{len(analyses)} symbols scanned  "
        f"{len(mstate['positions'])} positions open  "
        f"cash={mstate['cash']:.0f}"
    )
    think_log("CYCLE",
              f"{market_key.upper()} scan started — "
              f"{len(analyses)} symbols  "
              f"{len(mstate['positions'])} positions open  "
              f"cash={mstate['cash']:.0f}")

    # ── JEV Halt Gate: Check if new entries should be halted ──────────────────────
    if _JEV_AVAILABLE and cfg.get("jev_risk_enabled", True):
        jev_decisions = jev_decisions_per_market.get(market_key)
        if jev_decisions and _jev.should_halt(jev_decisions):
            think_log("RISK", f"JEV HALT: halt_new_buys={jev_decisions['halt_new_buys']['noul']:.2f} — no new entries this cycle", market_key.upper())
            apex_log.info(f"[{market_key.upper()}] JEV HALT triggered — skipping new entries")
            # Still process exits (SL/TP/trailing/EOD) but skip new entries
            pv = portfolio_value(mstate, prices)
            if pv > mstate.get("peak_portfolio", pv):
                mstate["peak_portfolio"] = pv
            _dc["sell"] += run_exit_policies(mstate, prices, mtc, jev_decisions)
            return

    # ── 1. Update peak portfolio (prerequisite for drawdown kill-switch) ──────
    pv = portfolio_value(mstate, prices)
    if pv > mstate.get("peak_portfolio", pv):
        mstate["peak_portfolio"] = pv

    # ── 2. Exits: EOD schedule, then trailing / stop / target ──────────────
    _dc["sell"] += run_exit_policies(
        mstate, prices, mtc, (jev_decisions_per_market or {}).get(market_key)
    )
    # ── 3. Surviving positions: let the RL agent vote an early exit ───────────
    _dc["sell"] += hold_review_pass(mstate, prices, market_key, _dc)

    # ── 4. Block new buys when EOD exit mode is active ───────────────────────
    if mtc is not None and mtc <= harvest_min:
        apex_log.info(
            f"[{market_key.upper()}] think loop ■  EOD mode  "
            f"SELL:{_dc['sell']}  remaining pos={len(mstate['positions'])}"
        )
        return

    # ── 5. Daily loss limit / drawdown kill-switch ────────────────────────────
    session_start = mstate.get("session_start_cash", 1.0)
    daily_loss_pct = mstate["realised_pnl"] / session_start if session_start > 0 else 0
    peak           = mstate.get("peak_portfolio", pv)
    drawdown_pct   = (peak - pv) / peak if peak > 0 else 0
    if (daily_loss_pct < -rcfg.get("daily_loss_limit_pct", 0.05)
            or drawdown_pct > rcfg.get("max_drawdown_pct", 0.08)):
        mstate["trading_halted"] = True
        think_log("RISK",
                  f"TRADING HALTED — daily_loss={daily_loss_pct:.1%}  "
                  f"drawdown={drawdown_pct:.1%}  No new entries this session")
        apex_log.warning(
            f"[{market_key.upper()}] ⛔ TRADING HALTED — "
            f"daily_loss={daily_loss_pct:.1%}  drawdown={drawdown_pct:.1%}"
        )
        return
    mstate["trading_halted"] = False

    # ── 6. Market open observation window filter ──────────────────────────────
    tz_local  = IST if market_key == "india" else EDT
    now_local = datetime.now(tz_local)
    open_h, open_m = (9, 15) if market_key == "india" else (9, 30)
    open_time = now_local.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    mins_since_open = (now_local - open_time).total_seconds() / 60
    in_open_window  = 0 <= mins_since_open < rcfg.get("open_filter_min", 15)
    open_conf_gate  = cfg.get("open_filter_confidence", 85)

    # ── 7. Index trend gate ───────────────────────────────────────────────────
    index_pct = _index_trend.get(market_key, 0.0)
    index_ok  = (not _strict) or (index_pct >= cfg.get("index_min_pct", -0.5))
    if not index_ok:
        apex_log.info(
            f"[{market_key.upper()}] Index filter: {index_pct:+.2f}% — "
            f"suppressing new longs"
        )

    open_pos = len(mstate["positions"])
    now_utc  = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for a in sorted(analyses, key=lambda x: x["confidence"], reverse=True):
        sym    = a["symbol"]
        in_pos = sym in mstate["positions"]
        conf   = a["confidence"]
        adx_val = a.get("adx", 25.0)

        # ── SCAN log for every symbol not already held ────────────────────────
        if not in_pos:
            direction = (f"▲long({conf:.0f}%)" if a["score"] > 0
                         else f"▼short({100-conf:.0f}%)" if a["score"] < 0
                         else "neutral")
            hw   = a.get("hist_win_days", 0)
            ht   = a.get("hist_total_days", 0)
            harr = "↑" if hw > ht / 2 else "↓" if hw < ht / 2 else "→"
            hstr = f"Hist={hw}/{ht}{harr}" if ht > 0 else "Hist=n/a"
            nc   = a.get("news_count", 0)
            ns   = a.get("news_score", 0.0)
            narr = "↑" if ns > 0 else "↓" if ns < 0 else "→"
            nstr = f"News={nc}{narr}" if nc > 0 else "News=n/a"
            think_log("SCAN",
                      f"RSI={a.get('rsi', 0):.1f}  "
                      f"Score={a['score']:+d}  {direction}  "
                      f"ADX={adx_val:.1f}  ATR={a.get('atr', 0):.3f}  {hstr}  {nstr}",
                      sym)

        # Signal exit for held positions (no window restriction)
        if in_pos:
            pos = mstate["positions"][sym]
            pos_side = pos.get("side", "long")
            
            # JEV Exit Override (Task 24, v2): per-symbol position_action first,
            # system-level fallback (which lacks position_action).
            jev_exit_triggered = False
            if _JEV_AVAILABLE and cfg.get("jev_action_enabled", True):
                jev_decisions = a.get("jev_decisions") or jev_decisions_per_market.get(market_key)
                if jev_decisions:
                    action, action_conf = _jev.get_position_action(jev_decisions)
                    if action == "exit" and action_conf >= cfg.get("jev_action_confidence_threshold", 0.65):
                        think_log("EXIT",
                                  f"{sym} JEV EXIT: conf={action_conf:.2f}  pos_side={pos_side}", sym)
                        if pos_side == "long":
                            paper_sell(sym, 0, prices, mstate, "JEV EXIT")
                        else:
                            paper_cover(sym, 0, prices, mstate, "JEV EXIT")
                        _dc["sell"] += 1
                        jev_exit_triggered = True
                    elif action == "trim" and action_conf >= 0.7:
                        think_log("EXIT",
                                  f"{sym} JEV TRIM: conf={action_conf:.2f}  pos_side={pos_side}", sym)
                        # Sell 50% of position
                        qty = pos["qty"] // 2
                        if qty > 0:
                            if pos_side == "long":
                                paper_sell(sym, qty, prices, mstate, "JEV TRIM")
                            else:
                                paper_cover(sym, qty, prices, mstate, "JEV TRIM")
                            _dc["sell"] += 1
                            jev_exit_triggered = True
            
            if jev_exit_triggered:
                continue
            
            if pos_side == "long" and a["score"] < -30:
                think_log("EXIT",
                          f"{sym} SIGNAL EXIT: score={a['score']}  conf={conf:.0f}%  "
                          f"(bearish reversal)", sym)
                paper_sell(sym, 0, prices, mstate, "SIGNAL EXIT")
                _dc["sell"] += 1
                continue
            elif pos_side == "short" and a["score"] > 30:
                think_log("EXIT",
                          f"{sym} SHORT COVER: score={a['score']}  conf={conf:.0f}%  "
                          f"(bullish reversal)", sym)
                paper_cover(sym, 0, prices, mstate, "SIGNAL COVER")
                _dc["sell"] += 1
                continue
            apex_log.debug(
                f"[{market_key.upper()}] HOLD {pos_side.upper()}  {sym}  "
                f"score={a['score']:+d}  conf={conf:.0f}%"
            )
            _dc["hold"] += 1
            continue

        # ── New entry checks ─────────────────────────────────────────────────

        # Open window: only allow trades if confidence >= 85%
        if in_open_window and conf < open_conf_gate:
            think_log("FILTER",
                      f"{sym} SKIP open-window: {mins_since_open:.0f}m since open  "
                      f"conf={conf:.0f}% < {open_conf_gate}%", sym)
            apex_log.debug(
                f"[{market_key.upper()}] OBSERVE  {sym}  "
                f"conf={conf:.0f}% < {open_conf_gate}% (open window)"
            )
            _dc["watch"] += 1
            continue
        elif in_open_window and conf >= open_conf_gate:
            think_log("FILTER",
                      f"{sym} ALLOW open-window: conf={conf:.0f}% >= {open_conf_gate}%  "
                      f"(high conviction override)", sym)

        # Index trend gate — longs blocked when index is down; shorts evaluated separately
        # (no hard `continue` here — the check is embedded in the entry conditions below
        #  so a bearish signal can still go to the short entry branch on a down-index day)

        # ADX trend gate — applies to both longs and shorts (Task 19: JEV trend_strength augmentation)
        effective_adx_min = cfg.get("adx_min", 20)
        if _JEV_AVAILABLE and cfg.get("jev_trend_enabled", True):
            jev_decisions = jev_decisions_per_market.get(market_key)
            if jev_decisions:
                effective_adx_min = _jev.get_effective_adx_min(effective_adx_min, jev_decisions)
        if _strict and adx_val < effective_adx_min:
            think_log("FILTER",
                      f"{sym} SKIP ADX: {adx_val:.1f} < {effective_adx_min:.1f}  "
                      f"(market not trending, signals unreliable)", sym)
            apex_log.debug(
                f"[{market_key.upper()}] NO-TREND  {sym}  ADX={adx_val:.1f} eff_min={effective_adx_min:.1f}"
            )
            _dc["watch"] += 1
            continue

        # Re-entry cooldown
        cooldown_until = mstate.get("cooldown_until", {}).get(sym, "")
        if cooldown_until and now_utc < cooldown_until:
            think_log("FILTER",
                      f"{sym} SKIP cooldown: re-entry blocked until {cooldown_until}  "
                      f"(60m pause after stop-loss)", sym)
            apex_log.debug(f"[COOLDOWN] {sym}  blocked until {cooldown_until}")
            _dc["watch"] += 1
            continue
        elif cooldown_until and now_utc >= cooldown_until:
            mstate["cooldown_until"].pop(sym, None)

        # ── JEV Position Action Gate (Tasks 22-27, v2: per-symbol first) ─────────
        jev_action = None
        jev_action_conf = 0.0
        if _JEV_AVAILABLE and cfg.get("jev_action_enabled", True):
            jev_decisions = a.get("jev_decisions") or jev_decisions_per_market.get(market_key)
            if jev_decisions:
                jev_action, jev_action_conf = _jev.get_position_action(jev_decisions)

        # ── LONG entry ───────────────────────────────────────────────────────
        short_enabled    = cfg.get("short_selling_enabled", False)
        bearish_conf     = 100 - conf
        short_threshold  = cfg.get("short_confidence_threshold", 75) if _strict else 60
        index_max_short  = cfg.get("index_max_pct_for_short", 0.5)

        if (conf >= conf_thr
                and a["score"] > 0
                and (a.get("rl_action") is not None or index_ok)  # RL bypasses index gate
                and open_pos < max_pos_eff
                and mstate["cash"] > a["price"] * 2
                and (jev_action is None or jev_action in ("buy", "add"))  # JEV action gate
                and (jev_action is None or jev_action_conf >= cfg.get("jev_action_confidence_threshold", 0.65))):
            think_log("ENTRY",
                      f"{sym} ENTRY: conf={conf:.0f}%  score={a['score']:+d}  "
                      f"@ {a['price']:.2f}  ADX={adx_val:.1f}  ATR={a.get('atr', 0):.3f}", sym)
            if paper_buy(sym, 0, prices, mstate, atr=a.get("atr"), rcfg=rcfg):
                open_pos += 1
                _dc["buy"] += 1
                # Online learning: record obs+action for any trade where RL built
                # an observation (RL mode runs _build_observation for all symbols
                # even when the entropy gate rejects the signal). For rule-based
                # fallback trades we use action=1 (BUY) to teach the model the outcome.
                rl_feedback_entry(sym, a.get("rl_action", 1), a["price"], a.get("atr", 0.0))

        # ── SHORT entry ──────────────────────────────────────────────────────
        elif (short_enabled
                and a.get("rl_action") is None      # RL agent is long-only
                and not in_open_window              # too volatile at open; skip shorts
                and bearish_conf >= short_threshold
                and a["score"] < 0
                and index_pct <= index_max_short    # block shorts on strongly bullish days
                and open_pos < max_pos_eff
                and mstate["cash"] > a["price"] * 2
                and (jev_action is None or jev_action in ("buy", "add"))  # JEV action gate (buy=short for JEV)
                and (jev_action is None or jev_action_conf >= cfg.get("jev_action_confidence_threshold", 0.65))):
            think_log("ENTRY",
                      f"{sym} SHORT ENTRY: bearish_conf={bearish_conf:.0f}%  score={a['score']:+d}  "
                      f"@ {a['price']:.2f}  ADX={adx_val:.1f}  ATR={a.get('atr', 0):.3f}", sym)
            if paper_short(sym, 0, prices, mstate, atr=a.get("atr"), rcfg=rcfg):
                open_pos += 1
                _dc["buy"] += 1
                rl_feedback_entry(sym, 2, a["price"], a.get("atr", 0.0))  # 2=SELL

        else:
            # Diagnose which condition failed
            if open_pos >= max_pos_eff:
                think_log("FILTER",
                          f"{sym} SKIP: max positions ({max_pos_eff}) already open", sym)
            elif mstate["cash"] <= a["price"] * 2:
                think_log("FILTER",
                          f"{sym} SKIP: insufficient cash  "
                          f"cash={mstate['cash']:.0f}  need≈{a['price']*2:.0f}", sym)
            elif not index_ok and a["score"] > 0:
                think_log("FILTER",
                          f"{sym} SKIP index: {market_key} {index_pct:+.2f}% < "
                          f"{cfg.get('index_min_pct', -0.5):.1f}% gate (long blocked)", sym)
            elif short_enabled and bearish_conf >= short_threshold and index_pct > index_max_short:
                think_log("FILTER",
                          f"{sym} SKIP short: index {index_pct:+.2f}% > {index_max_short:.1f}% "
                          f"(bull day, shorting blocked)", sym)
            elif conf < cfg["confidence_threshold"] and bearish_conf < short_threshold:
                think_log("FILTER",
                          f"{sym} SKIP: conf={conf:.0f}% (long<{cfg['confidence_threshold']}%)  "
                          f"bearish_conf={bearish_conf:.0f}% (short<{short_threshold}%)", sym)
            else:
                think_log("FILTER",
                          f"{sym} SKIP: score={a['score']} — not strong enough for long or short", sym)
            apex_log.debug(
                f"[{market_key.upper()}] WATCH  {sym}  "
                f"score={a['score']:+d}  conf={conf:.0f}%  ADX={adx_val:.1f}"
            )
            _dc["watch"] += 1

    apex_log.info(
        f"[{market_key.upper()}] think loop ■  "
        f"BUY:{_dc['buy']}  SELL:{_dc['sell']}  "
        f"HOLD:{_dc['hold']}  WATCH:{_dc['watch']}  "
        f"cash={mstate['cash']:.0f}  pos={len(mstate['positions'])}  "
        f"index={index_pct:+.2f}%"
        + (f"  [OPEN WINDOW {mins_since_open:.0f}m/{rcfg['open_filter_min']}m]"
           if in_open_window else "")
    )

def portfolio_value(mstate: dict, prices: dict) -> float:
    v = mstate["cash"]
    for s, p in mstate["positions"].items():
        cur = prices.get(s) or p["entry"]
        if p.get("side", "long") == "long":
            v += cur * p["qty"]
        else:  # short: contribution = (entry − current) × qty
            v += (p["entry"] - cur) * p["qty"]
    return v

def unrealised_pnl(mstate: dict, prices: dict) -> float:
    total = 0.0
    for s, p in mstate["positions"].items():
        cur = prices.get(s, p["entry"])
        if p.get("side", "long") == "long":
            total += (cur - p["entry"]) * p["qty"]
        else:
            total += (p["entry"] - cur) * p["qty"]
    return total

# ─── PAPER TRADING LAYER ──────────────────────────────────────────────────────

def paper_buy(symbol: str, qty: int, latest_prices: dict, mstate: dict, atr: float = None, rcfg: dict | None = None):
    """Paper-trade a buy using the price from the snapshot.
    qty is advisory; actual qty is computed from risk settings inside execute_buy."""
    price = latest_prices.get(symbol)
    if price is None:
        apex_log.warning(f"paper_buy: no price for {symbol} in snapshot — skipping")
        return None
    return execute_buy(symbol, price, mstate, atr=atr, rcfg=rcfg)

def paper_sell(symbol: str, qty: int, latest_prices: dict, mstate: dict, reason: str = "SIGNAL"):
    """Paper-trade a sell using the price from the snapshot (falls back to live fetch)."""
    price = latest_prices.get(symbol) or get_price(symbol)
    if price is None:
        apex_log.warning(f"paper_sell: no price for {symbol} — skipping")
        return
    execute_sell(symbol, price, reason, mstate)

def paper_short(symbol: str, qty: int, latest_prices: dict, mstate: dict, atr: float = None, rcfg: dict | None = None):
    """Paper-trade a short: sell-to-open at snapshot price."""
    price = latest_prices.get(symbol)
    if price is None:
        apex_log.warning(f"paper_short: no price for {symbol} — skipping")
        return None
    return execute_short(symbol, price, mstate, atr=atr, rcfg=rcfg)

def paper_cover(symbol: str, qty: int, latest_prices: dict, mstate: dict, reason: str = "SIGNAL"):
    """Paper-trade a short cover: buy-to-close at snapshot price (falls back to live)."""
    price = latest_prices.get(symbol) or get_price(symbol)
    if price is None:
        apex_log.warning(f"paper_cover: no price for {symbol} — skipping")
        return
    execute_cover(symbol, price, reason, mstate)

def get_summary(latest_prices: dict) -> dict:
    """Return a snapshot summary of both paper portfolios at current prices."""
    with _lock:
        # Tolerate a market that has not been loaded yet; this is called on the
        # /api/state poll and must not raise.
        im = _state.get("india") or _empty_mstate(cfg["india_capital"], "")
        um = _state.get("us")    or _empty_mstate(cfg["us_capital"], "")
        india_pv   = portfolio_value(im, latest_prices)
        us_pv      = portfolio_value(um, latest_prices)
        india_upnl = unrealised_pnl(im, latest_prices)
        us_upnl    = unrealised_pnl(um, latest_prices)
        return {
            "india": {
                "cash":            round(im["cash"], 2),
                "portfolio_value": round(india_pv, 2),
                "unrealised_pnl":  round(india_upnl, 2),
                "realised_pnl":    round(im["realised_pnl"], 2),
                "open_positions":  len(im["positions"]),
                "wins":            im["wins"],
                "losses":          im["losses"],
            },
            "us": {
                "cash":            round(um["cash"], 2),
                "portfolio_value": round(us_pv, 2),
                "unrealised_pnl":  round(us_upnl, 2),
                "realised_pnl":    round(um["realised_pnl"], 2),
                "open_positions":  len(um["positions"]),
                "wins":            um["wins"],
                "losses":          um["losses"],
            },
        }

# ─── THREAD MANAGEMENT ────────────────────────────────────────────────────────

_price_thread: threading.Thread = None

def start_price_updater():
    global _price_thread
    if _price_thread and _price_thread.is_alive():
        return
    _price_thread = threading.Thread(target=_price_updater, daemon=True, name="apex-prices")
    _price_thread.start()
    apex_log.info("Price updater thread started (10 s interval)")

# ─── AGENT THREAD ─────────────────────────────────────────────────────────────

def agent_loop():
    global _state
    # Loading is blocking I/O — done before taking the lock.
    load_cfg()
    _state = load_state()
    _agent["status"] = "running"
    apex_log.info("Agent started — dual-market cycle active")
    if RL_MODE and _RL_AVAILABLE:
        apex_log.info("RL agent ACTIVE — model loaded, inference enabled")
        think_log("RL", "RL agent online. Model loaded successfully.", "SYSTEM")
    elif RL_MODE and not _RL_AVAILABLE:
        apex_log.warning("RL_MODE=true but model failed to load — falling back to rule-based")
        think_log("RL", "WARNING: RL model not found. Falling back to rule-based signals.", "SYSTEM")

    # Track previous open state to detect market-close transitions
    _prev_india_open = False
    _prev_us_open    = False
    consecutive_failures = 0

    while _agent["running"]:
        if _agent["paused"]:
            _agent["status"] = "paused"
            time.sleep(5)
            continue

        # Everything from here to the sleep is one cycle. It is wrapped because an
        # exception used to escape this thread and stop trading permanently —
        # silently, since nothing logs a background-thread death.
        try:
            _cycle_failures = _run_one_cycle(
                _prev_india_open, _prev_us_open
            )
            _prev_india_open, _prev_us_open, sleep_min = _cycle_failures
            if consecutive_failures:
                apex_log.info(
                    "Agent cycle recovered after %d failure(s)", consecutive_failures
                )
                think_log(
                    "CYCLE", f"Cycle recovered after {consecutive_failures} failure(s).", "SYSTEM"
                )
            consecutive_failures = 0
        except Exception as e:  # noqa: BLE001
            consecutive_failures += 1
            _agent["status"] = "error"
            apex_log.exception("Agent cycle failed (%d in a row): %s", consecutive_failures, e)
            think_log("CYCLE", f"ERROR: cycle failed ({consecutive_failures}): {e}", "SYSTEM")
            if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                apex_log.error(
                    "Agent has failed %d consecutive cycles — trading may be stalled",
                    consecutive_failures,
                )
            time.sleep(30)   # back off rather than hot-looping on the failure
            continue

        # Sleep in 5-second ticks so stop/pause respond quickly
        for _ in range(sleep_min * 12):
            if not _agent["running"]:
                break
            time.sleep(5)

    _agent["status"]  = "stopped"
    _agent["running"] = False
    apex_log.info("Agent stopped")


def _run_one_cycle(prev_india_open: bool, prev_us_open: bool) -> tuple:
    """Execute exactly one scan/apply/save cycle and return
    (prev_india_open, prev_us_open, sleep_min).

    Split out of agent_loop purely so the caller can wrap it; the body is the
    original loop body verbatim.
    """
    global _state
    if True:
        india_open = is_india_open()
        us_open    = is_us_open()
        _agent["india_open"] = india_open
        _agent["us_open"]    = us_open

        apex_log.info(
            f"Cycle  India={'OPEN' if india_open else 'closed'}  "
            f"US={'OPEN' if us_open else 'closed'}"
        )
        if RL_MODE and _RL_AVAILABLE:
            watchlist_len = (len(INDIA_WATCHLIST) if india_open else 0) + (len(US_WATCHLIST) if us_open else 0)
            think_log("RL", f"Cycle start — scanning {watchlist_len} symbols in RL mode", "SYSTEM")

        india_analyses, india_prices = [], {}
        us_analyses,    us_prices    = [], {}

        # Snapshot latest prices ONCE before any strategy logic runs (Thread 2 rule)
        with _price_lock:
            prices_snapshot = dict(_latest_prices)

        # Rotate session if the calendar date has changed
        with _lock:
            _check_session_rotation(prices_snapshot)

        # ── Auto end session when a market transitions open → closed ──────────
        _session_closed = False
        with _lock:
            if _prev_india_open and not india_open:
                apex_log.info("[SESSION] India market just closed — auto-archiving session")
                _close_session("india", prices_snapshot)
                _session_closed = True
            if _prev_us_open and not us_open:
                apex_log.info("[SESSION] US market just closed — auto-archiving session")
                _close_session("us", prices_snapshot)
                _session_closed = True
            _session_snap = snapshot_state() if _session_closed else None
        if _session_snap is not None:
            persist_state(_session_snap)

        _prev_india_open = india_open
        _prev_us_open    = us_open

        # Refresh index trend once per cycle (no lock needed — floats are atomic)
        _refresh_index_trend()

        # ── JEV Risk Gates: Apply per-market config overrides ──────────────────────
        jev_decisions_per_market = {}
        _jev_gates.clear()
        if _JEV_AVAILABLE and cfg.get("jev_enabled", True):
            _vix_by_mkt = {m: (_jev.fetch_vix(m) if _JEV_AVAILABLE else 20.0)
                           for m in ("india", "us")}
            try:
                with _lock:
                    _br_by_mkt = {
                        m: _jev.calc_breadth(
                            [a.get("score", 0) for a in (_signals.get(m, []) or [])])
                        for m in ("india", "us")
                    }
            except Exception:
                _br_by_mkt = {"india": 0.5, "us": 0.5}
            market_context = {
                "spy_trend_pct": _index_trend.get("india", 0.0) if india_open else _index_trend.get("us", 0.0),
                "vix": _vix_by_mkt["india"] if india_open else _vix_by_mkt["us"],
                "breadth": _br_by_mkt["india"] if india_open else _br_by_mkt["us"],
            }

            for market_key, is_open in [("india", india_open), ("us", us_open)]:
                if not is_open:
                    continue
                try:
                    with _lock:
                        mstate = _state.get(market_key, {})
                        _pv = portfolio_value(mstate, prices_snapshot)
                        _peak = mstate.get("peak_portfolio", _pv)
                        _unrl = unrealised_pnl(mstate, prices_snapshot)
                        _start = max(mstate.get("session_start_cash", 1.0), 1.0)
                        _w, _l = mstate.get("wins", 0), mstate.get("losses", 0)
                        portfolio_ctx = {
                            "cash": mstate.get("cash", 0.0),
                            "cash_pct": round(mstate.get("cash", 0.0) / max(_pv, 1e-9), 4),
                            "drawdown_pct": round((_peak - _pv) / _peak * 100, 3) if _peak > 0 else 0.0,
                            "open_positions": len(mstate.get("positions", {})),
                            "exposure_pct": round((_pv - mstate.get("cash", 0.0)) / max(_pv, 1e-9), 4),
                            "daily_pnl_pct": (mstate.get("realised_pnl", 0.0) + _unrl) / _start,
                            "session_wr": round(_w / max(_w + _l, 1), 4),
                        }
                        market_context = {
                            "spy_trend_pct": _index_trend.get(market_key, 0.0),
                            "vix": _vix_by_mkt.get(market_key, 20.0),
                            "breadth": _br_by_mkt.get(market_key, 0.5),
                        }
                    
                    jev_decisions = _jev.get_system_decisions(market_key, portfolio_ctx, market_context)
                    
                    if jev_decisions:
                        jev_decisions_per_market[market_key] = jev_decisions

                        # Log each JEV decision to think buffer
                        regime = jev_decisions["regime"]
                        think_log("JEV", 
                            f"regime={regime['choice']} conf={regime['confidence']:.2f} "
                            f"probs={ {k: f'{v:.2f}' for k, v in regime['probabilities'].items()} }", 
                            market_key.upper())
                        
                        trend = jev_decisions["trend_strength"]
                        think_log("JEV",
                            f"trend_strength={trend['score']:.2f} conf={trend['confidence']:.2f}",
                            market_key.upper())
                        
                        stress = jev_decisions["portfolio_stress"]
                        think_log("JEV",
                            f"portfolio_stress={stress['score']:.2f} conf={stress['confidence']:.2f}",
                            market_key.upper())
                        
                        halt = jev_decisions["halt_new_buys"]
                        think_log("JEV",
                            f"halt_new_buys={halt['noul']:.2f} conf={halt['confidence']:.2f}",
                            market_key.upper())
                        
                        apex_log.info(
                            f"[JEV] {market_key.upper()} regime={regime['choice']} "
                            f"trend={trend['score']:.2f} stress={stress['score']:.2f} "
                            f"halt={halt['noul']:.2f} → risk_per_trade={cfg['risk_per_trade']:.4f} "
                            f"max_pos={cfg.get(f'{market_key}_max_positions', 4)} "
                            f"open_window={cfg.get('open_filter_min', 15)}m "
                            f"(base values — per-cycle gates applied below)"
                        )
                except (requests.RequestException, ValueError, KeyError) as e:
                    apex_log.warning(f"[JEV] Risk gates failed for {market_key}: {e}")

            # Fold the decisions into per-cycle risk overrides. This is what
            # actually applies the gates; the log line above only reports them.
            apply_jev_cycle_gates(jev_decisions_per_market)
            publish_jev_decisions(jev_decisions_per_market)

        if india_open:
            if not _agent["running"]:
                return prev_india_open, prev_us_open, cfg["check_interval_min"]
            _agent["status"] = "Scanning India"
            apex_log.info("Scanning India watchlist…")
            india_analyses, india_prices = fetch_cycle_data(INDIA_WATCHLIST, prices_snapshot, "india")
            apex_log.info(
                f"India scan done: {len(india_analyses)}/{len(INDIA_WATCHLIST)} symbols  "
                f"prices={len(india_prices)}"
            )
            with _lock:
                apply_cycle("india", india_analyses, india_prices,
                            effective_cfg("india")["india_max_positions"], jev_decisions_per_market)
                _signals["india"]        = india_analyses
                _signals["india_prices"] = india_prices
        else:
            apex_log.debug("India market closed — skipping")

        if us_open:
            if not _agent["running"]:
                return prev_india_open, prev_us_open, cfg["check_interval_min"]
            _agent["status"] = "Scanning US"
            apex_log.info("Scanning US watchlist…")
            us_analyses, us_prices = fetch_cycle_data(US_WATCHLIST, prices_snapshot, "us")
            apex_log.info(
                f"US scan done: {len(us_analyses)}/{len(US_WATCHLIST)} symbols  "
                f"prices={len(us_prices)}"
            )
            with _lock:
                apply_cycle("us", us_analyses, us_prices,
                            effective_cfg("us")["us_max_positions"], jev_decisions_per_market)
                _signals["us"]        = us_analyses
                _signals["us_prices"] = us_prices
        else:
            apex_log.debug("US market closed — skipping")

        with _lock:
            _cycle_snap = snapshot_state()
        persist_state(_cycle_snap)
        apex_log.info("State saved")

        _agent["last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sleep_min = cfg["check_interval_min"] if (india_open or us_open) else cfg["idle_interval_min"]
        if _agent["running"]:
            _agent["status"] = "waiting"
        next_t = (datetime.now(timezone.utc) + timedelta(minutes=sleep_min)).isoformat(timespec="seconds")
        _agent["next_check"] = next_t
        apex_log.info(f"Waiting {sleep_min} min — next check at {next_t}")

        return india_open, us_open, sleep_min

# ─── FLASK APP ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── AUTH ────────────────────────────────────────────────────────────────────
import hmac as _hmac
import secrets as _secrets

# Credentials must never fall back to a guessable value. These are the defaults
# that used to be live whenever APEX_PASS was unset, and that .env.example still
# documents.
_DEFAULT_AUTH_USER = "apex"
_DEFAULT_AUTH_PASS = "admin"

app.secret_key = os.environ.get("APEX_SECRET") or _secrets.token_hex(32)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024   # 256 KB; JSON bodies are tiny

_AUTH_USER = os.environ.get("APEX_USER", "")
_AUTH_PASS = os.environ.get("APEX_PASS", "")
# Optional stored hash so the plaintext password need not sit in the env.
# Format: "<salt_hex>$<scrypt_digest_hex>" with N=2**14, r=8, p=1, dklen=32.
_AUTH_PASS_HASH = os.environ.get("APEX_PASS_HASH", "")
_CREDENTIALS_DEFAULT = (
    _AUTH_USER == _DEFAULT_AUTH_USER and _AUTH_PASS == _DEFAULT_AUTH_PASS
)
if (not _AUTH_PASS or _CREDENTIALS_DEFAULT) and not _AUTH_PASS_HASH:
    # Fail closed: mint an unguessable password rather than serve the dashboard
    # behind a documented default. The documented defaults are also rejected at
    # login, so an explicit APEX_PASS=admin cannot re-enable them.
    _AUTH_PASS = _secrets.token_urlsafe(32)

_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32


def verify_password(candidate: str, encoded: str) -> bool:
    """Check a password against a "<salt_hex>$<digest_hex>" scrypt hash."""
    if not encoded or "$" not in encoded:
        return False
    import hashlib

    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            candidate.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
            p=_SCRYPT_P, dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return _hmac.compare_digest(actual, expected)


def _password_ok(candidate: str) -> bool:
    if _AUTH_PASS_HASH:
        return verify_password(candidate, _AUTH_PASS_HASH)
    return _hmac.compare_digest(candidate.encode(), _AUTH_PASS.encode())

# Per-IP failed-login counter. The password is the only thing protecting a live
# trading control panel, so guessing it must be expensive.
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300
_login_failures: dict = {}          # ip -> [count, first_attempt_monotonic]
_login_lock = threading.Lock()


def _login_throttled(ip: str) -> tuple[bool, int]:
    """True when ip has exhausted its attempts. Returns (throttled, seconds_left)."""
    now = time.monotonic()
    with _login_lock:
        entry = _login_failures.get(ip)
        if not entry:
            return False, 0
        count, first = entry
        if now - first > LOGIN_LOCKOUT_SECONDS:
            _login_failures.pop(ip, None)
            return False, 0
        if count >= LOGIN_MAX_ATTEMPTS:
            return True, int(LOGIN_LOCKOUT_SECONDS - (now - first)) + 1
    return False, 0


def _record_login_failure(ip: str) -> None:
    now = time.monotonic()
    with _login_lock:
        entry = _login_failures.get(ip)
        if entry and now - entry[1] <= LOGIN_LOCKOUT_SECONDS:
            _login_failures[ip] = [entry[0] + 1, entry[1]]
        else:
            _login_failures[ip] = [1, now]


def _clear_login_failures(ip: str) -> None:
    with _login_lock:
        _login_failures.pop(ip, None)


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    # The SPA is a single inline document, so style/script need 'unsafe-inline'.
    # 'self' alone still blocks any externally injected script source.
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'self'",
    )
    return resp


@app.errorhandler(Exception)
def _handle_unexpected(err):
    """Return JSON, never a stack trace or Flask's HTML error page.

    The SPA calls r.json() unconditionally; an HTML 500 body turned a single
    server-side slip into a broken dashboard.
    """
    if isinstance(err, HTTPException):
        return jsonify({"ok": False, "msg": err.description}), err.code
    apex_log.exception("Unhandled error on %s %s", request.method, request.path)
    return jsonify({"ok": False, "msg": "internal_error"}), 500


@app.before_request
def _require_login():
    if request.path in ("/login", "/logout"):
        return None
    if not session.get("logged_in"):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "msg": "Not authenticated"}), 401
        return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    if request.method == "POST":
        ip = request.remote_addr or "?"
        throttled, retry_after = _login_throttled(ip)
        if throttled:
            apex_log.warning("Login throttled for %s (%ds remaining)", ip, retry_after)
            return render_template("login.html", err="Too many attempts. Try again later."), 429
        if _AUTH_USER == _DEFAULT_AUTH_USER and _AUTH_PASS == _DEFAULT_AUTH_PASS \
                and not _AUTH_PASS_HASH:
            apex_log.error("Login rejected: APEX_USER/APEX_PASS are still the documented defaults")
            return render_template("login.html", err="Server credentials are not configured."), 503
        u = request.form.get("u", "")
        p = request.form.get("p", "")
        # Constant-time compare on both fields; short-circuiting on the username
        # would leak it.
        ok = _hmac.compare_digest(u.encode(), _AUTH_USER.encode()) & _password_ok(p)
        if ok:
            session.clear()
            session["logged_in"] = True
            _clear_login_failures(ip)
            apex_log.info("Dashboard login from " + ip)
            return redirect("/")
        _record_login_failure(ip)
        err = "Invalid credentials — try again"
        apex_log.warning("Failed login attempt from " + ip)
    return render_template("login.html", err=err)


@app.route("/logout")
def logout():
    apex_log.info("Dashboard logout")
    session.clear()
    return redirect("/login")

@app.route("/")
def index():
    # render_template caches the compiled template by filename; the old inline
    # literal was re-tokenised on every request.
    return render_template("index.html")

@app.route("/api/state")
def api_state():
    with _price_lock:
        live_prices = dict(_latest_prices)
    with _lock:
        st   = json.loads(json.dumps(_state,   default=str))
        sigs = json.loads(json.dumps(_signals, default=str))
    # Prefer live prices from the 10-s updater; fall back to last decision-cycle prices
    merged_india = {**sigs["india_prices"], **{k: live_prices[k] for k in INDIA_WATCHLIST if k in live_prices}}
    merged_us    = {**sigs["us_prices"],    **{k: live_prices[k] for k in US_WATCHLIST    if k in live_prices}}
    # A missing market must not blank the whole dashboard: this route is polled
    # every 30 s and a KeyError here used to take the UI down entirely.
    india_st = st.get("india") or _empty_mstate(cfg["india_capital"], "")
    us_st    = st.get("us")    or _empty_mstate(cfg["us_capital"], "")
    india_pv   = portfolio_value(india_st, merged_india)
    us_pv      = portfolio_value(us_st,    merged_us)
    india_upnl = unrealised_pnl(india_st, merged_india)
    us_upnl    = unrealised_pnl(us_st,    merged_us)

    def dd_calc(mstate, pv):
        peak = mstate.get("peak_portfolio", pv)
        return round((peak - pv) / peak * 100, 2) if peak > 0 else 0.0

    # Get JEV decisions for dashboard
    jev_decisions = {}
    if _JEV_AVAILABLE and cfg.get("jev_enabled", True):
        market_context = {
            "spy_trend_pct": _index_trend.get("india", 0.0),
            "vix": _jev.fetch_vix("india"),
            "breadth": _jev.calc_breadth(
                [a.get("score", 0) for a in (sigs.get("india", []) or [])]),
        }
        for market_key in ("india", "us"):
            try:
                with _lock:
                    mstate = _state.get(market_key, {})
                    _pv = portfolio_value(mstate, live_prices)
                    _peak = mstate.get("peak_portfolio", _pv)
                    _unrl = unrealised_pnl(mstate, live_prices)
                    _start = max(mstate.get("session_start_cash", 1.0), 1.0)
                    portfolio_ctx = {
                        "cash": mstate.get("cash", 0.0),
                        "drawdown_pct": round((_peak - _pv) / _peak * 100, 3) if _peak > 0 else 0.0,
                        "open_positions": len(mstate.get("positions", {})),
                        "daily_pnl_pct": (mstate.get("realised_pnl", 0.0) + _unrl) / _start,
                    }
                    market_context = {
                        "spy_trend_pct": _index_trend.get(market_key, 0.0),
                        "vix": _jev.fetch_vix(market_key),
                        "breadth": _jev.calc_breadth(
                            [a.get("score", 0) for a in (sigs.get(market_key, []) or [])]),
                    }
                decisions = _jev.get_system_decisions(market_key, portfolio_ctx, market_context)
                if decisions:
                    jev_decisions[market_key] = decisions
            except (requests.RequestException, ValueError, KeyError) as e:
                apex_log.warning(f"[JEV] api_state failed for {market_key}: {e}")
                jev_decisions[market_key] = {"error": str(e)}

    return jsonify({
        "agent":  _agent,
        "config": cfg,
        "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_log":    list(_log_buffer)[-100:],
        "decision_log": list(_think_buffer)[-150:],
        "india_mtc":  minutes_to_close("india"),
        "us_mtc":     minutes_to_close("us"),
        "india": {
            **india_st,
            "portfolio_value": round(india_pv, 2),
            "unrealised_pnl":  round(india_upnl, 2),
            "total_pnl":       round(india_st["realised_pnl"] + india_upnl, 2),
            "drawdown":        dd_calc(india_st, india_pv),
            "market_open":     is_india_open(),
        },
        "us": {
            **us_st,
            "portfolio_value": round(us_pv, 2),
            "unrealised_pnl":  round(us_upnl, 2),
            "total_pnl":       round(us_st["realised_pnl"] + us_upnl, 2),
            "drawdown":        dd_calc(us_st, us_pv),
            "market_open":     is_us_open(),
        },
        "signals":        sigs,
        "live_prices":    live_prices,
        "paper_summary":  get_summary(live_prices),
        "sessions_count": len(st.get("sessions", [])),
        "jev_decisions":  jev_decisions,
    })

@app.route("/api/agent/start", methods=["POST"])
def agent_start():
    global _agent_thread
    if _agent["running"]:
        apex_log.warning("Start requested but agent already running")
        return jsonify({"ok": True, "msg": "Agent already running"})
    _agent["running"] = True
    _agent["paused"]  = False
    apex_log.info("Agent start requested via dashboard")
    start_price_updater()
    t = threading.Thread(target=agent_loop, daemon=True, name="apex-agent")
    t.start()
    return jsonify({"ok": True, "msg": "Agent started"})

@app.route("/api/agent/stop", methods=["POST"])
def agent_stop():
    apex_log.info("Agent stop requested via dashboard")
    _agent["running"] = False
    _agent["paused"]  = False
    _agent["status"]  = "stopped"
    think_log("RL", "Agent stopped by user.", "SYSTEM")
    return jsonify({"ok": True, "msg": "Stopping agent after current cycle"})

@app.route("/api/agent/pause", methods=["POST"])
def agent_pause():
    if not _agent["running"]:
        return jsonify({"ok": False, "msg": "Agent not running"})
    _agent["paused"] = not _agent["paused"]
    state = "paused" if _agent["paused"] else "resumed"
    apex_log.info(f"Agent {state} via dashboard")
    return jsonify({"ok": True, "paused": _agent["paused"]})

@app.route("/api/sell/<market>/<path:symbol>", methods=["POST"])
def force_sell(market, symbol):
    if market not in ("india", "us"):
        return jsonify({"ok": False, "msg": "Invalid market"})
    with _price_lock:
        snap = dict(_latest_prices)
    with _lock:
        mstate = _state.get(market, {})
        if symbol not in mstate.get("positions", {}):
            return jsonify({"ok": False, "msg": "Position not found"})
        price = snap.get(symbol) or get_price(symbol)
        if price is None:
            return jsonify({"ok": False, "msg": "Could not fetch live price"})
        pos = mstate["positions"].get(symbol, {})
        if pos.get("side", "long") == "long":
            paper_sell(symbol, 0, snap, mstate, "MANUAL SELL")
        else:
            paper_cover(symbol, 0, snap, mstate, "MANUAL COVER")
        _snap_to_save = snapshot_state()
    persist_state(_snap_to_save)
    apex_log.info(f"Manual close: {symbol} @ {price:.2f}  market={market}  side={pos.get('side','long')}")
    return jsonify({"ok": True, "msg": f"Sold {symbol.replace('.NS', '')}"})

@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.get_json(silent=True)
    clean, err = validate_config_payload(data)
    if err:
        return jsonify({"ok": False, "msg": err}), 400
    if not clean:
        return jsonify({"ok": False, "msg": "No valid settings supplied"}), 400
    with _lock:
        cfg.update(clean)
        if _JEV_AVAILABLE:
            _jev.configure(cfg)
    save_cfg()
    if clean.get("settings_enabled") is False:
        apex_log.warning("RISK GATES DISABLED via /api/config by the operator")
        think_log("RISK", "WARNING: settings_enabled turned off — risk filters bypassed.", "SYSTEM")
    apex_log.info(f"Config updated: {', '.join(clean)}")
    return jsonify({"ok": True, "config": cfg})

@app.route("/api/logs")
def api_logs():
    return jsonify(list(_log_buffer))

@app.route("/api/prices/stream")
def api_prices_stream():
    """SSE endpoint — pushes price updates to the browser as they arrive."""
    from queue import Queue, Empty
    q = Queue(maxsize=10)
    with _price_sub_lock:
        _price_subscribers.append(q)
    # send current snapshot immediately so the page doesn't wait for first tick
    with _price_lock:
        snapshot = dict(_latest_prices)

    def generate():
        try:
            if snapshot:
                yield f"data: {json.dumps(snapshot)}\n\n"
            while True:
                try:
                    prices = q.get(timeout=25)
                    yield f"data: {json.dumps(prices)}\n\n"
                except Empty:
                    yield ": ping\n\n"  # keepalive so proxy doesn't close connection
        finally:
            with _price_sub_lock:
                try:
                    _price_subscribers.remove(q)
                except ValueError:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/api/think")
def api_think():
    return jsonify(list(_think_buffer))

@app.route("/api/think/clear", methods=["POST"])
def api_think_clear():
    _think_buffer.clear()
    return jsonify({"ok": True})

@app.route("/api/jev/status")
def api_jev_status():
    """JEV integration status for monitoring."""
    if not _JEV_AVAILABLE:
        return jsonify({"available": False, "error": "apex_jev module not available"})
    try:
        return jsonify(_jev.get_jev_status())
    except (requests.RequestException, ValueError, KeyError) as e:
        return jsonify({"available": True, "error": str(e)}), 500

@app.route("/api/retrain/log")
def api_retrain_log():
    """Retraining activity, including updates the validation gate held back.

    `rejected_updates` is the number to watch. A run of rejections means the
    learner is proposing changes that do not generalise, which is the gate
    working — but a persistently high count with zero acceptances means online
    learning is not actually learning and the feature set is probably the
    problem, not the gate.
    """
    try:
        from trading_agent.integration.online_learner import (
            get_retrain_log, stats, _MIN_BATCH_FOR_VALIDATION,
        )
        return jsonify({
            "log":               get_retrain_log(),
            **stats(),
            "min_batch_to_validate": _MIN_BATCH_FOR_VALIDATION,
        })
    except ImportError:
        return jsonify({
            "log": [], "is_training": False, "new_count": 0, "buffer_size": 0,
            "total_updates": 0, "rejected_updates": 0, "threshold": 16,
            "min_batch_to_validate": 0,
        })

@app.route("/api/rl/decisions")
def api_rl_decisions():
    if not _sb:
        return jsonify([])
    try:
        resp = _sb.table("apex_rl_decisions").select("*").order("ts", desc=True).limit(100).execute()
        return jsonify(resp.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset/<market>", methods=["POST"])
def reset_market(market):
    if market not in ("india", "us", "all"):
        return jsonify({"ok": False, "msg": "Invalid market"})
    with _lock:
        if market in ("india", "all"):
            _state["india"] = _empty_mstate(cfg["india_capital"], datetime.now(IST).strftime("%Y-%m-%d"))
        if market in ("us", "all"):
            _state["us"]    = _empty_mstate(cfg["us_capital"],    datetime.now(EDT).strftime("%Y-%m-%d"))
        _snap_to_save = snapshot_state()
    persist_state(_snap_to_save)
    apex_log.warning(f"State reset: market={market}")
    return jsonify({"ok": True, "msg": f"Reset {market}"})

@app.route("/api/edit/state", methods=["POST"])
def edit_state():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "msg": "Body must be a JSON object"}), 400
    clean_by_mkt: dict = {}
    for mkt in ("india", "us"):
        if mkt not in data:
            continue
        clean, err = clean_numeric(data[mkt], STATE_EDIT_SPEC)
        if err:
            return jsonify({"ok": False, "msg": f"{mkt}: {err}"}), 400
        clean_by_mkt[mkt] = clean
    if not clean_by_mkt:
        return jsonify({"ok": False, "msg": "No editable fields supplied"}), 400
    changed = []
    with _lock:
        for mkt, clean in clean_by_mkt.items():
            mstate = _state.get(mkt)
            if mstate is None:
                return jsonify({"ok": False, "msg": f"Unknown market: {mkt}"}), 400
            for field, value in clean.items():
                mstate[field] = value
                changed.append(f"{mkt}.{field}")
        _snap_to_save = snapshot_state()
    persist_state(_snap_to_save)
    apex_log.info(f"Manual state edit: {', '.join(changed)}")
    return jsonify({"ok": True, "changed": changed})


@app.route("/api/edit/position/<market>/<path:symbol>", methods=["POST"])
def edit_position(market, symbol):
    if market not in ("india", "us"):
        return jsonify({"ok": False, "msg": "Invalid market"}), 400
    data = request.get_json(silent=True)
    clean, err = clean_numeric(data, POSITION_EDIT_SPEC)
    if err:
        return jsonify({"ok": False, "msg": err}), 400
    if not clean:
        return jsonify({"ok": False, "msg": "No editable fields supplied"}), 400
    with _lock:
        pos = _state.get(market, {}).get("positions", {}).get(symbol)
        if pos is None:
            return jsonify({"ok": False, "msg": "Position not found"}), 404
        pos.update(clean)
        _snap_to_save = snapshot_state()
    persist_state(_snap_to_save)
    apex_log.info(f"Manual position edit: {symbol} ({market}) {clean}")
    return jsonify({"ok": True, "changed": sorted(clean)})

@app.route("/api/sessions")
def api_sessions():
    with _lock:
        sessions = list(reversed(_state.get("sessions", [])))  # newest first
    return jsonify(sessions)

@app.route("/api/sessions/close/<market>", methods=["POST"])
def close_session_api(market):
    """Manually archive the current session and reset. Useful for testing or early EOD."""
    if market not in ("india", "us"):
        return jsonify({"ok": False, "msg": "Invalid market"})
    with _price_lock:
        snap = dict(_latest_prices)
    with _lock:
        _close_session(market, snap)
        _snap_to_save = snapshot_state()
    persist_state(_snap_to_save)
    apex_log.info(f"Manual session close: {market}")
    return jsonify({"ok": True, "msg": f"{market} session archived and reset"})

# ─── GUNICORN / WSGI STARTUP ──────────────────────────────────────────────────
# Runs when gunicorn (or any WSGI server) imports this module.
# __main__ block below is kept for local `python apex_dashboard.py` usage.

def _on_startup():
    global _state
    _state = load_state()
    start_price_updater()
    apex_log.info("Apex started — Supabase storage, dual-loop active")


# APEX_SKIP_AUTOSTART=1 imports the module without touching Supabase or spawning
# the price-updater thread. Required by the test suite; also lets a liveness probe
# import the app without side effects.
if os.getenv("APEX_SKIP_AUTOSTART", "").lower() not in ("1", "true", "yes"):
    _on_startup()
else:
    apex_log.info("APEX_SKIP_AUTOSTART set — state load and price updater skipped")

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser

    # The startup banner and the cycle logs contain box-drawing and arrow
    # characters. A Windows console defaults to cp1252, which cannot encode
    # them, so an unguarded print raises UnicodeEncodeError and the process
    # dies before the server ever binds. Logging is unaffected — it writes to
    # apex.log with an explicit utf-8 FileHandler — but stdout is not ours, so
    # reconfigure it and let unencodable characters degrade instead of
    # killing startup. No-op on a UTF-8 console.
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s is None:
            continue
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError) as _enc_err:
            # No-op on a UTF-8 console; on a non-reconfigurable stream the banner
            # below still prints because it is pure ASCII.
            print(f"[apex] stdout encoding unchanged ({_enc_err})", file=sys.stderr)

    # _on_startup() already ran at import time; no need to reload state here.
    url = "http://localhost:7000"
    print(f"\n  Apex Dashboard  ->  {url}")
    print("  Press Ctrl+C to stop\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=7000, debug=False,
            use_reloader=False, threaded=True)
