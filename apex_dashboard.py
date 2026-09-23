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
try:
    import apex_jev as _jev
    _JEV_AVAILABLE = True
except Exception as _jev_import_err:  # noqa: BLE001
    _JEV_AVAILABLE = False
    _jev = None  # type: ignore[assignment]

import time
import warnings
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, render_template_string, session, redirect, Response, stream_with_context
import yfinance as yf
import pandas as pd
import numpy as np

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
    # ── Short selling ─────────────────────────────────────────────────────────
    "short_selling_enabled":       True,
    "short_confidence_threshold":    75,  # bearish confidence needed to short (0–100)
    "index_max_pct_for_short":      0.5,  # block shorts if index UP more than +0.5%
    "settings_enabled":            True,  # False = full liberty: bypass threshold/ADX/index/cap filters
}

INDIA_WATCHLIST = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",
    "ICICIBANK.NS",  "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",
    "MARUTI.NS",     "TITAN.NS",      "WIPRO.NS",      "SUNPHARMA.NS",
]
US_WATCHLIST = [
    "AAPL",  "MSFT",  "NVDA",  "GOOGL",
    "AMZN",  "META",  "TSLA",  "AMD",
    "NFLX",  "ORCL",  "INTC",  "CRM",
    "UBER",  "SHOP",  "PYPL",  "PLTR",
]

IST = timezone(timedelta(hours=5,  minutes=30))
EDT = timezone(timedelta(hours=-4))

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
    _fh = logging.FileHandler("apex.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    apex_log.addHandler(_fh)
    apex_log.addHandler(_BufHandler())

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
}
_agent   = {
    "running": False, "paused": False,
    "status":  "idle", "last_update": None, "next_check": None,
    "india_open": False, "us_open": False,
}

# ── Shared price store (Thread 1 writes, Thread 2 snapshots) ──────────────────
_latest_prices: dict = {}
_price_lock = threading.Lock()

# ─── MARKET HOURS ─────────────────────────────────────────────────────────────

def is_india_open() -> bool:
    n = datetime.now(IST)
    if n.weekday() >= 5:
        return False
    return (n.replace(hour=9,  minute=15, second=0, microsecond=0)
            <= n <=
            n.replace(hour=15, minute=30, second=0, microsecond=0))

def is_us_open() -> bool:
    n = datetime.now(EDT)
    if n.weekday() >= 5:
        return False
    return (n.replace(hour=9,  minute=30, second=0, microsecond=0)
            <= n <=
            n.replace(hour=16, minute=0,  second=0, microsecond=0))

def minutes_to_close(market_key: str):
    """Returns minutes remaining until market close, or None if market is not currently open."""
    if market_key == "india":
        n       = datetime.now(IST)
        open_t  = n.replace(hour=9,  minute=15, second=0, microsecond=0)
        close_t = n.replace(hour=15, minute=30, second=0, microsecond=0)
    else:
        n       = datetime.now(EDT)
        open_t  = n.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = n.replace(hour=16, minute=0,  second=0, microsecond=0)
    if n.weekday() >= 5 or n < open_t or n > close_t:
        return None
    return (close_t - n).total_seconds() / 60

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
            except Exception:
                pass
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
    except Exception:
        pass
    return sym, None

def _fetch_us_price(sym: str) -> tuple[str, float | None]:
    try:
        p = yf.Ticker(sym).fast_info.last_price
        if p and p > 0:
            return sym, float(p)
    except Exception:
        pass
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
            with _price_lock:
                _latest_prices.update(fresh)
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
    archive the old session and open a fresh one. Call with _lock held."""
    for market_key in ("india", "us"):
        tz        = IST if market_key == "india" else EDT
        today     = datetime.now(tz).strftime("%Y-%m-%d")
        sess_date = _state[market_key].get("session_date", today)
        if sess_date == today:
            continue
        # Date rolled over
        has_activity = (
            _state[market_key]["wins"] + _state[market_key]["losses"] > 0
            or len(_state[market_key]["trade_log"]) > 0
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

def log_trade(mstate: dict, msg: str, kind: str):
    mstate["trade_log"].append({
        "time":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": msg,
        "kind":    kind,
    })
    if len(mstate["trade_log"]) > 200:
        mstate["trade_log"] = mstate["trade_log"][-200:]

def execute_buy(symbol: str, price: float, mstate: dict, atr: float = None):
    # ── Position sizing: ATR-normalised risk, capped at max_position_pct ─────
    start_cash   = mstate.get("session_start_cash", price * 10)
    risk_dollars = start_cash * cfg["risk_per_trade"]
    atr_sl_mult  = cfg["atr_sl_mult"]

    if atr and atr > 0 and cfg.get("use_atr_exits", True):
        sl_dist  = atr * atr_sl_mult
        tp_dist  = atr * cfg["atr_tp_mult"]
        # Floor/cap: keep SL/TP within 0.5×–2× of the fixed-pct fallback
        sl_floor = price * cfg["stop_loss_pct"] * 0.5
        sl_cap   = price * cfg["stop_loss_pct"] * 2.0
        tp_floor = price * cfg["target_pct"]    * 0.5
        tp_cap   = price * cfg["target_pct"]    * 2.0
        if cfg.get("settings_enabled", True):
            sl_dist = max(sl_floor, min(sl_cap, sl_dist))
            tp_dist = max(tp_floor, min(tp_cap, tp_dist))
        # Risk-adjusted qty: how many shares until 1 SL hit = risk_dollars
        qty = max(1, int(risk_dollars / sl_dist)) if sl_dist > 0 else 1
    else:
        sl_dist = price * cfg["stop_loss_pct"]
        tp_dist = price * cfg["target_pct"]
        qty     = max(1, int((mstate["cash"] * cfg["risk_per_trade"]) / price))

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

def execute_short(symbol: str, price: float, mstate: dict, atr: float = None):
    """Open a short position: sell-to-open, profit when price falls."""
    start_cash   = mstate.get("session_start_cash", price * 10)
    risk_dollars = start_cash * cfg["risk_per_trade"]
    atr_sl_mult  = cfg["atr_sl_mult"]

    if atr and atr > 0 and cfg.get("use_atr_exits", True):
        sl_dist = atr * atr_sl_mult
        tp_dist = atr * cfg["atr_tp_mult"]
        sl_floor = price * cfg["stop_loss_pct"] * 0.5
        sl_cap   = price * cfg["stop_loss_pct"] * 2.0
        tp_floor = price * cfg["target_pct"]    * 0.5
        tp_cap   = price * cfg["target_pct"]    * 2.0
        if cfg.get("settings_enabled", True):
            sl_dist = max(sl_floor, min(sl_cap, sl_dist))
            tp_dist = max(tp_floor, min(tp_cap, tp_dist))
        qty = max(1, int(risk_dollars / sl_dist)) if sl_dist > 0 else 1
    else:
        sl_dist = price * cfg["stop_loss_pct"]
        tp_dist = price * cfg["target_pct"]
        qty     = max(1, int((start_cash * cfg["risk_per_trade"]) / price))

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

def execute_sell(symbol: str, price: float, reason: str, mstate: dict):
    pos = mstate["positions"].get(symbol)
    if not pos:
        return
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
    try:
        from trading_agent.integration.online_learner import record_exit as _rl_exit
        _rl_exit(symbol, price, pnl, pos.get("atr", 0.0), side="long")
    except Exception:
        pass

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
    try:
        from trading_agent.integration.online_learner import record_exit as _rl_exit
        _rl_exit(symbol, price, net_pnl, pos.get("atr", 0.0), side="short")
    except Exception:
        pass

    if reason == "STOP LOSS":
        cooldown_min = cfg.get("cooldown_after_sl_min", 60)
        until = (
            datetime.now(timezone.utc) + timedelta(minutes=cooldown_min)
        ).isoformat(timespec="seconds")
        mstate.setdefault("cooldown_until", {})[symbol] = until
        apex_log.info(f"[COOLDOWN] {symbol} blocked for {cooldown_min}m (short stop-loss triggered)")

# ─── MARKET CYCLE ─────────────────────────────────────────────────────────────

def fetch_cycle_data(watchlist: list, prices: dict, market_key: str = "india") -> tuple:
    """Phase 1: fetch signals (slow, no lock needed).
    In RL mode uses PPO inference on daily bars; falls back to rule-based on failure."""
    analyses = []
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
        # Rule-based fallback
        df = fetch_data(symbol)
        if df is None:
            continue
        live = live or float(df["close"].iloc[-1])
        
        # Task 17-18: Fetch per-symbol JEV decisions (news_bullishness, position_action)
        jev_symbol_decisions = None
        if _JEV_AVAILABLE and cfg.get("jev_enabled", True) and cfg.get("jev_news_enabled", True):
            try:
                # Build symbol-specific state for JEV
                indicators = {
                    "price": live,
                    "rsi": calc_rsi(df["close"].astype(float)) if len(df) > 14 else 50.0,
                    "macd_hist": calc_macd(df["close"].astype(float))[2] if len(df) > 26 else 0.0,
                    "adx": calc_adx(df) if len(df) > 14 else 20.0,
                    "atr": calc_atr(df) if len(df) > 14 else 0.0,
                    "vol_ratio": calc_vol_ratio(df["volume"].astype(float)) if len(df) > 20 else 1.0,
                }
                # Fetch news for this symbol
                news = []
                try:
                    yf_news = yf.Ticker(symbol).news or []
                    recent = [n for n in yf_news if time.time() - float(n.get("providerPublishTime", 0)) < 86400]
                    for item in recent[:5]:
                        news.append({"title": item.get("title", ""), "recency_h": (time.time() - float(item.get("providerPublishTime", 0))) / 3600})
                except Exception:
                    pass
                
                with _lock:
                    mstate = _state.get(market_key, {})
                    portfolio_ctx = {
                        "cash": mstate.get("cash", 0.0),
                        "drawdown_pct": mstate.get("max_drawdown", 0.0),
                        "open_positions": len(mstate.get("positions", {})),
                        "daily_pnl_pct": mstate.get("realised_pnl", 0.0) / max(mstate.get("session_start_cash", 1.0), 1.0),
                    }
                
                market_context = {
                    "spy_trend_pct": _index_trend.get(market_key, 0.0),
                    "vix": 20.0,
                    "breadth": 0.5,
                }
                
                # Get per-symbol JEV decisions (news_bullishness, position_action)
                symbol_state = _jev.build_market_state(symbol, indicators, news, portfolio_ctx, market_context)
                symbol_questions = {
                    "news_bullishness": _jev.JEV_QUESTIONS["news_bullishness"],
                    "position_action": _jev.JEV_QUESTIONS["position_action"],
                }
                symbol_result = _jev._call_jev_api(symbol_state, symbol_questions)
                if symbol_result:
                    jev_symbol_decisions = symbol_result
            except (requests.RequestException, ValueError, KeyError) as e:
                apex_log.debug(f"[JEV] Per-symbol decisions failed for {symbol}: {e}")
        
        analyses.append(analyse(symbol, df, live, jev_symbol_decisions))
        time.sleep(0.2)
    wl_prices = {s: prices[s] for s in watchlist if s in prices}
    return analyses, wl_prices

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
            
            # EOD exit logic still runs
            if mtc is not None:
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
                    elif mtc <= exit_min:
                        tag = "EOD EXIT" if pnl >= 0 else "EOD CUT LOSS"
                        think_log("EXIT", f"{sym} EOD FORCE EXIT ({tag}): {mtc:.0f}m to close  P&L={pnl:+.0f}", sym)
                        if is_short:
                            paper_cover(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
                        else:
                            paper_sell(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
                        apex_log.info(f"EOD force exit: {sym}  P&L={pnl:+.0f}  {mtc:.0f}m left")
            
            # Trailing stop updates
            for sym in list(mstate["positions"].keys()):
                pos = mstate["positions"].get(sym)
                if pos is None or prices.get(sym) is None:
                    continue
                price = prices[sym]
                is_short = pos.get("side", "long") == "short"
                if cfg.get("trailing_stop_enabled", True) and pos.get("atr", 0) > 0:
                    act_mult = cfg.get("trailing_activation_mult", 1.0)
                    dist_mult = _jev.get_trailing_dist_mult(cfg.get("trailing_dist_mult", 1.5), jev_decisions)
                    if not is_short:
                        activation = pos["entry"] + pos["atr"] * act_mult
                        if price >= activation:
                            pos["running_high"] = max(pos.get("running_high", price), price)
                            new_sl = pos["running_high"] - pos["atr"] * dist_mult
                            if new_sl > pos["stop_loss"]:
                                old_sl = pos["stop_loss"]
                                pos["stop_loss"] = round(new_sl, 2)
                                think_log("EXIT", f"{sym} TRAIL ↑ SL {old_sl:.2f}→{pos['stop_loss']:.2f} running_high={pos['running_high']:.2f}", sym)
                    else:
                        activation = pos["entry"] - pos["atr"] * act_mult
                        if price <= activation:
                            pos["running_low"] = min(pos.get("running_low", price), price)
                            new_sl = pos["running_low"] + pos["atr"] * dist_mult
                            if new_sl < pos["stop_loss"]:
                                old_sl = pos["stop_loss"]
                                pos["stop_loss"] = round(new_sl, 2)
                                think_log("EXIT", f"{sym} TRAIL ↓ SL {old_sl:.2f}→{pos['stop_loss']:.2f} running_low={pos['running_low']:.2f}", sym)
                
                # SL/TP checks
                for sym in list(mstate["positions"].keys()):
                    pos = mstate["positions"].get(sym)
                    if pos is None or prices.get(sym) is None:
                        continue
                    price = prices[sym]
                    is_short = pos.get("side", "long") == "short"
                    if not is_short:
                        sl_hit = price <= pos["stop_loss"]
                        tp_hit = price >= pos["target"]
                    else:
                        sl_hit = price >= pos["stop_loss"]
                        tp_hit = price <= pos["target"]
                    if sl_hit:
                        think_log("EXIT", f"{sym} STOP LOSS hit @ {price:.2f}  SL was {pos['stop_loss']:.2f}", sym)
                        if is_short:
                            paper_cover(sym, pos["qty"], prices, mstate, "STOP LOSS")
                        else:
                            paper_sell(sym, pos["qty"], prices, mstate, "STOP LOSS")
                    elif tp_hit:
                        think_log("EXIT", f"{sym} TARGET HIT @ {price:.2f}  T was {pos['target']:.2f}", sym)
                        if is_short:
                            paper_cover(sym, pos["qty"], prices, mstate, "TARGET HIT")
                        else:
                            paper_sell(sym, pos["qty"], prices, mstate, "TARGET HIT")
            
            return

    # ── 1. Update peak portfolio (prerequisite for drawdown kill-switch) ──────
    pv = portfolio_value(mstate, prices)
    if pv > mstate.get("peak_portfolio", pv):
        mstate["peak_portfolio"] = pv

    # ── 2. EOD exit: runs before normal SL/target checks ─────────────────────
    if mtc is not None:
        for sym in list(mstate["positions"].keys()):
            pos   = mstate["positions"].get(sym)
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

            elif mtc <= exit_min:
                tag = "EOD EXIT" if pnl >= 0 else "EOD CUT LOSS"
                think_log("EXIT", f"{sym} EOD FORCE EXIT ({tag}): {mtc:.0f}m to close  P&L={pnl:+.0f}", sym)
                if is_short:
                    paper_cover(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
                else:
                    paper_sell(sym, pos["qty"], prices, mstate, f"{tag} ({mtc:.0f}m to close)")
                apex_log.info(f"EOD force exit: {sym}  P&L={pnl:+.0f}  {mtc:.0f}m left")

    # ── 3. Trailing stop update + normal SL / target checks ──────────────────
    for sym in list(mstate["positions"].keys()):
        pos   = mstate["positions"].get(sym)
        if pos is None or prices.get(sym) is None:
            continue
        price = prices[sym]

        is_short = pos.get("side", "long") == "short"

        # Trailing stop — mirror logic for longs vs shorts
        if cfg.get("trailing_stop_enabled", True) and pos.get("atr", 0) > 0:
            act_mult = cfg.get("trailing_activation_mult", 1.0)
            dist_mult = cfg.get("trailing_dist_mult", 1.5)
            if not is_short:
                # LONG: activate when price rises 1×ATR above entry, SL trails up
                activation = pos["entry"] + pos["atr"] * act_mult
                if price >= activation:
                    pos["running_high"] = max(pos.get("running_high", price), price)
                    new_sl = pos["running_high"] - pos["atr"] * dist_mult
                    if new_sl > pos["stop_loss"]:
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
                # SHORT: activate when price falls 1×ATR below entry, SL trails down
                activation = pos["entry"] - pos["atr"] * act_mult
                if price <= activation:
                    pos["running_low"] = min(pos.get("running_low", price), price)
                    new_sl = pos["running_low"] + pos["atr"] * dist_mult
                    if new_sl < pos["stop_loss"]:   # tighten SL downward
                        old_sl = pos["stop_loss"]
                        pos["stop_loss"] = round(new_sl, 2)
                        think_log("EXIT",
                                  f"{sym} TRAIL ↓ SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                                  f"running_low={pos['running_low']:.2f}", sym)
                        apex_log.debug(
                            f"[TRAIL-SHORT] {sym}  SL {old_sl:.2f}→{pos['stop_loss']:.2f}  "
                            f"low={pos['running_low']:.2f}"
                        )

        # SL / target checks — inverted for shorts
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
            _dc["sell"] += 1
        elif tp_hit:
            think_log("EXIT", f"{sym} TARGET HIT @ {price:.2f}  T was {pos['target']:.2f}", sym)
            if is_short:
                paper_cover(sym, pos["qty"], prices, mstate, "TARGET HIT")
            else:
                paper_sell(sym, pos["qty"], prices, mstate, "TARGET HIT")
            _dc["sell"] += 1
        else:
            pnl_pct = ((pos["entry"] - price) / pos["entry"] * 100 if is_short
                       else (price - pos["entry"]) / pos["entry"] * 100)

            # RL agent gets to vote on early exit while SL/TP not yet hit
            rl_exited = False
            if cfg.get("rl_mode") and _RL_AVAILABLE:
                try:
                    rl_result = _get_rl_signal(sym, price)
                    if rl_result is not None:
                        rl_action = rl_result.get("rl_action", 0)
                        rl_conf   = rl_result.get("confidence", 0)
                        rl_exit_thr = cfg.get("rl_exit_confidence", 55)
                        should_exit = (
                            (not is_short and rl_action == 2 and rl_conf >= rl_exit_thr) or
                            (is_short     and rl_action == 1 and rl_conf >= rl_exit_thr)
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
                            rl_exited = True
                except Exception:
                    pass

            if not rl_exited:
                apex_log.debug(
                    f"[{market_key.upper()}] SL-CHECK HOLD  {sym}  "
                    f"price={price:.2f}  P&L={pnl_pct:+.1f}%  "
                    f"SL={pos['stop_loss']:.2f}  T={pos['target']:.2f}"
                )

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
    if (daily_loss_pct < -cfg.get("daily_loss_limit_pct", 0.05)
            or drawdown_pct > cfg.get("max_drawdown_pct", 0.08)):
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
    in_open_window  = 0 <= mins_since_open < cfg.get("open_filter_min", 15)
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

        # ── LONG entry ───────────────────────────────────────────────────────
        short_enabled    = cfg.get("short_selling_enabled", False)
        bearish_conf     = 100 - conf
        short_threshold  = cfg.get("short_confidence_threshold", 75) if _strict else 60
        index_max_short  = cfg.get("index_max_pct_for_short", 0.5)

        if (conf >= conf_thr
                and a["score"] > 0
                and (a.get("rl_action") is not None or index_ok)  # RL bypasses index gate
                and open_pos < max_pos_eff
                and mstate["cash"] > a["price"] * 2):
            think_log("ENTRY",
                      f"{sym} ENTRY: conf={conf:.0f}%  score={a['score']:+d}  "
                      f"@ {a['price']:.2f}  ADX={adx_val:.1f}  ATR={a.get('atr', 0):.3f}", sym)
            if paper_buy(sym, 0, prices, mstate, atr=a.get("atr")):
                open_pos += 1
                _dc["buy"] += 1
                # Online learning: record obs+action for any trade where RL built
                # an observation (RL mode runs _build_observation for all symbols
                # even when the entropy gate rejects the signal). For rule-based
                # fallback trades we use action=1 (BUY) to teach the model the outcome.
                try:
                    from trading_agent.integration.rl_signal import get_cached_obs as _get_obs
                    from trading_agent.integration.online_learner import record_entry as _rl_entry
                    _obs = _get_obs(sym)
                    if _obs is not None:
                        _rl_entry(sym, _obs, a.get("rl_action", 1), a["price"], a.get("atr", 0.0))
                except Exception:
                    pass

        # ── SHORT entry ──────────────────────────────────────────────────────
        elif (short_enabled
                and a.get("rl_action") is None      # RL agent is long-only
                and not in_open_window              # too volatile at open; skip shorts
                and bearish_conf >= short_threshold
                and a["score"] < 0
                and index_pct <= index_max_short    # block shorts on strongly bullish days
                and open_pos < max_pos_eff
                and mstate["cash"] > a["price"] * 2):
            think_log("ENTRY",
                      f"{sym} SHORT ENTRY: bearish_conf={bearish_conf:.0f}%  score={a['score']:+d}  "
                      f"@ {a['price']:.2f}  ADX={adx_val:.1f}  ATR={a.get('atr', 0):.3f}", sym)
            if paper_short(sym, 0, prices, mstate, atr=a.get("atr")):
                open_pos += 1
                _dc["buy"] += 1
                try:
                    from trading_agent.integration.rl_signal import get_cached_obs as _get_obs
                    from trading_agent.integration.online_learner import record_entry as _rl_entry
                    _obs = _get_obs(sym)
                    if _obs is not None:
                        _rl_entry(sym, _obs, 2, a["price"], a.get("atr", 0.0))  # 2=SELL
                except Exception:
                    pass

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
        + (f"  [OPEN WINDOW {mins_since_open:.0f}m/{cfg['open_filter_min']}m]"
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

def paper_buy(symbol: str, qty: int, latest_prices: dict, mstate: dict, atr: float = None):
    """Paper-trade a buy using the price from the snapshot.
    qty is advisory; actual qty is computed from risk settings inside execute_buy."""
    price = latest_prices.get(symbol)
    if price is None:
        apex_log.warning(f"paper_buy: no price for {symbol} in snapshot — skipping")
        return None
    return execute_buy(symbol, price, mstate, atr=atr)

def paper_sell(symbol: str, qty: int, latest_prices: dict, mstate: dict, reason: str = "SIGNAL"):
    """Paper-trade a sell using the price from the snapshot (falls back to live fetch)."""
    price = latest_prices.get(symbol) or get_price(symbol)
    if price is None:
        apex_log.warning(f"paper_sell: no price for {symbol} — skipping")
        return
    execute_sell(symbol, price, reason, mstate)

def paper_short(symbol: str, qty: int, latest_prices: dict, mstate: dict, atr: float = None):
    """Paper-trade a short: sell-to-open at snapshot price."""
    price = latest_prices.get(symbol)
    if price is None:
        apex_log.warning(f"paper_short: no price for {symbol} — skipping")
        return None
    return execute_short(symbol, price, mstate, atr=atr)

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
        india_pv   = portfolio_value(_state["india"], latest_prices)
        us_pv      = portfolio_value(_state["us"],    latest_prices)
        india_upnl = unrealised_pnl(_state["india"],  latest_prices)
        us_upnl    = unrealised_pnl(_state["us"],     latest_prices)
        return {
            "india": {
                "cash":            round(_state["india"]["cash"], 2),
                "portfolio_value": round(india_pv, 2),
                "unrealised_pnl":  round(india_upnl, 2),
                "realised_pnl":    round(_state["india"]["realised_pnl"], 2),
                "open_positions":  len(_state["india"]["positions"]),
                "wins":            _state["india"]["wins"],
                "losses":          _state["india"]["losses"],
            },
            "us": {
                "cash":            round(_state["us"]["cash"], 2),
                "portfolio_value": round(us_pv, 2),
                "unrealised_pnl":  round(us_upnl, 2),
                "realised_pnl":    round(_state["us"]["realised_pnl"], 2),
                "open_positions":  len(_state["us"]["positions"]),
                "wins":            _state["us"]["wins"],
                "losses":          _state["us"]["losses"],
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
    with _lock:
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

    while _agent["running"]:
        if _agent["paused"]:
            _agent["status"] = "paused"
            time.sleep(5)
            continue

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
        with _lock:
            if _prev_india_open and not india_open:
                apex_log.info("[SESSION] India market just closed — auto-archiving session")
                _close_session("india", prices_snapshot)
                save_state(_state)
            if _prev_us_open and not us_open:
                apex_log.info("[SESSION] US market just closed — auto-archiving session")
                _close_session("us", prices_snapshot)
                save_state(_state)

        _prev_india_open = india_open
        _prev_us_open    = us_open

        # Refresh index trend once per cycle (no lock needed — floats are atomic)
        _refresh_index_trend()

        # ── JEV Risk Gates: Apply per-market config overrides ──────────────────────
        jev_decisions_per_market = {}
        if _JEV_AVAILABLE and cfg.get("jev_enabled", True):
            market_context = {
                "spy_trend_pct": _index_trend.get("india", 0.0) if india_open else _index_trend.get("us", 0.0),
                "vix": 20.0,
                "breadth": 0.5,
            }
            
            for market_key, is_open in [("india", india_open), ("us", us_open)]:
                if not is_open:
                    continue
                try:
                    with _lock:
                        mstate = _state.get(market_key, {})
                        portfolio_ctx = {
                            "cash": mstate.get("cash", 0.0),
                            "drawdown_pct": mstate.get("max_drawdown", 0.0),
                            "open_positions": len(mstate.get("positions", {})),
                            "daily_pnl_pct": mstate.get("realised_pnl", 0.0) / max(mstate.get("session_start_cash", 1.0), 1.0),
                        }
                    
                    jev_decisions = _jev.get_system_decisions(market_key, portfolio_ctx, market_context)
                    
                    if jev_decisions:
                        jev_decisions_per_market[market_key] = jev_decisions
                        
                        # Apply JEV gates to config for this market
                        _jev.apply_jev_gates(cfg, jev_decisions)
                        
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
                            f"open_window={cfg.get('open_filter_min', 15)}m"
                        )
                except (requests.RequestException, ValueError, KeyError) as e:
                    apex_log.warning(f"[JEV] Risk gates failed for {market_key}: {e}")

        if india_open:
            if not _agent["running"]:
                break
            _agent["status"] = "Scanning India"
            apex_log.info("Scanning India watchlist…")
            india_analyses, india_prices = fetch_cycle_data(INDIA_WATCHLIST, prices_snapshot, "india")
            apex_log.info(
                f"India scan done: {len(india_analyses)}/{len(INDIA_WATCHLIST)} symbols  "
                f"prices={len(india_prices)}"
            )
            with _lock:
                apply_cycle("india", india_analyses, india_prices, cfg["india_max_positions"], jev_decisions_per_market)
                _signals["india"]        = india_analyses
                _signals["india_prices"] = india_prices
        else:
            apex_log.debug("India market closed — skipping")

        if us_open:
            if not _agent["running"]:
                break
            _agent["status"] = "Scanning US"
            apex_log.info("Scanning US watchlist…")
            us_analyses, us_prices = fetch_cycle_data(US_WATCHLIST, prices_snapshot, "us")
            apex_log.info(
                f"US scan done: {len(us_analyses)}/{len(US_WATCHLIST)} symbols  "
                f"prices={len(us_prices)}"
            )
            with _lock:
                apply_cycle("us", us_analyses, us_prices, cfg["us_max_positions"], jev_decisions_per_market)
                _signals["us"]        = us_analyses
                _signals["us_prices"] = us_prices
        else:
            apex_log.debug("US market closed — skipping")

        with _lock:
            save_state(_state)
        apex_log.info("State saved")

        _agent["last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sleep_min = cfg["check_interval_min"] if (india_open or us_open) else cfg["idle_interval_min"]
        if _agent["running"]:
            _agent["status"] = "waiting"
        next_t = (datetime.now(timezone.utc) + timedelta(minutes=sleep_min)).isoformat(timespec="seconds")
        _agent["next_check"] = next_t
        apex_log.info(f"Waiting {sleep_min} min — next check at {next_t}")

        # Sleep in 5-second ticks so stop/pause respond quickly
        for _ in range(sleep_min * 12):
            if not _agent["running"]:
                break
            time.sleep(5)

    _agent["status"]  = "stopped"
    _agent["running"] = False
    apex_log.info("Agent stopped")

# ─── FLASK APP ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── AUTH ────────────────────────────────────────────────────────────────────
import secrets as _secrets
app.secret_key        = os.environ.get("APEX_SECRET", _secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
_AUTH_USER = os.environ.get("APEX_USER", "apex")
_AUTH_PASS = os.environ.get("APEX_PASS", "admin")

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
        if (request.form.get("u", "") == _AUTH_USER and
                request.form.get("p", "") == _AUTH_PASS):
            session["logged_in"] = True
            apex_log.info("Dashboard login from " + (request.remote_addr or "?"))
            return redirect("/")
        err = "Invalid credentials — try again"
        apex_log.warning("Failed login attempt from " + (request.remote_addr or "?"))
    return render_template_string(LOGIN_HTML, err=err)

@app.route("/logout")
def logout():
    apex_log.info("Dashboard logout")
    session.clear()
    return redirect("/login")

@app.route("/")
def index():
    return render_template_string(HTML)

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
    india_pv   = portfolio_value(st["india"], merged_india)
    us_pv      = portfolio_value(st["us"],    merged_us)
    india_upnl = unrealised_pnl(st["india"], merged_india)
    us_upnl    = unrealised_pnl(st["us"],    merged_us)

    def dd_calc(mstate, pv):
        peak = mstate.get("peak_portfolio", pv)
        return round((peak - pv) / peak * 100, 2) if peak > 0 else 0.0

    # Get JEV decisions for dashboard
    jev_decisions = {}
    if _JEV_AVAILABLE and cfg.get("jev_enabled", True):
        market_context = {
            "spy_trend_pct": _index_trend.get("india", 0.0),
            "vix": 20.0,
            "breadth": 0.5,
        }
        for market_key in ("india", "us"):
            try:
                with _lock:
                    mstate = _state.get(market_key, {})
                    portfolio_ctx = {
                        "cash": mstate.get("cash", 0.0),
                        "drawdown_pct": mstate.get("max_drawdown", 0.0),
                        "open_positions": len(mstate.get("positions", {})),
                        "daily_pnl_pct": mstate.get("realised_pnl", 0.0) / max(mstate.get("session_start_cash", 1.0), 1.0),
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
            **st["india"],
            "portfolio_value": round(india_pv, 2),
            "unrealised_pnl":  round(india_upnl, 2),
            "total_pnl":       round(st["india"]["realised_pnl"] + india_upnl, 2),
            "drawdown":        dd_calc(st["india"], india_pv),
            "market_open":     is_india_open(),
        },
        "us": {
            **st["us"],
            "portfolio_value": round(us_pv, 2),
            "unrealised_pnl":  round(us_upnl, 2),
            "total_pnl":       round(st["us"]["realised_pnl"] + us_upnl, 2),
            "drawdown":        dd_calc(st["us"], us_pv),
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
        save_state(_state)
        apex_log.info(f"Manual close: {symbol} @ {price:.2f}  market={market}  side={pos.get('side','long')}")
    return jsonify({"ok": True, "msg": f"Sold {symbol.replace('.NS', '')}"})

@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.json or {}
    int_keys = {"india_max_positions", "us_max_positions",
                "check_interval_min",  "idle_interval_min",
                "confidence_threshold", "eod_harvest_min", "eod_exit_min"}
    changed = []
    for k in ("risk_per_trade", "confidence_threshold", "stop_loss_pct", "target_pct",
              "check_interval_min", "idle_interval_min",
              "india_max_positions", "us_max_positions",
              "eod_harvest_min", "eod_exit_min",
              "rl_exit_confidence"):
        if k in data:
            cfg[k] = int(data[k]) if k in int_keys else float(data[k])
            changed.append(k)
    if "settings_enabled" in data:
        cfg["settings_enabled"] = bool(data["settings_enabled"])
        changed.append("settings_enabled")
    save_cfg()
    apex_log.info(f"Config updated: {', '.join(changed)}")
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
    try:
        from trading_agent.integration.online_learner import (
            get_retrain_log, _new_count, _is_training, _buffer, _total_updates,
        )
        return jsonify({
            "log":           get_retrain_log(),
            "is_training":   _is_training,
            "new_count":     _new_count,
            "buffer_size":   len(_buffer),
            "total_updates": _total_updates,
            "threshold":     16,
        })
    except ImportError:
        return jsonify({
            "log": [], "is_training": False,
            "new_count": 0, "buffer_size": 0, "total_updates": 0, "threshold": 16,
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
        save_state(_state)
    apex_log.warning(f"State reset: market={market}")
    return jsonify({"ok": True, "msg": f"Reset {market}"})

@app.route("/api/edit/state", methods=["POST"])
def edit_state():
    data = request.json or {}
    changed = []
    with _lock:
        for mkt in ("india", "us"):
            if mkt not in data:
                continue
            mstate = _state[mkt]
            md = data[mkt]
            for field, cast in [("cash", float), ("realised_pnl", float),
                                 ("wins", int), ("losses", int),
                                 ("peak_portfolio", float), ("max_drawdown", float)]:
                if field in md:
                    mstate[field] = cast(md[field])
                    changed.append(f"{mkt}.{field}")
        save_state(_state)
    apex_log.info(f"Manual state edit: {', '.join(changed)}")
    return jsonify({"ok": True, "changed": changed})

@app.route("/api/edit/position/<market>/<path:symbol>", methods=["POST"])
def edit_position(market, symbol):
    if market not in ("india", "us"):
        return jsonify({"ok": False, "msg": "Invalid market"})
    data = request.json or {}
    with _lock:
        pos = _state.get(market, {}).get("positions", {}).get(symbol)
        if pos is None:
            return jsonify({"ok": False, "msg": "Position not found"})
        for field, cast in [("qty", int), ("entry", float),
                             ("stop_loss", float), ("target", float)]:
            if field in data:
                pos[field] = cast(data[field])
        save_state(_state)
    apex_log.info(f"Manual position edit: {symbol} ({market})")
    return jsonify({"ok": True})

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
        save_state(_state)
    apex_log.info(f"Manual session close: {market}")
    return jsonify({"ok": True, "msg": f"{market} session archived and reset"})

# ─── HTML / CSS / JS TEMPLATE ─────────────────────────────────────────────────

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apex — Sign In</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0d1117;color:#e6edf3;display:flex;align-items:center;
     justify-content:center;min-height:100vh}
.box{background:#161b22;border:1px solid #30363d;border-radius:10px;
     padding:36px 40px;width:340px}
.logo{font-size:20px;font-weight:700;letter-spacing:2px;color:#58a6ff;
      text-align:center;margin-bottom:28px}
.logo b{color:#f0883e}
label{font-size:10px;color:#8b949e;text-transform:uppercase;
      letter-spacing:.5px;display:block;margin-bottom:5px}
input[type=text],input[type=password]{background:#1c2128;border:1px solid #30363d;
      border-radius:6px;color:#e6edf3;padding:9px 12px;font-size:14px;width:100%;
      margin-bottom:16px;transition:border-color .2s}
input:focus{outline:none;border-color:#58a6ff}
.err{color:#f85149;font-size:11px;text-align:center;margin-bottom:14px;
     background:#200a0a;border:1px solid #5a1a1a;border-radius:5px;padding:7px 10px}
.btn{width:100%;padding:10px;border-radius:6px;cursor:pointer;font-size:13px;
     font-weight:700;letter-spacing:.4px;background:#0d1e3a;color:#58a6ff;
     border:1px solid #1a3a6a;transition:filter .15s}
.btn:hover{filter:brightness(1.2)}
.sub{font-size:10px;color:#444c56;text-align:center;margin-top:18px}
</style>
</head>
<body>
<div class="box">
  <div class="logo">APEX <b>▲</b></div>
  {% if err %}<div class="err">{{ err }}</div>{% endif %}
  <form method="POST" autocomplete="off">
    <label>Username</label>
    <input type="text" name="u" autofocus autocomplete="username">
    <label>Password</label>
    <input type="password" name="p" autocomplete="current-password">
    <button class="btn" type="submit">Sign In</button>
  </form>
  <div class="sub">Apex Paper Trading — restricted access</div>
</div>
</body>
</html>"""

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apex Trading Dashboard</title>
<style>
:root{
  --bg:#0d1117;--card:#161b22;--card2:#1c2128;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;
  --green:#3fb950;--red:#f85149;--yellow:#d29922;
  --blue:#58a6ff;--orange:#f0883e;--purple:#bc8cff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--bg);color:var(--text);font-size:13px;line-height:1.5}

/* ── Header ── */
.hdr{background:var(--card);border-bottom:1px solid var(--border);
     padding:10px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;position:sticky;top:0;z-index:100}
.logo{font-size:15px;font-weight:700;letter-spacing:2px;color:var(--blue)}
.logo b{color:var(--orange)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
#clock{font-family:monospace;color:var(--muted);font-size:12px}

/* ── Badges ── */
.badge{padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;
       letter-spacing:.6px;white-space:nowrap}
.b-open  {background:#0d2318;color:var(--green);border:1px solid #1a4a2a}
.b-closed{background:#1e1a0a;color:var(--yellow);border:1px solid #3a2f10}
.b-run   {background:#0d1e3a;color:var(--blue);border:1px solid #1a3a6a;animation:pulse 2s infinite}
.b-pause {background:#2a1800;color:var(--orange);border:1px solid #5a3500}
.b-idle  {background:var(--card2);color:var(--muted);border:1px solid var(--border)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}

/* ── Buttons ── */
.btn{padding:5px 14px;border-radius:6px;border:1px solid;cursor:pointer;
     font-size:11px;font-weight:700;letter-spacing:.4px;transition:filter .15s}
.btn:hover{filter:brightness(1.2)}
.btn-green{background:#0d2318;color:var(--green);border-color:#1a4a2a}
.btn-red  {background:#2a0d0d;color:var(--red);border-color:#5a1a1a}
.btn-blue {background:#0d1e3a;color:var(--blue);border-color:#1a3a6a}
.btn-muted{background:var(--card2);color:var(--muted);border-color:var(--border)}
.btn-sm{padding:3px 9px;font-size:10px}

/* ── Layout ── */
.main{padding:14px 18px;max-width:1700px;margin:0 auto}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
.g5{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}

/* ── Cards ── */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}
.c-title{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:4px}
.c-val{font-size:20px;font-weight:700;font-family:monospace}
.c-sub{font-size:11px;margin-top:1px}

/* ── Market section header ── */
.mkt-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.mkt-name{font-size:13px;font-weight:700}

/* ── Section divider ── */
.sec{display:flex;align-items:center;gap:8px;margin:14px 0 8px;
     padding-bottom:5px;border-bottom:1px solid var(--border)}
.sec-title{font-size:12px;font-weight:600;letter-spacing:.4px}
.sec-badge{font-size:10px;padding:1px 7px;border-radius:10px;
           background:var(--card2);color:var(--muted);border:1px solid var(--border)}

/* ── Tabs ── */
.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:14px}
.tab{padding:7px 16px;cursor:pointer;font-size:12px;font-weight:500;
     border-bottom:2px solid transparent;color:var(--muted);transition:all .15s}
.tab:hover{color:var(--text)}
.tab.on{color:var(--blue);border-bottom-color:var(--blue)}
.pane{display:none}.pane.on{display:block}

/* ── Tables ── */
.tbl-wrap{overflow-x:auto;border-radius:6px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px}
th{padding:6px 10px;text-align:left;color:var(--muted);font-weight:500;
   font-size:10px;text-transform:uppercase;letter-spacing:.5px;
   border-bottom:1px solid var(--border);background:var(--card2);white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid #21262d;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1c2330}

/* ── Badges inside table ── */
.act-buy {display:inline-block;padding:1px 8px;border-radius:4px;font-size:10px;font-weight:700;
          background:#0d2318;color:var(--green);border:1px solid #1a4a2a}
.act-sell{display:inline-block;padding:1px 8px;border-radius:4px;font-size:10px;font-weight:700;
          background:#2a0d0d;color:var(--red);border:1px solid #5a1a1a}
.act-hold{display:inline-block;padding:1px 8px;border-radius:4px;font-size:10px;font-weight:700;
          background:var(--card2);color:var(--muted);border:1px solid var(--border)}

/* ── Confidence bar ── */
.cbar{display:inline-flex;align-items:center;gap:5px}
.cbar-bg{width:48px;height:4px;border-radius:2px;background:var(--card2);overflow:hidden}
.cbar-fg{height:100%;border-radius:2px}

/* ── Trade log ── */
.log-box{background:var(--card);border:1px solid var(--border);border-radius:6px;
         max-height:200px;overflow-y:auto}
.log-row{padding:5px 10px;border-bottom:1px solid #21262d;
         font-family:monospace;font-size:11px;display:flex;gap:10px}
.log-row:last-child{border-bottom:none}
.lt{color:var(--muted);min-width:62px}
.lb{color:var(--green)}.ls{color:var(--blue)}.ll{color:var(--red)}

/* ── Agent system log ── */
.alog-box{background:#0a0e14;border:1px solid var(--border);border-radius:6px;
          max-height:400px;overflow-y:auto}
.alog-row{padding:4px 10px;border-bottom:1px solid #161b22;
          font-family:monospace;font-size:11px;display:flex;gap:8px;align-items:baseline}
.alog-row:last-child{border-bottom:none}
.al-ts{color:#444c56;min-width:120px;font-size:10px}
.al-lvl{min-width:52px;font-size:9px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}
.al-info{color:var(--blue)}.al-debug{color:#444c56}.al-warning{color:var(--yellow)}.al-error{color:var(--red)}

/* ── Decision / thinking log ── */
.dlog-box{background:#0a0e14;border:1px solid var(--border);border-radius:6px;
          max-height:420px;overflow-y:auto}
.drow{padding:3px 10px;border-bottom:1px solid #161b22;
      font-family:monospace;font-size:11px;display:flex;gap:8px;align-items:baseline;flex-wrap:nowrap}
.drow:last-child{border-bottom:none}
.drow:hover{background:#12161f}
.dcat{min-width:48px;font-size:9px;font-weight:700;letter-spacing:.4px;
      text-transform:uppercase;padding:1px 5px;border-radius:3px;white-space:nowrap}
.dcat-CYCLE {background:#1e1e1e;color:#666}
.dcat-SCAN  {background:#0d2040;color:#4a9eff}
.dcat-FILTER{background:#2e1800;color:#f0900a}
.dcat-ENTRY {background:#0a2016;color:#27c46b}
.dcat-EXIT  {background:#2a0a0a;color:#e05454}
.dcat-RISK  {background:#3a0000;color:#ff4444;font-weight:900}
.dcat-INDEX {background:#1a0a30;color:#b47fff}
.dcat-RL    {background:#1a0040;color:#c060ff;font-weight:900}
.dsym{color:#bbb;min-width:88px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dmsg{color:#ddd;flex:1;white-space:pre-wrap;word-break:break-word}
/* Position side badges */
.side-badge{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;vertical-align:middle;letter-spacing:.5px}
.side-long {background:#0a2016;color:#27c46b;border:1px solid #1a4a2a}
.side-short{background:#2a0a0a;color:#e05454;border:1px solid #4a1a1a}

/* ── Settings form ── */
.fg{display:flex;flex-direction:column;gap:4px}
label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
input[type=number]{background:var(--card2);border:1px solid var(--border);
  border-radius:5px;color:var(--text);padding:5px 9px;font-size:13px;
  font-family:monospace;width:100%;transition:border-color .2s}
input:focus{outline:none;border-color:var(--blue)}
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}

/* ── Footer / refresh bar ── */
.foot{display:flex;align-items:center;gap:10px;padding:8px 0;
      font-size:10px;color:var(--muted);border-top:1px solid var(--border);margin-top:14px}
#prog{flex:1;height:3px;background:var(--card2);border-radius:2px;overflow:hidden}
#prog-f{height:100%;background:var(--blue);width:0;transition:width linear}

/* ── Toast ── */
.toasts{position:fixed;bottom:18px;right:18px;display:flex;flex-direction:column;gap:7px;z-index:9999}
.toast{background:var(--card2);border:1px solid var(--border);border-radius:7px;
       padding:9px 14px;font-size:12px;animation:tin .25s ease;
       display:flex;align-items:center;gap:8px;max-width:300px}
.tok{border-left:3px solid var(--green)}.terr{border-left:3px solid var(--red)}
@keyframes tin{from{transform:translateX(110%);opacity:0}to{transform:translateX(0);opacity:1}}

.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}
.blue{color:var(--blue)}.muted{color:var(--muted)}.mono{font-family:monospace}

/* ── EOD ── */
.b-eod-warn{background:#1a1000;color:var(--orange);border:1px solid #5a3500}
.b-eod-exit{background:#2a0d0d;color:var(--red);border:1px solid #5a1a1a}
.eod-banner{border-radius:6px;padding:8px 14px;margin-bottom:8px;font-size:11px;
            display:flex;align-items:center;gap:10px;border:1px solid}
.eod-warn{background:#1a1000;border-color:#5a3500}
.eod-exit{background:#200a0a;border-color:#6a1a1a}

/* ── Log filter ── */
.log-filter-btn.on{background:#0d1e3a;color:var(--blue);border-color:#1a3a6a}

/* ── Mobile responsive ── */
@media (max-width:640px){
  /* Main padding */
  .main{padding:10px 10px}

  /* Header: hide clock to save room, tighten gaps */
  .hdr{padding:8px 10px;gap:6px}
  #clock{display:none}
  .hdr-right{gap:5px}
  .logo{font-size:13px}

  /* Market summary: stack India / US vertically */
  .g2{grid-template-columns:1fr}

  /* Stats row: 3 columns instead of 6 */
  .g5{grid-template-columns:repeat(3,1fr)}

  /* Smaller stat values so they fit in 3-col */
  .c-val{font-size:16px}

  /* Tabs: horizontal scroll, no wrapping */
  .tabs{overflow-x:auto;flex-wrap:nowrap;-webkit-overflow-scrolling:touch;
        scrollbar-width:none}
  .tabs::-webkit-scrollbar{display:none}
  .tab{white-space:nowrap;flex-shrink:0;padding:7px 12px}

  /* Settings: single-column forms */
  .fgrid{grid-template-columns:1fr}
  .es-grid{grid-template-columns:1fr}

  /* Modal: full-width on small screens */
  .modal-box{min-width:unset;width:calc(100vw - 24px);padding:16px}

  /* Agent log timestamp: slightly narrower */
  .al-ts{min-width:90px}

  /* Toasts: keep inside viewport */
  .toasts{right:10px;bottom:10px;max-width:calc(100vw - 20px)}
  .toast{max-width:100%}
}

/* Extra-narrow phones (< 400px): 2-col stats */
@media (max-width:400px){
  .g5{grid-template-columns:repeat(2,1fr)}
  .btn{padding:4px 8px;font-size:10px}
}

/* ── Position / state edit modal ── */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);
               z-index:600;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal-box{background:var(--card);border:1px solid var(--border);border-radius:10px;
           padding:24px;min-width:340px;max-width:600px;width:100%;max-height:90vh;overflow-y:auto}
.modal-title{font-size:13px;font-weight:700;margin-bottom:18px;color:var(--text)}
.modal-title span{color:var(--blue)}
.es-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.es-col-title{font-size:11px;font-weight:700;color:var(--muted);margin-bottom:10px;
              text-transform:uppercase;letter-spacing:.5px;padding-bottom:5px;
              border-bottom:1px solid var(--border)}

/* ── Session cards ── */
.sess-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
           margin-bottom:8px;overflow:hidden}
.sess-hdr{display:flex;align-items:center;gap:12px;padding:10px 14px;
          cursor:pointer;user-select:none;transition:background .1s}
.sess-hdr:hover{background:var(--card2)}
.sess-date{font-size:13px;font-weight:700;font-family:monospace;min-width:90px;color:var(--text)}
.sess-mkt{font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px;
          min-width:48px;text-align:center;white-space:nowrap}
.sess-mkt-india{background:#0d1e3a;color:var(--blue);border:1px solid #1a3a6a}
.sess-mkt-us{background:#1a0d2a;color:var(--purple);border:1px solid #3a1a5a}
.sess-body{display:none;border-top:1px solid var(--border);background:#0a0e14}
.sess-body.open{display:block}
.sess-metrics{display:flex;gap:24px;padding:10px 14px;flex-wrap:wrap;
              border-bottom:1px solid var(--border)}
.sess-m{display:flex;flex-direction:column;gap:2px}
.sess-m-l{font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
.sess-m-v{font-size:13px;font-weight:600;font-family:monospace}
.sess-filter.on{background:#0d1e3a;color:var(--blue);border-color:#1a3a6a}
.sess-label{font-size:10px;color:var(--muted);margin-top:2px}
</style>
</head>
<body>

<div class="hdr">
  <div class="logo">APEX <b>▲</b></div>
  <span id="nse-badge"    class="badge b-closed">NSE CLOSED</span>
  <span id="nyse-badge"   class="badge b-closed">NYSE CLOSED</span>
  <span id="india-eod"   class="badge" style="display:none">NSE EOD</span>
  <span id="us-eod"      class="badge" style="display:none">NYSE EOD</span>
  <span id="agent-badge" class="badge b-idle">IDLE</span>
  <div class="hdr-right">
    <span id="clock">—</span>
    <button class="btn btn-muted btn-sm" onclick="doRefresh()">↻ Refresh</button>
    <button id="btn-pause"  class="btn btn-muted btn-sm" onclick="pauseAgent()">⏸ Pause</button>
    <button id="btn-toggle" class="btn btn-green"         onclick="toggleAgent()">▶ Start Agent</button>
    <button class="btn btn-muted btn-sm" onclick="location.href='/logout'" title="Sign out">⎋ Logout</button>
  </div>
</div>

<div class="main">

  <!-- ── Market summary cards ── -->
  <div class="g2">
    <!-- India -->
    <div class="card">
      <div class="mkt-hdr">
        <div>
          <span class="mkt-name">India — NSE / NIFTY 50</span>
          <div class="sess-label" id="india-sess-lbl">Session —</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <button class="btn btn-muted btn-sm" title="Archive current session and reset"
            onclick="closeSessionNow('india')">End Session</button>
          <span id="india-mkt" class="badge b-closed">CLOSED</span>
        </div>
      </div>
      <div class="g5">
        <div><div class="c-title">Portfolio</div>
          <div class="c-val mono" id="india-pv">₹0</div>
          <div class="c-sub" id="india-diff">—</div></div>
        <div><div class="c-title">Cash</div>
          <div class="c-val mono" id="india-cash">₹0</div>
          <div class="c-sub muted" id="india-npos">0 positions</div></div>
        <div><div class="c-title">Realised P&amp;L</div>
          <div class="c-val mono" id="india-pnl">₹0</div>
          <div class="c-sub muted" id="india-ntrades">0 trades</div></div>
        <div><div class="c-title">Win Rate</div>
          <div class="c-val mono" id="india-wr">0%</div>
          <div class="c-sub muted" id="india-wl">0W 0L</div></div>
        <div><div class="c-title">Drawdown</div>
          <div class="c-val mono" id="india-dd">0.0%</div>
          <div class="c-sub muted">from peak</div></div>
        <div style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:10px">
          <div class="c-title">Total Profit</div>
          <div class="c-val mono" id="india-tpnl">₹0</div>
          <div class="c-sub" id="india-upnl" style="font-size:10px">Float: ₹0</div></div>
      </div>
    </div>
    <!-- US -->
    <div class="card">
      <div class="mkt-hdr">
        <div>
          <span class="mkt-name">US — NYSE / NASDAQ</span>
          <div class="sess-label" id="us-sess-lbl">Session —</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <button class="btn btn-muted btn-sm" title="Archive current session and reset"
            onclick="closeSessionNow('us')">End Session</button>
          <span id="us-mkt" class="badge b-closed">CLOSED</span>
        </div>
      </div>
      <div class="g5">
        <div><div class="c-title">Portfolio</div>
          <div class="c-val mono" id="us-pv">$0</div>
          <div class="c-sub" id="us-diff">—</div></div>
        <div><div class="c-title">Cash</div>
          <div class="c-val mono" id="us-cash">$0</div>
          <div class="c-sub muted" id="us-npos">0 positions</div></div>
        <div><div class="c-title">Realised P&amp;L</div>
          <div class="c-val mono" id="us-pnl">$0</div>
          <div class="c-sub muted" id="us-ntrades">0 trades</div></div>
        <div><div class="c-title">Win Rate</div>
          <div class="c-val mono" id="us-wr">0%</div>
          <div class="c-sub muted" id="us-wl">0W 0L</div></div>
        <div><div class="c-title">Drawdown</div>
          <div class="c-val mono" id="us-dd">0.0%</div>
          <div class="c-sub muted">from peak</div></div>
        <div style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:10px">
          <div class="c-title">Total Profit</div>
          <div class="c-val mono" id="us-tpnl">$0</div>
          <div class="c-sub" id="us-upnl" style="font-size:10px">Float: $0</div></div>
      </div>
    </div>
  </div>

  <!-- ── Tabs ── -->
  <div class="tabs">
    <div class="tab on"  onclick="tab('india',this)">India (NSE)</div>
    <div class="tab"     onclick="tab('us',this)">US (NYSE)</div>
    <div class="tab"     onclick="tab('sessions',this)">Sessions <span id="sess-tab-cnt" style="font-size:9px;color:var(--muted)"></span></div>
    <div class="tab"     onclick="tab('settings',this)">Settings</div>
    <div class="tab"     onclick="tab('logs',this)">All Logs</div>
    <div class="tab"     onclick="tab('agent',this)" style="color:#c060ff">Agent Terminal</div>
    <div class="tab"     onclick="tab('retrain',this)" style="color:#f0883e">Retraining</div>
  </div>

  <!-- India pane -->
  <div class="pane on" id="pane-india">
    <div class="sec"><span class="sec-title">Open Positions</span><span class="sec-badge" id="india-pc">0</span></div>
    <div id="india-eod-banner" style="display:none" class="eod-banner eod-warn">
      <span id="india-eod-phase" style="font-weight:700;color:var(--orange)">EOD</span>
      <span id="india-eod-detail" style="color:var(--muted)"></span>
    </div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>Symbol</th><th>Qty</th><th>Entry ₹</th><th>Current ₹</th>
        <th>P&amp;L</th><th>Stop ₹</th><th>Target ₹</th><th>Since</th><th>EOD</th><th></th>
      </tr></thead>
      <tbody id="india-pos"><tr><td colspan="10" style="text-align:center;padding:18px;color:var(--muted)">No open positions</td></tr></tbody>
    </table></div>

    <div class="sec" style="margin-top:18px"><span class="sec-title">Signal Board</span><span class="sec-badge">NIFTY 50 Top 16</span></div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>Symbol</th><th>Price ₹</th><th>RSI</th><th>MACD</th>
        <th>Bollinger</th><th>EMA</th><th>Volume</th><th>4d Trend</th><th>News</th><th>Confidence</th><th>Action</th>
      </tr></thead>
      <tbody id="india-sig"><tr><td colspan="11" style="text-align:center;padding:18px;color:var(--muted)">No scan data yet — start the agent</td></tr></tbody>
    </table></div>

    <div class="sec" style="margin-top:18px"><span class="sec-title">Recent Trades</span></div>
    <div class="log-box" id="india-log">
      <div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>
    </div>
  </div>

  <!-- US pane -->
  <div class="pane" id="pane-us">
    <div class="sec"><span class="sec-title">Open Positions</span><span class="sec-badge" id="us-pc">0</span></div>
    <div id="us-eod-banner" style="display:none" class="eod-banner eod-warn">
      <span id="us-eod-phase" style="font-weight:700;color:var(--orange)">EOD</span>
      <span id="us-eod-detail" style="color:var(--muted)"></span>
    </div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>Symbol</th><th>Qty</th><th>Entry $</th><th>Current $</th>
        <th>P&amp;L</th><th>Stop $</th><th>Target $</th><th>Since</th><th>EOD</th><th></th>
      </tr></thead>
      <tbody id="us-pos"><tr><td colspan="10" style="text-align:center;padding:18px;color:var(--muted)">No open positions</td></tr></tbody>
    </table></div>

    <div class="sec" style="margin-top:18px"><span class="sec-title">Signal Board</span><span class="sec-badge">Top 16 Tech</span></div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th>Symbol</th><th>Price $</th><th>RSI</th><th>MACD</th>
        <th>Bollinger</th><th>EMA</th><th>Volume</th><th>4d Trend</th><th>News</th><th>Confidence</th><th>Action</th>
      </tr></thead>
      <tbody id="us-sig"><tr><td colspan="11" style="text-align:center;padding:18px;color:var(--muted)">No scan data yet — start the agent</td></tr></tbody>
    </table></div>

    <div class="sec" style="margin-top:18px"><span class="sec-title">Recent Trades</span></div>
    <div class="log-box" id="us-log">
      <div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>
    </div>
  </div>

  <!-- Sessions pane -->
  <div class="pane" id="pane-sessions">
    <div class="sec" style="flex-wrap:wrap;gap:8px">
      <span class="sec-title">Past Sessions</span>
      <span class="sec-badge" id="sess-count">0 sessions</span>
      <div style="margin-left:auto;display:flex;gap:5px;flex-wrap:wrap">
        <button class="btn btn-muted btn-sm sess-filter on" onclick="setSessFilter('all',this)">All</button>
        <button class="btn btn-muted btn-sm sess-filter" onclick="setSessFilter('india',this)">India (NSE)</button>
        <button class="btn btn-muted btn-sm sess-filter" onclick="setSessFilter('us',this)">US (NYSE)</button>
      </div>
    </div>
    <div id="sess-list">
      <div style="text-align:center;padding:36px;color:var(--muted)">No past sessions yet — sessions are archived automatically when the calendar date changes</div>
    </div>
  </div>

  <!-- Settings pane -->
  <div class="pane" id="pane-settings">
    <div style="max-width:580px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;padding:12px 14px;background:#161b22;border:1px solid #30363d;border-radius:6px">
        <input type="checkbox" id="settings-toggle" onchange="toggleSettings(this)" checked
               style="width:18px;height:18px;cursor:pointer;accent-color:#3fb950">
        <span style="font-weight:600;font-size:13px">Settings Constraints</span>
        <span id="settings-toggle-label" style="color:#3fb950;font-size:11px;margin-left:4px">● ACTIVE</span>
        <span style="color:#484f58;font-size:11px;margin-left:auto">Unchecked = full liberty (no filters)</span>
      </div>
      <div class="sec"><span class="sec-title">Risk Parameters</span></div>
      <div class="fgrid">
        <div class="fg"><label>Risk per Trade (0.01–1.0)</label><input class="settings-input" type="number" id="s-risk" step="0.01" min="0.01" max="1"></div>
        <div class="fg"><label>Confidence Threshold (%)</label><input class="settings-input" type="number" id="s-conf" step="1"    min="50"   max="95"></div>
        <div class="fg"><label>Stop Loss (0.005–0.2)</label>   <input class="settings-input" type="number" id="s-sl"   step="0.005" min="0.005" max="0.2"></div>
        <div class="fg"><label>Target (0.01–0.5)</label>       <input class="settings-input" type="number" id="s-tgt"  step="0.005" min="0.01"  max="0.5"></div>
        <div class="fg"><label>Check Interval (min)</label>    <input class="settings-input" type="number" id="s-chk"  step="1"    min="1"    max="60"></div>
        <div class="fg"><label>Idle Interval (min)</label>     <input class="settings-input" type="number" id="s-idle" step="1"    min="5"    max="120"></div>
        <div class="fg"><label>India Max Positions</label>     <input class="settings-input" type="number" id="s-ip"   step="1"    min="1"    max="16"></div>
        <div class="fg"><label>US Max Positions</label>        <input class="settings-input" type="number" id="s-up"   step="1"    min="1"    max="16"></div>
        <div class="fg"><label>EOD Profit Harvest (min before close)</label><input class="settings-input" type="number" id="s-eod-h" step="1" min="10" max="60"></div>
        <div class="fg"><label>EOD Force Exit (min before close)</label>    <input class="settings-input" type="number" id="s-eod-e" step="1" min="3"  max="30"></div>
        <div class="fg"><label>RL Exit Confidence (%)</label>               <input class="settings-input" type="number" id="s-rl-exit" step="1" min="50" max="95" title="Min RL confidence to trigger early exit from a held position"></div>
      </div>
      <div style="margin-top:14px;display:flex;gap:10px">
        <button class="btn btn-blue" onclick="saveConfig()">Save Settings</button>
      </div>

      <div class="sec" style="margin-top:24px"><span class="sec-title">Edit Portfolio State</span>
        <span class="sec-badge">raw values</span></div>
      <div class="es-grid">
        <div>
          <div class="es-col-title">India (₹)</div>
          <div class="fg"><label>Cash</label><input type="number" id="es-india-cash" step="0.01"></div>
          <div class="fg" style="margin-top:8px"><label>Realised P&amp;L</label><input type="number" id="es-india-rpnl" step="0.01"></div>
          <div class="fg" style="margin-top:8px"><label>Wins</label><input type="number" id="es-india-wins" step="1" min="0"></div>
          <div class="fg" style="margin-top:8px"><label>Losses</label><input type="number" id="es-india-losses" step="1" min="0"></div>
          <div class="fg" style="margin-top:8px"><label>Peak Portfolio</label><input type="number" id="es-india-peak" step="0.01"></div>
        </div>
        <div>
          <div class="es-col-title">US ($)</div>
          <div class="fg"><label>Cash</label><input type="number" id="es-us-cash" step="0.01"></div>
          <div class="fg" style="margin-top:8px"><label>Realised P&amp;L</label><input type="number" id="es-us-rpnl" step="0.01"></div>
          <div class="fg" style="margin-top:8px"><label>Wins</label><input type="number" id="es-us-wins" step="1" min="0"></div>
          <div class="fg" style="margin-top:8px"><label>Losses</label><input type="number" id="es-us-losses" step="1" min="0"></div>
          <div class="fg" style="margin-top:8px"><label>Peak Portfolio</label><input type="number" id="es-us-peak" step="0.01"></div>
        </div>
      </div>
      <div style="margin-top:14px;display:flex;gap:10px">
        <button class="btn btn-blue" onclick="saveEditState()">Apply State Changes</button>
      </div>
      <p style="font-size:10px;color:var(--muted);margin-top:8px">
        Position fields (qty, entry, stop-loss, target) are editable via the Edit button in each position row.
      </p>

      <div class="sec" style="margin-top:24px"><span class="sec-title" style="color:var(--red)">Danger Zone</span></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-red btn-sm" onclick="resetMkt('india')">Reset India State</button>
        <button class="btn btn-red btn-sm" onclick="resetMkt('us')">Reset US State</button>
        <button class="btn btn-red btn-sm" onclick="resetMkt('all')">Reset Everything</button>
      </div>
    </div>
  </div>

  <!-- Logs pane -->
  <div class="pane" id="pane-logs">

    <!-- Decision Log -->
    <div class="sec">
      <span class="sec-title">Decision Log</span>
      <span class="sec-badge" id="dlog-count">0 entries</span>
      <span style="margin-left:auto;font-size:10px;color:var(--muted)">why the bot does what it does</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap">
      <button class="btn btn-muted btn-sm log-filter-btn on" id="df-ALL"    onclick="setDlogFilter('ALL',this)">ALL</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-SCAN"   onclick="setDlogFilter('SCAN',this)">SCAN</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-ENTRY"  onclick="setDlogFilter('ENTRY',this)">ENTRY</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-EXIT"   onclick="setDlogFilter('EXIT',this)">EXIT</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-FILTER" onclick="setDlogFilter('FILTER',this)">FILTER</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-RISK"   onclick="setDlogFilter('RISK',this)">RISK</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-INDEX"  onclick="setDlogFilter('INDEX',this)">INDEX</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-CYCLE"  onclick="setDlogFilter('CYCLE',this)">CYCLE</button>
      <button class="btn btn-muted btn-sm log-filter-btn"   id="df-RL"     onclick="setDlogFilter('RL',this)" style="color:#c060ff">RL</button>
      <label style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);
                    margin-left:8px;text-transform:none;letter-spacing:0;cursor:pointer">
        <input type="checkbox" id="dlog-autoscroll" checked> Auto-scroll
      </label>
      <span id="dlog-live-dot" style="margin-left:4px;font-size:9px;color:var(--muted)">● live</span>
    </div>
    <div class="dlog-box" id="dlog-box">
      <div class="drow"><span class="al-ts">—</span><span class="dcat dcat-CYCLE">CYCLE</span><span class="dsym"></span><span class="dmsg muted">No decisions yet — start the agent</span></div>
    </div>

    <div class="sec" style="margin-top:18px">
      <span class="sec-title">Agent System Log</span>
      <span class="sec-badge" id="alog-count">0 entries</span>
      <span style="margin-left:auto;font-size:10px;color:var(--muted)">→ <span class="mono" style="color:var(--blue)">apex.log</span></span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;flex-wrap:wrap">
      <button class="btn btn-muted btn-sm log-filter-btn on"  onclick="setLogFilter('ALL',this)">ALL</button>
      <button class="btn btn-muted btn-sm log-filter-btn"     onclick="setLogFilter('INFO',this)">INFO</button>
      <button class="btn btn-muted btn-sm log-filter-btn"     onclick="setLogFilter('WARNING',this)">WARN</button>
      <button class="btn btn-muted btn-sm log-filter-btn"     onclick="setLogFilter('ERROR',this)">ERROR</button>
      <button class="btn btn-muted btn-sm log-filter-btn"     onclick="setLogFilter('DEBUG',this)">DEBUG</button>
      <label style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);
                    margin-left:8px;text-transform:none;letter-spacing:0;cursor:pointer">
        <input type="checkbox" id="auto-scroll" checked onchange="_autoScroll=this.checked"> Auto-scroll
      </label>
      <span id="log-live-dot" style="margin-left:4px;font-size:9px;color:var(--muted)">● live</span>
    </div>
    <div class="alog-box" id="agent-log" style="max-height:500px">
      <div class="alog-row"><span class="al-ts">—</span><span class="al-lvl al-debug">—</span><span class="muted">No agent activity yet — start the agent</span></div>
    </div>

    <div class="sec" style="margin-top:18px"><span class="sec-title">All Trade Activity</span></div>
    <div class="log-box" style="max-height:320px" id="all-log">
      <div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>
    </div>
  </div>

  <!-- Agent Terminal pane -->
  <div class="pane" id="pane-agent">
    <div class="sec">
      <span class="sec-title">RL Agent Terminal</span>
      <span id="rl-status-badge" style="padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;background:#1a0040;color:#666">● CHECKING</span>
      <span style="margin-left:auto;font-size:10px;color:var(--muted)">live PPO reasoning — updates every 4s</span>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <button class="btn btn-muted btn-sm" onclick="clearAgentTerminal()">Clear</button>
      <label style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);text-transform:none;letter-spacing:0;cursor:pointer">
        <input type="checkbox" id="agent-autoscroll" checked> Auto-scroll
      </label>
      <span id="agent-live-dot" style="font-size:9px;color:var(--muted)">● live</span>
    </div>
    <div id="agent-terminal" style="background:#0a0c10;border:1px solid #30363d;border-radius:6px;
         padding:14px;height:500px;overflow-y:auto;font-family:'Courier New',monospace;
         font-size:12px;color:#c9d1d9;line-height:1.6">
      <span style="color:#484f58">—</span> <span style="color:#666">Agent not yet started or no RL decisions recorded</span>
    </div>
  </div>

  <!-- Retraining Terminal pane -->
  <div class="pane" id="pane-retrain">
    <div class="sec">
      <span class="sec-title">RL Retraining Terminal</span>
      <span id="retrain-status-badge" style="padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;background:#1a1a1a;color:#8b949e">● IDLE</span>
      <span style="margin-left:auto;font-size:10px;color:var(--muted)">auto-triggered after every 16 closed RL trades</span>
    </div>
    <div style="display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap">
      <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 14px;font-size:11px;min-width:160px">
        <span style="color:var(--muted)">Trade buffer </span>
        <span id="retrain-buf-count" style="color:var(--text);font-weight:600">0</span>
        <span style="color:var(--muted)"> / 16</span>
        <div id="retrain-buf-bar" style="margin-top:5px;height:4px;background:#21262d;border-radius:2px">
          <div id="retrain-buf-fill" style="height:4px;background:#f0883e;border-radius:2px;width:0%;transition:width .4s"></div>
        </div>
      </div>
      <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 14px;font-size:11px;min-width:130px">
        <span style="color:var(--muted)">Total updates </span>
        <span id="retrain-total" style="color:var(--text);font-weight:600">0</span>
      </div>
      <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 14px;font-size:11px;min-width:130px">
        <span style="color:var(--muted)">Pending trades </span>
        <span id="retrain-pending" style="color:var(--text);font-weight:600">0</span>
      </div>
    </div>
    <div id="retrain-terminal" style="background:#0a0c10;border:1px solid #30363d;border-radius:6px;
         padding:14px;height:450px;overflow-y:auto;font-family:'Courier New',monospace;
         font-size:12px;color:#c9d1d9;line-height:1.7">
      <span style="color:#484f58">—</span> <span style="color:#666">No retraining events yet — waiting for 16 closed RL trades to accumulate</span>
    </div>
  </div>

  <!-- Footer -->
  <div class="foot">
    <span id="last-upd">Last update: —</span>
    <div id="prog"><div id="prog-f"></div></div>
    <span id="nxt-ref">Refreshing in 30s</span>
  </div>
</div>

<div class="toasts" id="toasts"></div>

<!-- Position edit modal -->
<div class="modal-overlay" id="pos-edit-modal" onclick="if(event.target===this)closeEditPos()">
  <div class="modal-box">
    <div class="modal-title">Edit Position: <span id="epm-sym">—</span></div>
    <input type="hidden" id="epm-mkt">
    <input type="hidden" id="epm-sym-h">
    <div class="fgrid">
      <div class="fg"><label>Qty (shares)</label>
        <input type="number" id="epm-qty" step="1" min="1"></div>
      <div class="fg"><label>Entry Price</label>
        <input type="number" id="epm-entry" step="0.01" min="0"></div>
      <div class="fg"><label>Stop Loss</label>
        <input type="number" id="epm-sl" step="0.01" min="0"></div>
      <div class="fg"><label>Target</label>
        <input type="number" id="epm-tgt" step="0.01" min="0"></div>
    </div>
    <div style="display:flex;gap:10px;margin-top:18px">
      <button class="btn btn-blue" onclick="saveEditPos()">Save Changes</button>
      <button class="btn btn-muted" onclick="closeEditPos()">Cancel</button>
    </div>
  </div>
</div>

<script>
const INTERVAL = 30;
let _progStart = null, _progTimer = null, _refreshTimer = null;
let _cfg = {};        // latest config snapshot — set in doRefresh
let _lastState = null; // last full /api/state response for SSE PnL recalc
let _activeTab = "india";
let _logPollTimer = null;
let _sessPollTimer = null;
let _agentPollTimer = null;
let _retrainPollTimer = null;
let _logFilter = "ALL";
let _autoScroll = true;
let _lastLogEntries = null;
let _lastDlogEntries = [];
let _dlogFilter = "ALL";
let _sessFilter = "all";
let _sessData   = [];

// ── Utilities ──────────────────────────────────────────────────────────────

// Format a UTC ISO timestamp string into local date + time.
// Always shows "15 Apr 14:30:22" so the date is visible in the log.
function fmtTs(ts) {
  if (!ts || ts === "—") return ts;
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;          // graceful fallback for legacy strings
  const timePart = d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
  const datePart = d.toLocaleDateString(undefined, {day:"2-digit", month:"short"});
  return datePart + "  " + timePart;
}

// Full date + time for the clock / last-update label.
function fmtDt(ts) {
  const d = ts ? new Date(ts) : new Date();
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleDateString(undefined, {day:"2-digit", month:"short", year:"numeric"}) + "  " +
         d.toLocaleTimeString(undefined, {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false});
}

function fc(v, sym) {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (sym === "₹") {
    if (a >= 1e7) return sym + (v/1e7).toFixed(2) + "Cr";
    if (a >= 1e5) return sym + (v/1e5).toFixed(2) + "L";
  }
  if (a >= 1e6) return sym + (v/1e6).toFixed(2) + "M";
  if (a >= 1e3) return sym + Math.abs(v).toLocaleString("en-IN",{maximumFractionDigits:0});
  return sym + v.toFixed(2);
}
const sc  = v => v > 0 ? "green" : v < 0 ? "red" : "muted";
const sgn = v => v >= 0 ? "+" : "";
const disp = s => s.replace(".NS","");

function toast(msg, ok=true) {
  const d = document.createElement("div");
  d.className = "toast " + (ok ? "tok" : "terr");
  d.innerHTML = `<span>${ok?"✓":"✗"}</span>${msg}`;
  document.getElementById("toasts").appendChild(d);
  setTimeout(() => d.remove(), 3200);
}

function tab(name, el) {
  _activeTab = name;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("on"));
  document.querySelectorAll(".pane").forEach(p => p.classList.remove("on"));
  el.classList.add("on");
  document.getElementById("pane-"+name).classList.add("on");
  clearInterval(_logPollTimer);
  clearInterval(_sessPollTimer);
  clearInterval(_agentPollTimer);
  clearInterval(_retrainPollTimer);
  if (name === "logs") {
    _logPollTimer = setInterval(pollLogs, 5000);
    setInterval(pollDecisions, 4000);
    pollLogs();
    pollDecisions();
  } else if (name === "sessions") {
    _sessPollTimer = setInterval(pollSessions, 15000);
    pollSessions();
  } else if (name === "agent") {
    _agentPollTimer = setInterval(pollAgentTerminal, 4000);
    pollAgentTerminal();
  } else if (name === "retrain") {
    _retrainPollTimer = setInterval(pollRetrainTerminal, 5000);
    pollRetrainTerminal();
  }
}

async function pollLogs() {
  try {
    const r = await fetch("/api/logs");
    if (!r.ok) return;
    const entries = await r.json();
    _lastLogEntries = entries;
    renderAgentLog(entries);
    const dot = document.getElementById("log-live-dot");
    if (dot) { dot.style.color = "var(--green)"; setTimeout(()=>{ dot.style.color="var(--muted)"; }, 800); }
  } catch(e) {}
}

function setLogFilter(lvl, el) {
  _logFilter = lvl;
  document.querySelectorAll(".log-filter-btn").forEach(b => b.classList.remove("on"));
  el.classList.add("on");
  if (_lastLogEntries) renderAgentLog(_lastLogEntries);
}

// ── Decision log ──────────────────────────────────────────────────────────
async function pollDecisions() {
  try {
    const r = await fetch("/api/think");
    if (!r.ok) return;
    const entries = await r.json();
    _lastDlogEntries = entries;
    renderDlog(entries);
    const dot = document.getElementById("dlog-live-dot");
    if (dot) { dot.style.color = "var(--green)"; setTimeout(()=>{ dot.style.color="var(--muted)"; }, 800); }
  } catch(e) {}
}

// ── Agent Terminal ────────────────────────────────────────────────────────
const _RL_COLORS = {BUY:'#3fb950', SELL:'#f85149', HOLD:'#8b949e', WARNING:'#f0883e', online:'#3fb950', Cycle:'#58a6ff'};

function renderAgentTerminal(entries) {
  const el = document.getElementById('agent-terminal');
  if (!el) return;
  const badge = document.getElementById('rl-status-badge');
  if (!entries || !entries.length) {
    el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">No RL decisions yet — start the agent</span>';
    if (badge) { badge.textContent = '● INACTIVE'; badge.style.color = '#f85149'; }
    return;
  }
  const isActive = entries.some(e => Date.now() - new Date(e.ts).getTime() < 600000);
  if (badge) {
    badge.textContent = isActive ? '● ACTIVE' : '● INACTIVE';
    badge.style.color  = isActive ? '#3fb950' : '#f85149';
    badge.style.background = isActive ? '#0a2016' : '#2a0a0a';
  }
  el.innerHTML = entries.map(e => {
    const colorKey = Object.keys(_RL_COLORS).find(k => e.msg && e.msg.includes(k));
    const c = colorKey ? _RL_COLORS[colorKey] : '#58a6ff';
    const ts = e.ts ? new Date(e.ts).toLocaleString(undefined,{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).replace(',','') : '';
    const sym = (e.sym && e.sym !== 'SYSTEM') ? `<span style="color:#58a6ff;margin-right:6px">${e.sym}</span>` : '';
    return `<div style="margin-bottom:3px;border-bottom:1px solid #161b22;padding-bottom:3px">` +
      `<span style="color:#484f58">${ts}</span>` +
      `<span style="color:#8b949e;margin:0 6px;font-size:10px">[RL]</span>` +
      sym +
      `<span style="color:${c}">${e.msg || ''}</span></div>`;
  }).join('');
  const auto = document.getElementById('agent-autoscroll');
  if (auto?.checked) el.scrollTop = el.scrollHeight;
}

async function pollAgentTerminal() {
  try {
    const r = await fetch('/api/think');
    if (!r.ok) return;
    const entries = await r.json();
    const rl = entries.filter(e => e.cat === 'RL');
    renderAgentTerminal(rl);
    const dot = document.getElementById('agent-live-dot');
    if (dot) { dot.style.color='var(--green)'; setTimeout(()=>{ dot.style.color='var(--muted)'; }, 800); }
  } catch(e) {}
}

function clearAgentTerminal() {
  fetch('/api/think/clear', {method:'POST'}).catch(()=>{});
  const el = document.getElementById('agent-terminal');
  if (el) el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">Cleared — waiting for next cycle</span>';
  _lastDlogEntries = [];
  const dbox = document.getElementById("dlog-box");
  if (dbox) renderDlog([]);
}

// ── Retraining Terminal ───────────────────────────────────────────────────
function renderRetrainTerminal(data) {
  const el      = document.getElementById('retrain-terminal');
  const badge   = document.getElementById('retrain-status-badge');
  const bufCnt  = document.getElementById('retrain-buf-count');
  const bufFill = document.getElementById('retrain-buf-fill');
  const total   = document.getElementById('retrain-total');
  const pending = document.getElementById('retrain-pending');
  if (!el) return;

  const thr  = data.threshold || 16;
  const nc   = data.new_count  || 0;
  const bufsz= data.buffer_size || 0;

  if (bufCnt)  bufCnt.textContent  = nc;
  if (bufFill) bufFill.style.width = Math.min(100, Math.round(nc / thr * 100)) + '%';
  if (total)   total.textContent   = data.total_updates || 0;
  if (pending) pending.textContent = bufsz;

  if (badge) {
    if (data.is_training) {
      badge.textContent = '⟳ TRAINING'; badge.style.color = '#f0883e'; badge.style.background = '#2a1800';
    } else if ((data.total_updates || 0) > 0) {
      badge.textContent = '● UPDATED';  badge.style.color = '#3fb950'; badge.style.background = '#0a2016';
    } else {
      badge.textContent = '● IDLE';     badge.style.color = '#8b949e'; badge.style.background = '#1a1a1a';
    }
  }

  const entries = data.log || [];
  if (!entries.length) {
    el.innerHTML = '<span style="color:#484f58">—</span> <span style="color:#666">No retraining events yet — waiting for 16 closed RL trades to accumulate</span>';
    return;
  }

  el.innerHTML = [...entries].reverse().map(e => {
    const ts = e.ts ? new Date(e.ts).toLocaleString(undefined,
      {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}
    ).replace(',','') : '';

    let color, prefix, icon;
    if (e.event === 'started') {
      color = '#58a6ff'; prefix = 'STARTED'; icon = '🔄';
    } else if (e.event === 'completed') {
      color = (e.improvement_pct || 0) >= 0 ? '#3fb950' : '#f0883e';
      prefix = 'APPLIED'; icon = '✓';
    } else if (e.event === 'failed') {
      color = '#f85149'; prefix = 'FAILED'; icon = '✗';
    } else if (e.event === 'skipped') {
      color = '#8b949e'; prefix = 'SKIPPED'; icon = '○';
    } else {
      color = '#8b949e'; prefix = (e.event || '').toUpperCase(); icon = '·';
    }

    const tradesTag = e.trades
      ? `<span style="color:#484f58;font-size:10px;margin-left:8px">${e.trades} trades</span>` : '';
    const improvTag = e.improvement_pct !== undefined
      ? `<span style="color:${(e.improvement_pct||0)>=0?'#3fb950':'#f85149'};font-size:10px;margin-left:8px">${e.improvement_pct>0?'+':''}${e.improvement_pct}%</span>` : '';
    const errTag = e.error
      ? `<div style="color:#f85149;font-size:10px;margin-left:16px;margin-top:2px">${e.error}</div>` : '';

    return `<div style="margin-bottom:5px;border-bottom:1px solid #161b22;padding-bottom:5px">` +
      `<span style="color:#484f58">${ts}</span>` +
      `<span style="color:${color};margin:0 7px;font-weight:700">${icon} ${prefix}</span>` +
      `<span style="color:${color}">${e.msg || ''}</span>` +
      tradesTag + improvTag + errTag +
      `</div>`;
  }).join('');
}

async function pollRetrainTerminal() {
  try {
    const r = await fetch('/api/retrain/log');
    if (!r.ok) return;
    renderRetrainTerminal(await r.json());
  } catch(e) {}
}

function setDlogFilter(cat, el) {
  _dlogFilter = cat;
  document.querySelectorAll('[id^="df-"]').forEach(b => b.classList.remove("on"));
  if (el) el.classList.add("on");
  renderDlog(_lastDlogEntries);
}

function renderDlog(entries) {
  const box = document.getElementById("dlog-box");
  const cnt = document.getElementById("dlog-count");
  if (!box) return;
  const all = entries || [];
  const filtered = _dlogFilter === "ALL" ? all : all.filter(e => e.cat === _dlogFilter);
  if (cnt) cnt.textContent = filtered.length + (filtered.length < all.length ? " / "+all.length : "") + " entries";
  if (!filtered.length) {
    box.innerHTML = `<div class="drow"><span class="al-ts">—</span><span class="dcat dcat-CYCLE">CYCLE</span><span class="dsym"></span><span class="dmsg muted">No ${_dlogFilter === "ALL" ? "" : _dlogFilter+" "}decisions yet</span></div>`;
    return;
  }
  box.innerHTML = [...filtered].reverse().map(e => {
    const cat = e.cat || "CYCLE";
    const ts  = fmtTs(e.ts || "");
    const sym = e.sym ? `<span class="dsym">${e.sym}</span>` : `<span class="dsym"></span>`;
    return `<div class="drow">
      <span class="al-ts">${ts}</span>
      <span class="dcat dcat-${cat}">${cat}</span>
      ${sym}
      <span class="dmsg">${e.msg || ""}</span>
    </div>`;
  }).join("");
  const auto = document.getElementById("dlog-autoscroll");
  if (auto?.checked) box.scrollTop = 0;
}

// ── Clock (browser local time) ────────────────────────────────────────────
function tick() {
  document.getElementById("clock").textContent = fmtDt();
}
setInterval(tick, 1000); tick();

// ── Progress bar ──────────────────────────────────────────────────────────
function startProg() {
  _progStart = Date.now();
  clearInterval(_progTimer);
  _progTimer = setInterval(() => {
    const e = (Date.now()-_progStart)/1000;
    document.getElementById("prog-f").style.width = Math.min(100, e/INTERVAL*100)+"%";
    const r = Math.max(0, INTERVAL - Math.floor(e));
    document.getElementById("nxt-ref").textContent = "Refresh in "+r+"s";
  }, 500);
}

// ── Render stats cards ────────────────────────────────────────────────────
function renderStats(d) {
  const ic = d.config.india_capital, uc = d.config.us_capital;
  [["india","₹",ic],["us","$",uc]].forEach(([mkt,sym,cap]) => {
    const m  = d[mkt];
    const pv = m.portfolio_value || 0;
    const df = pv - cap, pct = df/cap*100;
    const pnl = m.realised_pnl || 0;
    const tot = (m.wins||0)+(m.losses||0);
    const wr  = tot > 0 ? m.wins/tot*100 : 0;
    const dd   = m.drawdown || 0;
    const tpnl = m.total_pnl || 0;
    const upnl = m.unrealised_pnl || 0;
    document.getElementById(mkt+"-pv").textContent   = fc(pv,sym);
    document.getElementById(mkt+"-pv").className     = "c-val mono "+sc(df);
    document.getElementById(mkt+"-diff").innerHTML   = `<span class="${sc(df)}">${sgn(df)}${fc(df,sym)} (${sgn(pct)}${pct.toFixed(1)}%)</span>`;
    document.getElementById(mkt+"-cash").textContent = fc(m.cash,sym);
    document.getElementById(mkt+"-npos").textContent = Object.keys(m.positions||{}).length+" positions";
    document.getElementById(mkt+"-pnl").textContent  = fc(pnl,sym);
    document.getElementById(mkt+"-pnl").className    = "c-val mono "+sc(pnl);
    document.getElementById(mkt+"-ntrades").textContent = tot+" trades";
    document.getElementById(mkt+"-wr").textContent   = wr.toFixed(0)+"%";
    document.getElementById(mkt+"-wr").className     = "c-val mono "+(wr>=50?"green":"red");
    document.getElementById(mkt+"-wl").textContent   = (m.wins||0)+"W "+(m.losses||0)+"L";
    document.getElementById(mkt+"-dd").textContent   = dd.toFixed(1)+"%";
    document.getElementById(mkt+"-dd").className     = "c-val mono "+(dd>5?"red":"green");
    document.getElementById(mkt+"-tpnl").textContent = (tpnl>=0?"+":"")+fc(tpnl,sym);
    document.getElementById(mkt+"-tpnl").className   = "c-val mono "+sc(tpnl);
    document.getElementById(mkt+"-upnl").innerHTML   =
      `Float: <span class="${sc(upnl)}">${sgn(upnl)}${fc(upnl,sym)}</span>`;
    // market open badge
    const mb = document.getElementById(mkt+"-mkt");
    mb.className = "badge "+(m.market_open?"b-open":"b-closed");
    mb.textContent = m.market_open ? "OPEN" : "CLOSED";
    // session info
    const sl = document.getElementById(mkt+"-sess-lbl");
    if (sl) {
      const sessDate = m.session_date || "—";
      const nTrades  = (m.trade_log||[]).length;
      const pnlStr   = (m.realised_pnl||0) >= 0
        ? "+" + fc(m.realised_pnl||0, sym) : fc(m.realised_pnl||0, sym);
      sl.innerHTML = `Session <span class="mono" style="color:var(--muted)">${sessDate}</span>`
        + `  ·  ${nTrades} trade${nTrades!==1?"s":""}  ·  `
        + `<span class="${sc(m.realised_pnl||0)}">${pnlStr}</span>`;
    }
  });
  // header badges
  const nb = document.getElementById("nse-badge"), ub = document.getElementById("nyse-badge");
  nb.className = "badge "+(d.india.market_open?"b-open":"b-closed");
  nb.textContent = "NSE "+(d.india.market_open?"OPEN":"CLOSED");
  ub.className = "badge "+(d.us.market_open?"b-open":"b-closed");
  ub.textContent = "NYSE "+(d.us.market_open?"OPEN":"CLOSED");
}

// ── EOD per-position cell ─────────────────────────────────────────────────
function eodCell(p, cur, sym, mtc) {
  if (mtc == null || mtc <= 0) return `<td class="muted" style="font-size:10px">—</td>`;
  const harvestMin = _cfg.eod_harvest_min || 30;
  const exitMin    = _cfg.eod_exit_min    || 15;
  const isShort    = p.side === "short";
  const pnl        = isShort ? (p.entry - cur) * p.qty : (cur - p.entry) * p.qty;
  const needPct    = isShort
    ? ((cur / p.entry) - 1) * 100   // % drop needed to break even for shorts
    : ((p.entry / cur) - 1) * 100;  // % rise needed to break even for longs
  let badge = "", note = "";

  if (mtc <= exitMin) {
    badge = `<span style="color:var(--red);font-size:10px;font-weight:700">⚡ FORCE EXIT</span>`;
  } else if (mtc <= harvestMin) {
    if (pnl > 0) {
      badge = `<span style="color:var(--green);font-size:10px;font-weight:700">✓ HARVESTING</span>`;
    } else {
      badge = `<span style="color:var(--orange);font-size:10px;font-weight:700">⏳ HOLDING</span>`;
      if (needPct > 0.05)
        note = `<div style="color:var(--yellow);font-size:9px">need +${needPct.toFixed(1)}% → ${sym}${p.entry.toFixed(2)} BE</div>`;
    }
  } else {
    const m = Math.floor(mtc);
    badge = `<span style="color:var(--muted);font-size:10px">${m}m left</span>`;
    if (pnl < 0 && mtc < 60 && needPct > 0.05)
      note = `<div style="color:var(--yellow);font-size:9px">+${needPct.toFixed(1)}% to BE</div>`;
  }
  return `<td>${badge}${note}</td>`;
}

// ── Render positions table ────────────────────────────────────────────────
function renderPos(positions, prices, sym, bodyId, countId, mkt, mtc) {
  const cnt = Object.keys(positions||{}).length;
  document.getElementById(countId).textContent = cnt;
  const tb = document.getElementById(bodyId);
  if (!cnt) {
    tb.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:18px;color:var(--muted)">No open positions</td></tr>`;
    return;
  }
  tb.innerHTML = Object.entries(positions).map(([s,p]) => {
    const cur      = prices[s] || p.entry;
    const isShort  = p.side === "short";
    const pnl      = isShort ? (p.entry - cur) * p.qty : (cur - p.entry) * p.qty;
    const pct      = pnl / (p.entry * p.qty) * 100;
    const c        = pnl >= 0 ? "green" : "red";
    const sideBadge = `<span class="side-badge ${isShort ? 'side-short' : 'side-long'}">${isShort ? 'SHORT' : 'LONG'}</span>`;
    const btnLabel  = isShort ? "Cover" : "Sell";
    return `<tr>
      <td class="mono" style="font-weight:700;color:var(--blue)">${disp(s)} ${sideBadge}</td>
      <td class="mono">${p.qty}</td>
      <td class="mono">${sym}${p.entry.toFixed(2)}</td>
      <td class="mono">${sym}${cur.toFixed(2)}</td>
      <td class="mono ${c}">${pnl>=0?"+":""}${sym}${Math.abs(pnl).toFixed(0)} (${pct>=0?"+":""}${pct.toFixed(1)}%)</td>
      <td class="mono">${sym}${p.stop_loss.toFixed(2)}</td>
      <td class="mono">${sym}${p.target.toFixed(2)}</td>
      <td class="muted" style="font-size:11px">${fmtTs(p.entered_at||"")}</td>
      ${eodCell(p, cur, sym, mtc)}
      <td style="white-space:nowrap">
        <button class="btn btn-muted btn-sm" style="margin-right:4px"
          onclick="editPos('${mkt}','${s}',${p.qty},${p.entry},${p.stop_loss},${p.target})">Edit</button>
        <button class="btn btn-red btn-sm" onclick="sellPos('${mkt}','${s}')">${btnLabel}</button>
      </td>
    </tr>`;
  }).join("");
}

// ── EOD header badges + per-market banners ────────────────────────────────
function renderEodHeader(d) {
  [["india","NSE"],["us","NYSE"]].forEach(([mkt, label]) => {
    const mtc        = d[mkt+"_mtc"];
    const nPos       = Object.keys(d[mkt].positions||{}).length;
    const harvestMin = d.config.eod_harvest_min || 30;
    const exitMin    = d.config.eod_exit_min    || 15;
    const badge      = document.getElementById(mkt+"-eod");
    const banner     = document.getElementById(mkt+"-eod-banner");
    const phase      = document.getElementById(mkt+"-eod-phase");
    const detail     = document.getElementById(mkt+"-eod-detail");

    if (mtc == null || mtc <= 0) {
      badge.style.display  = "none";
      banner.style.display = "none";
      return;
    }

    // Header badge (visible whenever market is open and within 60 min of close)
    if (mtc <= 60) {
      badge.style.display = "";
      badge.textContent   = `${label} EOD ${Math.floor(mtc)}m`;
      badge.className     = "badge " + (mtc <= exitMin ? "b-eod-exit" : "b-eod-warn");
    } else {
      badge.style.display = "none";
    }

    // Banner above positions table
    if (nPos > 0 && mtc <= harvestMin) {
      banner.style.display = "flex";
      if (mtc <= exitMin) {
        banner.className   = "eod-banner eod-exit";
        phase.textContent  = "⚡ FORCE EXIT";
        phase.style.color  = "var(--red)";
        detail.textContent = `Closing all ${nPos} position${nPos>1?"s":""} — ${Math.floor(mtc)} min until ${label} close`;
      } else {
        banner.className   = "eod-banner eod-warn";
        phase.textContent  = "⏳ EOD HARVEST";
        phase.style.color  = "var(--orange)";
        detail.textContent =
          `Selling profitable positions now. Force exit in ${Math.floor(mtc-exitMin)} min. No new buys.`;
      }
    } else if (nPos > 0 && mtc <= 60) {
      banner.style.display = "flex";
      banner.className     = "eod-banner eod-warn";
      phase.textContent    = "EOD approaching";
      phase.style.color    = "var(--muted)";
      detail.textContent   =
        `${Math.floor(mtc)} min until ${label} close — profit harvest starts in ${Math.floor(mtc-harvestMin)} min`;
    } else {
      banner.style.display = "none";
    }
  });
}

// ── Render signal board ───────────────────────────────────────────────────
function renderSig(analyses, positions, sym, bodyId) {
  const tb = document.getElementById(bodyId);
  if (!analyses||!analyses.length) {
    tb.innerHTML = `<tr><td colspan="11" style="text-align:center;padding:18px;color:var(--muted)">No scan data — start the agent</td></tr>`;
    return;
  }
  tb.innerHTML = analyses.map(a => {
    const inP  = positions&&positions[a.symbol];
    const conf = a.confidence;
    const cc   = conf>65?"var(--green)":conf>40?"var(--yellow)":"var(--muted)";
    const actC = a.score>25?"act-buy":a.score<-25?"act-sell":"act-hold";
    const actT = a.score>25?"BUY":a.score<-25?"SELL":"HOLD";
    const rsiC = a.rsi<35?"var(--green)":a.rsi>65?"var(--red)":"inherit";
    const macd = (a.signals?.MACD?.signal||"").split(" ")[0];
    const bb   = (a.signals?.BB?.signal||"").split(" ")[0];
    const ema  = (a.signals?.EMA?.signal||"").split(" ")[0];
    const vol  = (a.signals?.Vol?.signal||"").split(" ")[0];
    const hw   = a.hist_win_days   ?? 0;
    const ht   = a.hist_total_days ?? 0;
    const hArr = hw > ht/2 ? "↑" : hw < ht/2 ? "↓" : "→";
    const hTxt = ht > 0 ? `${hw}/${ht}${hArr}` : "–";
    const hClr = hw > ht/2 ? "var(--green)" : hw < ht/2 ? "var(--red)" : "var(--muted)";
    const nc   = a.news_count ?? 0;
    const ns   = a.news_score ?? 0;
    const nArr = ns > 0 ? "↑" : ns < 0 ? "↓" : "→";
    const nTxt = nc > 0 ? `${nc}${nArr}` : "–";
    const nClr = ns > 0 ? "var(--green)" : ns < 0 ? "var(--red)" : "var(--muted)";
    return `<tr>
      <td class="mono" style="font-weight:600${inP?";color:var(--blue)":""}">${disp(a.symbol)}${inP?" *":""}</td>
      <td class="mono">${sym}${(a.price||0).toFixed(2)}</td>
      <td class="mono" style="color:${rsiC}">${(a.rsi||0).toFixed(0)}</td>
      <td class="muted">${macd}</td>
      <td class="muted">${bb}</td>
      <td class="muted">${ema}</td>
      <td class="muted">${vol}</td>
      <td class="mono" style="color:${hClr};font-weight:600">${hTxt}</td>
      <td class="mono" style="color:${nClr};font-weight:600">${nTxt}</td>
      <td><div class="cbar"><div class="cbar-bg"><div class="cbar-fg" style="width:${conf}%;background:${cc}"></div></div>
          <span class="mono" style="color:${cc};font-size:11px">${conf.toFixed(0)}%</span></div></td>
      <td><span class="${actC}">${actT}</span></td>
    </tr>`;
  }).join("");
}

// ── Render trade log ──────────────────────────────────────────────────────
function renderLog(trades, id, n=20) {
  const el = document.getElementById(id);
  if (!trades||!trades.length) {
    el.innerHTML = `<div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>`;
    return;
  }
  el.innerHTML = [...trades].reverse().slice(0,n).map(t => {
    const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
    return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span><span class="${c}">${t.message}</span></div>`;
  }).join("");
}

function renderAllLogs(il, ul) {
  const all = [...(il||[]).map(t=>({...t,mkt:"NSE"})),
               ...(ul||[]).map(t=>({...t,mkt:"NYSE"}))]
    .sort((a,b)=>b.time.localeCompare(a.time)).slice(0,150);
  const el = document.getElementById("all-log");
  if (!all.length) { el.innerHTML=`<div class="log-row"><span class="lt">—</span><span class="muted">No trades yet</span></div>`; return; }
  el.innerHTML = all.map(t=>{
    const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
    return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span>
      <span class="muted" style="min-width:44px;font-size:10px">${t.mkt}</span>
      <span class="${c}">${t.message}</span></div>`;
  }).join("");
}

function renderAgentLog(entries) {
  if (entries) _lastLogEntries = entries;
  const all = _lastLogEntries || [];
  const el  = document.getElementById("agent-log");
  const cnt = document.getElementById("alog-count");
  if (!all.length) {
    el.innerHTML = `<div class="alog-row"><span class="al-ts">—</span><span class="al-lvl al-debug">—</span><span class="muted">No agent activity yet — start the agent</span></div>`;
    if (cnt) cnt.textContent = "0 entries";
    return;
  }
  const filtered = _logFilter === "ALL" ? all
    : all.filter(e => (e.level||"").toUpperCase() === _logFilter);
  if (!filtered.length) {
    el.innerHTML = `<div class="alog-row"><span class="al-ts">—</span><span class="al-lvl al-debug">—</span><span class="muted">No ${_logFilter} entries</span></div>`;
    if (cnt) cnt.textContent = "0 / " + all.length + " entries";
    return;
  }
  el.innerHTML = [...filtered].reverse().map(e => {
    const lc = "al-"+(e.level||"info").toLowerCase();
    return `<div class="alog-row">
      <span class="al-ts">${fmtTs(e.ts||"")}</span>
      <span class="al-lvl ${lc}">${e.level||""}</span>
      <span style="color:var(--text);flex:1">${e.msg||""}</span>
    </div>`;
  }).join("");
  if (cnt) cnt.textContent = filtered.length + (filtered.length < all.length ? " / "+all.length : "") + " entries";
  if (_autoScroll) el.scrollTop = 0;
}

// ── Render agent badge + controls ─────────────────────────────────────────
function renderAgent(a) {
  const ab  = document.getElementById("agent-badge");
  const bt  = document.getElementById("btn-toggle");
  const bp  = document.getElementById("btn-pause");
  if (a.running && !a.paused) {
    ab.className  = "badge b-run"; ab.textContent = "● "+(a.status||"RUNNING").toUpperCase();
    bt.className  = "btn btn-red"; bt.textContent = "■ Stop Agent";
    bp.textContent = "⏸ Pause";
  } else if (a.paused) {
    ab.className  = "badge b-pause"; ab.textContent = "⏸ PAUSED";
    bt.className  = "btn btn-red";   bt.textContent = "■ Stop Agent";
    bp.textContent = "▶ Resume";
  } else {
    ab.className  = "badge b-idle"; ab.textContent = "IDLE";
    bt.className  = "btn btn-green"; bt.textContent = "▶ Start Agent";
    bp.textContent = "⏸ Pause";
  }
  if (a.last_update) document.getElementById("last-upd").textContent = "Last update: "+fmtDt(a.last_update);
}

// ── Load config into form ─────────────────────────────────────────────────
function toggleSettings(el) {
  const on = el.checked;
  const lbl = document.getElementById('settings-toggle-label');
  lbl.textContent = on ? '● ACTIVE' : '○ FULL LIBERTY';
  lbl.style.color = on ? '#3fb950' : '#484f58';
  document.querySelectorAll('.settings-input').forEach(inp => inp.disabled = !on);
  fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({settings_enabled: on})
  });
}

function loadCfg(c) {
  document.getElementById("s-risk").value  = c.risk_per_trade;
  document.getElementById("s-conf").value  = c.confidence_threshold;
  document.getElementById("s-sl").value    = c.stop_loss_pct;
  document.getElementById("s-tgt").value   = c.target_pct;
  document.getElementById("s-chk").value   = c.check_interval_min;
  document.getElementById("s-idle").value  = c.idle_interval_min;
  document.getElementById("s-ip").value    = c.india_max_positions;
  document.getElementById("s-up").value    = c.us_max_positions;
  document.getElementById("s-eod-h").value = c.eod_harvest_min;
  document.getElementById("s-eod-e").value = c.eod_exit_min;
  document.getElementById("s-rl-exit").value = c.rl_exit_confidence ?? 55;
  // Sync settings toggle state
  const stToggle = document.getElementById('settings-toggle');
  if (stToggle && c.settings_enabled !== undefined) {
    const on = !!c.settings_enabled;
    stToggle.checked = on;
    const lbl = document.getElementById('settings-toggle-label');
    if (lbl) { lbl.textContent = on ? '● ACTIVE' : '○ FULL LIBERTY'; lbl.style.color = on ? '#3fb950' : '#484f58'; }
    document.querySelectorAll('.settings-input').forEach(inp => inp.disabled = !on);
  }
}

// ── Load editable state into form ─────────────────────────────────────────
function loadEditState(d) {
  const fill = (id, val, decimals=2) =>
    { const el = document.getElementById(id); if(el) el.value = (val||0).toFixed ? (val||0).toFixed(decimals) : (val||0); };
  fill("es-india-cash",   d.india.cash);
  fill("es-india-rpnl",   d.india.realised_pnl);
  fill("es-india-wins",   d.india.wins,    0);
  fill("es-india-losses", d.india.losses,  0);
  fill("es-india-peak",   d.india.peak_portfolio || d.india.portfolio_value);
  fill("es-us-cash",      d.us.cash);
  fill("es-us-rpnl",      d.us.realised_pnl);
  fill("es-us-wins",      d.us.wins,    0);
  fill("es-us-losses",    d.us.losses,  0);
  fill("es-us-peak",      d.us.peak_portfolio || d.us.portfolio_value);
}

async function saveEditState() {
  const gf = id => parseFloat(document.getElementById(id).value);
  const gi = id => parseInt(document.getElementById(id).value);
  const body = {
    india: { cash: gf("es-india-cash"), realised_pnl: gf("es-india-rpnl"),
             wins: gi("es-india-wins"), losses: gi("es-india-losses"),
             peak_portfolio: gf("es-india-peak") },
    us:    { cash: gf("es-us-cash"),    realised_pnl: gf("es-us-rpnl"),
             wins: gi("es-us-wins"),    losses: gi("es-us-losses"),
             peak_portfolio: gf("es-us-peak") },
  };
  const r = await fetch("/api/edit/state", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  toast(d.ok ? "Portfolio state updated" : "Update failed", d.ok);
  if (d.ok) setTimeout(doRefresh, 300);
}

// ── Position editing ──────────────────────────────────────────────────────
function editPos(mkt, sym, qty, entry, sl, tgt) {
  document.getElementById("epm-sym").textContent = disp(sym);
  document.getElementById("epm-mkt").value = mkt;
  document.getElementById("epm-sym-h").value = sym;
  document.getElementById("epm-qty").value   = qty;
  document.getElementById("epm-entry").value = entry.toFixed(2);
  document.getElementById("epm-sl").value    = sl.toFixed(2);
  document.getElementById("epm-tgt").value   = tgt.toFixed(2);
  document.getElementById("pos-edit-modal").classList.add("open");
}

function closeEditPos() {
  document.getElementById("pos-edit-modal").classList.remove("open");
}

async function saveEditPos() {
  const mkt = document.getElementById("epm-mkt").value;
  const sym = document.getElementById("epm-sym-h").value;
  const body = {
    qty:       parseInt(document.getElementById("epm-qty").value),
    entry:     parseFloat(document.getElementById("epm-entry").value),
    stop_loss: parseFloat(document.getElementById("epm-sl").value),
    target:    parseFloat(document.getElementById("epm-tgt").value),
  };
  const r = await fetch(`/api/edit/position/${mkt}/${sym}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  toast(d.ok ? "Position updated" : d.msg, d.ok);
  closeEditPos();
  if (d.ok) setTimeout(doRefresh, 300);
}

// ── Sessions ──────────────────────────────────────────────────────────────
async function pollSessions() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) return;
    _sessData = await r.json();
    renderSessions(_sessData);
  } catch(e) {}
}

function setSessFilter(f, el) {
  _sessFilter = f;
  document.querySelectorAll(".sess-filter").forEach(b => b.classList.remove("on"));
  el.classList.add("on");
  renderSessions(_sessData);
}

function toggleSessBody(safeId) {
  const el = document.getElementById("sb-"+safeId);
  if (el) el.classList.toggle("open");
}

function renderSessions(sessions) {
  const filtered = _sessFilter === "all" ? sessions
    : sessions.filter(s => s.market === _sessFilter);
  const cnt = document.getElementById("sess-count");
  if (cnt) cnt.textContent = filtered.length + " session" + (filtered.length !== 1 ? "s" : "");
  const tc = document.getElementById("sess-tab-cnt");
  if (tc) tc.textContent = sessions.length > 0 ? "("+sessions.length+")" : "";
  const el = document.getElementById("sess-list");
  if (!filtered.length) {
    el.innerHTML = `<div style="text-align:center;padding:36px;color:var(--muted)">No past sessions yet — sessions are archived automatically when the calendar date changes</div>`;
    return;
  }
  el.innerHTML = filtered.map(s => {
    const sm      = s.market === "india" ? "₹" : "$";
    const safe    = (s.id||"").replace(/[^a-zA-Z0-9]/g,"_");
    const pnlCls  = (s.net_pnl||0) >= 0 ? "green" : "red";
    const tot     = (s.wins||0) + (s.losses||0);
    const wr      = tot > 0 ? (s.wins/tot*100).toFixed(0)+"%" : "—";
    const mktCls  = s.market === "india" ? "sess-mkt-india" : "sess-mkt-us";
    const mktLbl  = s.market === "india" ? "NSE" : "NYSE";
    const sign    = (s.net_pnl||0) >= 0 ? "+" : "";
    const trades  = s.trades || [];
    const trHtml  = trades.length
      ? [...trades].reverse().map(t => {
          const c = t.kind==="BUY"?"lb":t.kind==="SELL"?"ls":"ll";
          return `<div class="log-row"><span class="lt">${fmtTs(t.time)}</span><span class="${c}">${t.message}</span></div>`;
        }).join("")
      : `<div class="log-row"><span class="muted" style="padding:4px 10px">No trades this session</span></div>`;
    return `<div class="sess-card">
      <div class="sess-hdr" onclick="toggleSessBody('${safe}')">
        <span class="sess-date">${s.date||"—"}</span>
        <span class="sess-mkt ${mktCls}">${mktLbl}</span>
        <div style="flex:1;display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:0 6px">
          <span class="mono" style="font-size:12px;color:var(--muted)">
            ${sm}${(s.start_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}
            → ${sm}${(s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}
          </span>
          <span class="mono ${pnlCls}" style="font-size:14px;font-weight:700">
            ${sign}${sm}${Math.abs(s.net_pnl||0).toLocaleString(undefined,{maximumFractionDigits:0})}
            <span style="font-size:11px;font-weight:400;color:inherit;opacity:.8">
              (${sign}${(s.pnl_pct||0).toFixed(1)}%)
            </span>
          </span>
          <span class="muted" style="font-size:11px">
            ${s.wins||0}W ${s.losses||0}L · ${wr} WR · ${s.n_trades||trades.length} trade${(s.n_trades||trades.length)!==1?"s":""}
          </span>
        </div>
        <span style="color:var(--muted);font-size:14px;padding-left:4px">›</span>
      </div>
      <div class="sess-body" id="sb-${safe}">
        <div class="sess-metrics">
          <div class="sess-m">
            <span class="sess-m-l">Start Capital</span>
            <span class="sess-m-v">${sm}${(s.start_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">End Cash</span>
            <span class="sess-m-v">${sm}${(s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">End Portfolio</span>
            <span class="sess-m-v">${sm}${(s.end_portfolio||s.end_cash||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Net P&amp;L</span>
            <span class="sess-m-v ${pnlCls}">${sign}${sm}${Math.abs(s.net_pnl||0).toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Wins / Losses</span>
            <span class="sess-m-v">${s.wins||0}W / ${s.losses||0}L</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Win Rate</span>
            <span class="sess-m-v">${wr}</span>
          </div>
          <div class="sess-m">
            <span class="sess-m-l">Archived at</span>
            <span class="sess-m-v muted" style="font-size:11px">${fmtTs(s.archived_at||"")}</span>
          </div>
        </div>
        <div class="log-box" style="max-height:200px;border-radius:0;border:none;background:#0a0e14">
          ${trHtml}
        </div>
      </div>
    </div>`;
  }).join("");
}

async function closeSessionNow(market) {
  if (!confirm(`Archive current ${market.toUpperCase()} session and reset to starting capital?\n\nAll open positions will be force-closed at current prices.`)) return;
  const r = await fetch("/api/sessions/close/"+market, {method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  if (d.ok) { setTimeout(doRefresh, 400); setTimeout(pollSessions, 600); }
}

// ── Main refresh ──────────────────────────────────────────────────────────
async function doRefresh() {
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    _cfg = d.config;
    _lastState = d;
    renderStats(d);
    renderAgent(d.agent);
    loadCfg(d.config);
    loadEditState(d);
    renderEodHeader(d);
    renderPos(d.india.positions, d.signals.india_prices, "₹", "india-pos", "india-pc", "india", d.india_mtc);
    renderPos(d.us.positions,    d.signals.us_prices,    "$",  "us-pos",    "us-pc",    "us",    d.us_mtc);
    renderSig(d.signals.india, d.india.positions, "₹", "india-sig");
    renderSig(d.signals.us,    d.us.positions,    "$",  "us-sig");
    renderLog(d.india.trade_log, "india-log");
    renderLog(d.us.trade_log,    "us-log");
    renderAllLogs(d.india.trade_log, d.us.trade_log);
    renderAgentLog(d.agent_log || []);
    if (d.decision_log) { _lastDlogEntries = d.decision_log; renderDlog(d.decision_log); }
    // update sessions tab badge + refresh sessions data in background
    const tc = document.getElementById("sess-tab-cnt");
    if (tc) tc.textContent = d.sessions_count > 0 ? "("+d.sessions_count+")" : "";
    if (_activeTab === "sessions") pollSessions();
    startProg();
  } catch(e) { toast("Refresh failed: "+e.message, false); }
}

// ── SSE live price updates ─────────────────────────────────────────────────
(function startPriceSSE() {
  const src = new EventSource('/api/prices/stream');
  src.onmessage = (e) => {
    try {
      const prices = JSON.parse(e.data);
      if (!_lastState) return;
      // recalculate PnL for each market using cached positions + fresh prices
      ['india', 'us'].forEach(mkt => {
        const sym  = mkt === 'india' ? '₹' : '$';
        const m    = _lastState[mkt];
        const cap  = mkt === 'india' ? _cfg.india_capital : _cfg.us_capital;
        if (!m) return;
        // merge: latest prices override signal prices
        const merged = Object.assign({}, _lastState.signals[mkt+'_prices'], prices);
        let upnl = 0;
        Object.entries(m.positions || {}).forEach(([s, p]) => {
          const cur = merged[s] || p.entry;
          upnl += p.side === 'short'
            ? (p.entry - cur) * p.qty
            : (cur - p.entry) * p.qty;
        });
        const pv    = m.cash + Object.entries(m.positions||{}).reduce((acc,[s,p])=>{
          const cur = merged[s]||p.entry;
          return acc + (p.side==='short' ? (p.entry-cur)*p.qty : cur*p.qty);
        }, 0);
        const tpnl  = (m.realised_pnl||0) + upnl;
        const df    = pv - cap, pct = cap > 0 ? df/cap*100 : 0;
        const fc    = (v,s) => s+(Math.abs(v)<1e6 ? Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0}) : (Math.abs(v)/1e5).toFixed(1)+'L');
        const sc    = v => v>=0?'green':'red';
        const sgn   = v => v>=0?'+':'-';
        const el    = id => document.getElementById(mkt+'-'+id);
        if (el('pv'))   { el('pv').textContent = fc(pv,sym); el('pv').className='c-val mono '+sc(df); }
        if (el('diff')) el('diff').innerHTML = `<span class="${sc(df)}">${sgn(df)}${fc(df,sym)} (${sgn(pct)}${Math.abs(pct).toFixed(1)}%)</span>`;
        if (el('tpnl')) { el('tpnl').textContent=(tpnl>=0?'+':'')+fc(tpnl,sym); el('tpnl').className='c-val mono '+sc(tpnl); }
        if (el('upnl')) el('upnl').innerHTML=`Float: <span class="${sc(upnl)}">${sgn(upnl)}${fc(upnl,sym)}</span>`;
      });
    } catch(_) {}
  };
  src.onerror = () => { src.close(); setTimeout(startPriceSSE, 5000); };
})();

function scheduleRefresh() {
  clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(()=>{ doRefresh(); scheduleRefresh(); }, INTERVAL*1000);
}

// ── Controls ──────────────────────────────────────────────────────────────
async function toggleAgent() {
  const running = document.getElementById("btn-toggle").textContent.includes("Stop");
  const url = running ? "/api/agent/stop" : "/api/agent/start";
  const r = await fetch(url,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 400);
}

async function pauseAgent() {
  const r = await fetch("/api/agent/pause",{method:"POST"});
  const d = await r.json();
  if (d.ok) toast(d.paused ? "Agent paused" : "Agent resumed");
  else toast(d.msg, false);
  setTimeout(doRefresh, 300);
}

async function sellPos(market, symbol) {
  if (!confirm("Force sell "+disp(symbol)+"?")) return;
  const r = await fetch("/api/sell/"+market+"/"+symbol,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 400);
}

async function saveConfig() {
  const p = {
    risk_per_trade:       parseFloat(document.getElementById("s-risk").value),
    confidence_threshold: parseInt(document.getElementById("s-conf").value),
    stop_loss_pct:        parseFloat(document.getElementById("s-sl").value),
    target_pct:           parseFloat(document.getElementById("s-tgt").value),
    check_interval_min:   parseInt(document.getElementById("s-chk").value),
    idle_interval_min:    parseInt(document.getElementById("s-idle").value),
    india_max_positions:  parseInt(document.getElementById("s-ip").value),
    us_max_positions:     parseInt(document.getElementById("s-up").value),
    eod_harvest_min:      parseInt(document.getElementById("s-eod-h").value),
    eod_exit_min:         parseInt(document.getElementById("s-eod-e").value),
    rl_exit_confidence:   parseInt(document.getElementById("s-rl-exit").value),
  };
  const r = await fetch("/api/config",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
  const d = await r.json();
  toast(d.ok ? "Settings saved — applies next cycle" : "Save failed", d.ok);
}

async function resetMkt(market) {
  if (!confirm("Reset "+market+" paper state? This erases all positions, trades and P&L.")) return;
  const r = await fetch("/api/reset/"+market,{method:"POST"});
  const d = await r.json();
  toast(d.msg, d.ok);
  setTimeout(doRefresh, 300);
}

// ── Boot ──────────────────────────────────────────────────────────────────
doRefresh();
scheduleRefresh();
startProg();
</script>
</body>
</html>"""

# ─── GUNICORN / WSGI STARTUP ──────────────────────────────────────────────────
# Runs when gunicorn (or any WSGI server) imports this module.
# __main__ block below is kept for local `python apex_dashboard.py` usage.

def _on_startup():
    global _state
    with _lock:
        _state = load_state()
    start_price_updater()
    apex_log.info("Apex started — Supabase storage, dual-loop active")

_on_startup()

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    # _on_startup() already ran at import time; no need to reload state here.
    url = "http://localhost:7000"
    print(f"\n  Apex Dashboard  →  {url}")
    print("  Press Ctrl+C to stop\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=7000, debug=False,
            use_reloader=False, threaded=True)
