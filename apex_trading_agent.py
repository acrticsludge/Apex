"""
Apex Trading Agent -- Dual Market Edition
NSE NIFTY 50 (morning IST) + NYSE/NASDAQ (night IST)
Separate capital allocation for each market.

HOW TO RUN:
  pip install yfinance pandas numpy rich
  python apex_trading_agent.py

MARKET SCHEDULE (IST):
  India (NSE)      -- Mon-Fri  09:15 - 15:30 IST
  US (NYSE/NASDAQ) -- Mon-Fri  19:00 - 01:30 IST
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Indian Market
INDIA_CAPITAL       = 50_000    # Paper capital in INR
INDIA_MAX_POSITIONS = 4

# US Market
US_CAPITAL          = 1_800      # Paper capital in USD
US_MAX_POSITIONS    = 4

# Shared risk settings
RISK_PER_TRADE       = 0.12     # 12% of available cash per trade
CONFIDENCE_THRESHOLD = 62
STOP_LOSS_PCT        = 0.03     # 3% stop loss
TARGET_PCT           = 0.045    # 4.5% profit target
CHECK_INTERVAL_MIN   = 1        # when a market is open
IDLE_INTERVAL_MIN    = 15       # when both are closed
LIVE_MODE            = False
STATE_FILE           = "apex_dual_state.json"

# Top NIFTY 50 stocks -- yfinance uses .NS suffix for NSE
INDIA_WATCHLIST = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",
    "ICICIBANK.NS",  "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",
    "MARUTI.NS",     "TITAN.NS",      "WIPRO.NS",      "SUNPHARMA.NS",
]

# Top US tech stocks
US_WATCHLIST = [
    "AAPL",  "MSFT",  "NVDA",  "GOOGL",
    "AMZN",  "META",  "TSLA",  "AMD",
    "NFLX",  "ORCL",  "INTC",  "CRM",
    "UBER",  "SHOP",  "PYPL",  "PLTR",
]

console = Console()

# ─────────────────────────────────────────────
#  MARKET HOURS
# ─────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))
EDT = timezone(timedelta(hours=-4))   # EDT summer; change to -5 in Nov-Mar

def is_india_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_ = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_ <= now <= close_

def is_us_market_open() -> bool:
    now = datetime.now(EDT)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_

def active_markets():
    return is_india_market_open(), is_us_market_open()

# ─────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────

def fetch_data(symbol: str):
    try:
        df = yf.download(symbol, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns=str.lower).dropna()
        return df
    except Exception as e:
        console.print(f"[red]Fetch error {symbol}: {e}[/red]")
        return None

def get_current_price(symbol: str):
    try:
        return float(yf.Ticker(symbol).fast_info.last_price)
    except Exception:
        df = fetch_data(symbol)
        return float(df["close"].iloc[-1]) if df is not None else None

def fetch_all_prices(symbols: list) -> dict:
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

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def calc_rsi(close, period=14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float((100 - 100 / (1 + rs)).iloc[-1])

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return float(line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

def calc_bollinger(close, period=20):
    mid = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    return float((mid + 2*sd).iloc[-1]), float(mid.iloc[-1]), float((mid - 2*sd).iloc[-1])

def calc_ema(close, period: int) -> float:
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])

def calc_volume_ratio(volume, period=20) -> float:
    avg = volume.rolling(period).mean().iloc[-1]
    return float(volume.iloc[-1] / avg) if avg > 0 else 1.0

# ─────────────────────────────────────────────
#  SIGNAL ENGINE  (market-agnostic)
# ─────────────────────────────────────────────

def analyse(symbol: str, df, live_price: float) -> dict:
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)
    price  = live_price or float(close.iloc[-1])
    signals = {}
    score   = 0

    rsi = calc_rsi(close)
    signals["RSI"] = {"value": f"{rsi:.1f}"}
    if rsi < 30:
        signals["RSI"]["signal"] = "OVERSOLD BUY"; score += 25
    elif rsi > 70:
        signals["RSI"]["signal"] = "OVERBOUGHT SELL"; score -= 25
    elif rsi < 50:
        signals["RSI"]["signal"] = "Mildly bullish"; score += 8
    else:
        signals["RSI"]["signal"] = "Mildly bearish"; score -= 8

    bb_u, bb_m, bb_l = calc_bollinger(close)
    signals["Bollinger"] = {"value": f"U:{bb_u:.0f} M:{bb_m:.0f} L:{bb_l:.0f}"}
    if price < bb_l:
        signals["Bollinger"]["signal"] = "Below band BUY"; score += 20
    elif price > bb_u:
        signals["Bollinger"]["signal"] = "Above band SELL"; score -= 20
    else:
        signals["Bollinger"]["signal"] = "Inside band HOLD"

    macd_l, macd_s, macd_h = calc_macd(close)
    signals["MACD"] = {"value": f"Hist:{macd_h:.2f}"}
    if macd_h > 0 and macd_l > macd_s:
        signals["MACD"]["signal"] = "Bullish crossover BUY"; score += 20
    elif macd_h < 0 and macd_l < macd_s:
        signals["MACD"]["signal"] = "Bearish crossover SELL"; score -= 20
    else:
        signals["MACD"]["signal"] = "Neutral HOLD"

    ema9  = calc_ema(close, 9)
    ema21 = calc_ema(close, 21)
    signals["EMA 9/21"] = {"value": f"9:{ema9:.0f} 21:{ema21:.0f}"}
    if ema9 > ema21:
        signals["EMA 9/21"]["signal"] = "Bullish cross BUY"; score += 20
    else:
        signals["EMA 9/21"]["signal"] = "Bearish cross SELL"; score -= 20

    vr = calc_volume_ratio(volume)
    signals["Volume"] = {"value": f"{vr:.2f}x avg"}
    if vr > 1.5:
        signals["Volume"]["signal"] = "High volume confirms"; score += 15
    elif vr < 0.7:
        signals["Volume"]["signal"] = "Low volume weak"; score -= 5
    else:
        signals["Volume"]["signal"] = "Normal volume"

    confidence = min(100, max(0, (score + 100) / 2))
    return {
        "symbol":     symbol,
        "price":      price,
        "score":      score,
        "confidence": round(confidence, 1),
        "signals":    signals,
        "rsi":        rsi,
        "bb_upper":   bb_u,
        "bb_lower":   bb_l,
    }

# ─────────────────────────────────────────────
#  STATE MANAGEMENT
# ─────────────────────────────────────────────

def _empty_mstate(capital: float) -> dict:
    return {
        "cash":           float(capital),
        "positions":      {},
        "realised_pnl":   0.0,
        "wins":           0,
        "losses":         0,
        "peak_portfolio": float(capital),
        "max_drawdown":   0.0,
        "trade_log":      [],
    }

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "india":      _empty_mstate(INDIA_CAPITAL),
        "us":         _empty_mstate(US_CAPITAL),
        "started_at": datetime.now().isoformat(),
    }

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ─────────────────────────────────────────────
#  TRADE EXECUTION  (works on a market sub-state dict)
# ─────────────────────────────────────────────

def log_trade(mstate: dict, message: str, kind: str):
    mstate["trade_log"].append({
        "time":    datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "kind":    kind,
    })
    if len(mstate["trade_log"]) > 100:
        mstate["trade_log"] = mstate["trade_log"][-100:]

def execute_buy(symbol: str, price: float, mstate: dict):
    alloc = mstate["cash"] * RISK_PER_TRADE
    qty   = max(1, int(alloc / price))
    cost  = qty * price
    if cost > mstate["cash"]:
        return None
    mstate["cash"] -= cost
    mstate["positions"][symbol] = {
        "qty":        qty,
        "entry":      price,
        "stop_loss":  round(price * (1 - STOP_LOSS_PCT), 2),
        "target":     round(price * (1 + TARGET_PCT),    2),
        "entered_at": datetime.now().isoformat(),
    }
    log_trade(mstate,
        f"BUY  {qty:>4} {symbol:<16} @ {price:>10.2f} | "
        f"SL:{price*(1-STOP_LOSS_PCT):.2f}  T:{price*(1+TARGET_PCT):.2f}  Cost:{cost:.0f}",
        "BUY")
    return qty

def execute_sell(symbol: str, price: float, reason: str, mstate: dict):
    pos = mstate["positions"].get(symbol)
    if not pos:
        return
    proceeds = pos["qty"] * price
    pnl      = proceeds - pos["qty"] * pos["entry"]
    mstate["cash"]         += proceeds
    mstate["realised_pnl"] += pnl
    if pnl >= 0: mstate["wins"]   += 1
    else:        mstate["losses"] += 1
    sign = "+" if pnl >= 0 else ""
    log_trade(mstate,
        f"SELL {pos['qty']:>4} {symbol:<16} @ {price:>10.2f} | "
        f"P&L:{sign}{abs(pnl):.0f}  ({reason})",
        "SELL" if pnl >= 0 else "LOSS")
    del mstate["positions"][symbol]

# ─────────────────────────────────────────────
#  MARKET CYCLE  (generic, reused for India and US)
# ─────────────────────────────────────────────

def run_market_cycle(watchlist: list, mstate: dict, max_pos: int):
    prices = fetch_all_prices(watchlist)

    for sym in list(mstate["positions"].keys()):
        pos   = mstate["positions"][sym]
        price = prices.get(sym) or get_current_price(sym)
        if price is None:
            continue
        if price <= pos["stop_loss"]:
            execute_sell(sym, price, "STOP LOSS",  mstate)
        elif price >= pos["target"]:
            execute_sell(sym, price, "TARGET HIT", mstate)

    analyses = []
    for symbol in watchlist:
        df = fetch_data(symbol)
        if df is None:
            continue
        live_price = prices.get(symbol) or float(df["close"].iloc[-1])
        analyses.append(analyse(symbol, df, live_price))
        time.sleep(0.3)

    open_pos = len(mstate["positions"])
    for a in sorted(analyses, key=lambda x: x["confidence"], reverse=True):
        sym    = a["symbol"]
        in_pos = sym in mstate["positions"]
        if (not in_pos
                and a["confidence"] >= CONFIDENCE_THRESHOLD
                and a["score"] > 0
                and open_pos < max_pos
                and mstate["cash"] > a["price"] * 2):
            qty = execute_buy(sym, a["price"], mstate)
            if qty:
                open_pos += 1
        elif in_pos and a["score"] < -30:
            execute_sell(sym, a["price"], "SIGNAL EXIT", mstate)

    return analyses, prices

# ─────────────────────────────────────────────
#  DASHBOARD HELPERS
# ─────────────────────────────────────────────

def disp(symbol: str) -> str:
    return symbol.replace(".NS", "")

def portfolio_value(mstate: dict, prices: dict) -> float:
    val = mstate["cash"]
    for sym, pos in mstate["positions"].items():
        val += (prices.get(sym) or pos["entry"]) * pos["qty"]
    return val

def _stats_row(mstate: dict, prices: dict, capital: float,
               currency: str, label: str, mkt_open: bool):
    pval = portfolio_value(mstate, prices)
    diff = pval - capital
    pct  = diff / capital * 100
    wins, losses = mstate["wins"], mstate["losses"]
    total    = wins + losses
    win_rate = (wins / total * 100) if total else 0

    if pval > mstate["peak_portfolio"]:
        mstate["peak_portfolio"] = pval
    dd = (mstate["peak_portfolio"] - pval) / mstate["peak_portfolio"] * 100
    if dd > mstate["max_drawdown"]:
        mstate["max_drawdown"] = dd

    sign    = "+" if diff >= 0 else ""
    col     = "green" if diff >= 0 else "red"
    mkt_tag = "[green]OPEN[/green]" if mkt_open else "[yellow]CLOSED[/yellow]"

    pnl_sign = "+" if mstate["realised_pnl"] >= 0 else ""
    pnl_col  = "green" if mstate["realised_pnl"] >= 0 else "red"
    wr_col   = "green" if win_rate >= 50 else "red"
    dd_col   = "red"   if dd > 5        else "green"

    g = Table.grid(expand=True, padding=(0, 1))
    for _ in range(5): g.add_column(ratio=1)
    g.add_row(
        Panel(
            f"[bold]{currency}{pval:,.0f}[/bold]\n"
            f"[{col}]{sign}{currency}{abs(diff):,.0f} ({sign}{pct:.1f}%)[/{col}]",
            title=f"{label}  |  {mkt_tag}", box=box.ROUNDED),
        Panel(
            f"[bold]{currency}{mstate['cash']:,.0f}[/bold]\n[dim]free capital[/dim]",
            title="Cash", box=box.ROUNDED),
        Panel(
            f"[bold][{pnl_col}]{pnl_sign}{currency}{abs(mstate['realised_pnl']):,.0f}[/{pnl_col}][/bold]\n"
            f"[dim]{total} trades[/dim]",
            title="Realised P&L", box=box.ROUNDED),
        Panel(
            f"[bold][{wr_col}]{win_rate:.0f}%[/{wr_col}][/bold]\n[dim]{wins}W  {losses}L[/dim]",
            title="Win rate", box=box.ROUNDED),
        Panel(
            f"[bold][{dd_col}]{dd:.1f}%[/{dd_col}][/bold]\n[dim]from peak[/dim]",
            title="Drawdown", box=box.ROUNDED),
    )
    return g

def _signal_table(analyses: list, mstate: dict, currency: str, title: str) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAD, expand=True,
              header_style="bold dim", show_lines=False)
    t.add_column("Symbol",     style="bold",    min_width=12)
    t.add_column("Price",      justify="right",  min_width=12)
    t.add_column("RSI",        justify="right",  min_width=6)
    t.add_column("MACD",       justify="center", min_width=18)
    t.add_column("Bollinger",  justify="center", min_width=16)
    t.add_column("EMA",        justify="center", min_width=14)
    t.add_column("Confidence", justify="right",  min_width=12)
    t.add_column("Action",     justify="center", min_width=10)

    for a in analyses:
        sym  = a["symbol"]
        conf = a["confidence"]
        if a["score"] > 25:
            action = "[green bold]BUY[/green bold]"
        elif a["score"] < -25:
            action = "[red bold]SELL[/red bold]"
        else:
            action = "[dim]HOLD[/dim]"
        conf_col = "green" if conf > 65 else "yellow" if conf > 40 else "dim"
        rsi_col  = "green" if a["rsi"] < 35 else "red" if a["rsi"] > 65 else "white"
        in_pos   = " [cyan]*[/cyan]" if sym in mstate["positions"] else ""
        macd_sig = a["signals"]["MACD"]["signal"].split(" ")[0]
        bb_sig   = a["signals"]["Bollinger"]["signal"].split(" ")[0]
        ema_sig  = a["signals"]["EMA 9/21"]["signal"].split(" ")[0]
        t.add_row(
            disp(sym) + in_pos,
            f"{currency}{a['price']:,.2f}",
            f"[{rsi_col}]{a['rsi']:.0f}[/{rsi_col}]",
            macd_sig, bb_sig, ema_sig,
            f"[{conf_col}]{conf:.0f}%[/{conf_col}]",
            action,
        )
    return t

def _positions_table(mstate: dict, prices: dict, currency: str, title: str):
    if not mstate["positions"]:
        return None
    pt = Table(title=title, box=box.SIMPLE_HEAD, expand=True, header_style="bold dim")
    pt.add_column("Symbol"); pt.add_column("Qty",    justify="right")
    pt.add_column("Entry",   justify="right"); pt.add_column("Current", justify="right")
    pt.add_column("P&L",     justify="right"); pt.add_column("Stop",    justify="right")
    pt.add_column("Target",  justify="right"); pt.add_column("Entered", justify="right")
    for sym, pos in mstate["positions"].items():
        cur = prices.get(sym) or pos["entry"]
        pnl = (cur - pos["entry"]) * pos["qty"]
        pc  = pnl / (pos["entry"] * pos["qty"]) * 100
        c   = "green" if pnl >= 0 else "red"
        s2  = "+" if pnl >= 0 else ""
        pt.add_row(
            disp(sym), str(pos["qty"]),
            f"{currency}{pos['entry']:,.2f}",     f"{currency}{cur:,.2f}",
            f"[{c}]{s2}{currency}{abs(pnl):,.2f} ({s2}{pc:.1f}%)[/{c}]",
            f"{currency}{pos['stop_loss']:,.2f}",  f"{currency}{pos['target']:,.2f}",
            pos["entered_at"][11:19],
        )
    return pt

def _trade_log_table(mstate: dict, n: int = 5, title: str = "Recent trades"):
    if not mstate["trade_log"]:
        return None
    lt = Table(title=title, box=box.SIMPLE_HEAD, expand=True, header_style="bold dim")
    lt.add_column("Time", min_width=8); lt.add_column("Log")
    for entry in reversed(mstate["trade_log"][-n:]):
        c = "green" if entry["kind"] == "BUY" else "red" if entry["kind"] == "LOSS" else "cyan"
        lt.add_row(entry["time"], f"[{c}]{entry['message']}[/{c}]")
    return lt

# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────

def render_dashboard(state: dict,
                     india_analyses: list, india_prices: dict, india_open: bool,
                     us_analyses:    list, us_prices:    dict, us_open:    bool):
    console.clear()
    now_ist = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S IST")
    active  = []
    if india_open: active.append("[green]NSE OPEN[/green]")
    if us_open:    active.append("[green]NYSE OPEN[/green]")
    if not active: active = ["[yellow]BOTH MARKETS CLOSED[/yellow]"]

    console.print(Panel(
        "[bold]Apex Dual-Market Trading Agent[/bold]  |  " + now_ist + "  |  "
        + "  |  ".join(active),
        style="bold", box=box.SIMPLE,
    ))

    # India
    console.print(_stats_row(state["india"], india_prices, INDIA_CAPITAL,
                             "Rs", "India  NSE / NIFTY 50", india_open))
    if india_analyses:
        console.print(_signal_table(india_analyses, state["india"], "Rs",
                                    "India  Signal board (NIFTY 50 top 16)"))
    pt = _positions_table(state["india"], india_prices, "Rs", "India  Open positions")
    if pt: console.print(pt)
    lt = _trade_log_table(state["india"], 4, "India  Recent trades")
    if lt: console.print(lt)

    console.rule(style="dim")

    # US
    console.print(_stats_row(state["us"], us_prices, US_CAPITAL,
                             "$", "US  NYSE / NASDAQ", us_open))
    if us_analyses:
        console.print(_signal_table(us_analyses, state["us"], "$",
                                    "US  Signal board (Top 16 Tech)"))
    pt = _positions_table(state["us"], us_prices, "$", "US  Open positions")
    if pt: console.print(pt)
    lt = _trade_log_table(state["us"], 4, "US  Recent trades")
    if lt: console.print(lt)

    interval = CHECK_INTERVAL_MIN if (india_open or us_open) else IDLE_INTERVAL_MIN
    console.print(
        f"\n[dim]Refreshing every {interval} min  |  State: {STATE_FILE}  |  Ctrl+C to stop[/dim]")

# ─────────────────────────────────────────────
#  LIVE ORDER STUBS
# ─────────────────────────────────────────────
# India: Zerodha Kite Connect / Angel One SmartAPI
# US:    Alpaca (paper -> live)
#
# def place_live_order_india(symbol, qty, side): ...
# def place_live_order_us(symbol, qty, side):    ...

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    state = load_state()
    console.print(Panel(
        "[bold green]Apex Dual-Market Trading Agent[/bold green]\n\n"
        f"[bold cyan]India (NSE)[/bold cyan]  Rs{INDIA_CAPITAL:,} paper"
        f"   09:15-15:30 IST  |  Top 16 NIFTY 50 stocks\n"
        f"[bold cyan]US (NYSE)[/bold cyan]    ${US_CAPITAL:,} paper    "
        f"   19:00-01:30 IST  |  Top 16 Tech stocks\n\n"
        f"Risk/trade: {int(RISK_PER_TRADE*100)}%  |  "
        f"SL: {int(STOP_LOSS_PCT*100)}%  |  "
        f"Target: {TARGET_PCT*100:.1f}%  |  "
        "Data: [bold]yfinance[/bold] (free)\n"
        f"Mode: [bold]{'LIVE TRADING' if LIVE_MODE else 'PAPER SIMULATION'}[/bold]",
        box=box.ROUNDED,
    ))
    time.sleep(2)

    try:
        while True:
            india_open, us_open = active_markets()
            india_analyses, india_prices = [], {}
            us_analyses,    us_prices    = [], {}

            if india_open:
                console.print("[bold cyan]Fetching India market data (NSE)...[/bold cyan]")
                india_analyses, india_prices = run_market_cycle(
                    INDIA_WATCHLIST, state["india"], INDIA_MAX_POSITIONS)

            if us_open:
                console.print("[bold cyan]Fetching US market data (NYSE/NASDAQ)...[/bold cyan]")
                us_analyses, us_prices = run_market_cycle(
                    US_WATCHLIST, state["us"], US_MAX_POSITIONS)

            render_dashboard(state,
                             india_analyses, india_prices, india_open,
                             us_analyses,    us_prices,    us_open)
            save_state(state)

            sleep_min = CHECK_INTERVAL_MIN if (india_open or us_open) else IDLE_INTERVAL_MIN
            console.print(f"\n[dim]Next check in {sleep_min} minutes...[/dim]")
            time.sleep(sleep_min * 60)

    except KeyboardInterrupt:
        save_state(state)
        india_pval = portfolio_value(state["india"], {})
        us_pval    = portfolio_value(state["us"],    {})
        india_diff = india_pval - INDIA_CAPITAL
        us_diff    = us_pval   - US_CAPITAL
        console.print("\n[bold]Agent stopped.[/bold]")
        console.print(
            f"India portfolio : Rs{india_pval:,.0f}  "
            f"({'+'if india_diff>=0 else ''}Rs{india_diff:,.0f})")
        console.print(
            f"US    portfolio : ${us_pval:,.2f}  "
            f"({'+'if us_diff>=0 else ''}${us_diff:,.2f})")
        console.print(f"State saved to [cyan]{STATE_FILE}[/cyan]")

if __name__ == "__main__":
    main()
