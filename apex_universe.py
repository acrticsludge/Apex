"""Canonical trading universe.

The single definition of the symbols the bot trades. Both the dashboard
(``apex_dashboard.py``) and the RL package (``trading_agent/config.py``) import
this, so the model trains on exactly what gets traded.

This is a leaf module on purpose: it imports nothing from the project, so
importing it cannot trigger Flask startup, Supabase access, or any other side
effect. It previously lived as a literal in ``apex_dashboard.py``, which forced
``trading_agent`` to AST-parse that file to copy the lists out without importing
it — a silent-drift failure mode when the parse returned empty.
"""

INDIA_WATCHLIST: list[str] = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",
    "ICICIBANK.NS",  "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",
    "MARUTI.NS",     "TITAN.NS",      "WIPRO.NS",      "SUNPHARMA.NS",
]

US_WATCHLIST: list[str] = [
    "AAPL",  "MSFT",  "NVDA",  "GOOGL",
    "AMZN",  "META",  "TSLA",  "AMD",
    "NFLX",  "ORCL",  "INTC",  "CRM",
    "UBER",  "SHOP",  "PYPL",  "PLTR",
]
