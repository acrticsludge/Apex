"""apex_market — exchange calendars and time zones.

Extracted from apex_dashboard.py so the market-hours rules can be tested and
reasoned about without importing the Flask app. Leaf module: no project imports,
so it cannot create a cycle.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# IST has no DST. US Eastern does: UTC-5 in winter, UTC-4 in summer. A frozen
# -4 offset made the app believe the US close was 21:00 UTC year round, so the
# EOD harvest and force-exit fired an hour late for ~5 months a year.
IST = timezone(timedelta(hours=5, minutes=30))
EDT = ZoneInfo("America/New_York")

# Local session times, (open, close).
SESSION_HOURS = {
    "india": ((9, 15), (15, 30)),
    "us": ((9, 30), (16, 0)),
}


def tz_for(market_key: str):
    return IST if market_key == "india" else EDT


def _at(local: datetime, hour: int, minute: int) -> datetime:
    return local.replace(hour=hour, minute=minute, second=0, microsecond=0)


def session_bounds(market_key: str, now: datetime | None = None):
    """(open, close) as timezone-aware datetimes in the market's local time."""
    local = now if now is not None else datetime.now(tz_for(market_key))
    (oh, om), (ch, cm) = SESSION_HOURS[market_key]
    return _at(local, oh, om), _at(local, ch, cm)


def is_open(market_key: str, now: datetime | None = None) -> bool:
    """True during the local continuous session, weekdays only.

    Exchange holidays are not modelled; this is a weekday + session-hours check,
    which is what the original implementation did.
    """
    local = now if now is not None else datetime.now(tz_for(market_key))
    if local.weekday() >= 5:
        return False
    open_t, close_t = session_bounds(market_key, local)
    return open_t <= local <= close_t


def minutes_to_close(market_key: str, now: datetime | None = None):
    """Minutes remaining until the close, or None when not in session."""
    local = now if now is not None else datetime.now(tz_for(market_key))
    if local.weekday() >= 5:
        return None
    open_t, close_t = session_bounds(market_key, local)
    if local < open_t or local > close_t:
        return None
    return (close_t - local).total_seconds() / 60


def minutes_since_open(market_key: str, now: datetime | None = None) -> float:
    open_t, _ = session_bounds(market_key, now)
    local = now if now is not None else datetime.now(tz_for(market_key))
    return (local - open_t).total_seconds() / 60


def session_date(market_key: str, now: datetime | None = None) -> str:
    """Local calendar date, used as the session key in the trading ledger."""
    local = now if now is not None else datetime.now(tz_for(market_key))
    return local.strftime("%Y-%m-%d")


# ── Backwards-compatible names the dashboard imported ────────────────────────

def is_india_open() -> bool:
    return is_open("india")


def is_us_open() -> bool:
    return is_open("us")
