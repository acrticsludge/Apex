"""Regression tests for DST-correct US market hours.

EDT was hardcoded to UTC-4, but US Eastern is UTC-5 during standard time. That
made the app believe the market closed at 21:00 UTC instead of 22:00 UTC for
roughly five months a year, so the EOD harvest and force-exit fired an hour late
and positions were held through the real close.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import apex_dashboard as dash
import apex_market as mkt

EST = ZoneInfo("America/New_York")
EDT_ZONE = ZoneInfo("America/New_York")


def test_eastern_offset_is_dst_aware():
    """EDT must be a real zone, not a frozen -4 timedelta."""
    assert dash.EDT is not None
    jan = datetime(2026, 1, 15, 12, 0, tzinfo=dash.EDT)
    jul = datetime(2026, 7, 15, 12, 0, tzinfo=dash.EDT)
    assert jan.utcoffset().total_seconds() == -5 * 3600, "January must be UTC-5 (EST)"
    assert jul.utcoffset().total_seconds() == -4 * 3600, "July must be UTC-4 (EDT)"


def test_eastern_zone_is_actually_dst_aware():
    """The old value could not tell EST from EDT. A ZoneInfo can."""
    assert dash.EDT == EDT_ZONE
    winter = datetime(2026, 1, 15, 12, 0, tzinfo=dash.EDT)
    assert winter.tzname() == "EST"
    summer = datetime(2026, 7, 15, 12, 0, tzinfo=dash.EDT)
    assert summer.tzname() == "EDT"


def test_india_offset_is_unchanged():
    """IST has no DST, so the refactor must not disturb it."""
    jan = datetime(2026, 1, 15, 12, 0, tzinfo=dash.IST)
    assert jan.utcoffset().total_seconds() == 5.5 * 3600


@pytest.mark.parametrize(
    "utc_moment,expect_open,why",
    [
        # 14:30 UTC == 09:30 EST (winter) -> open at the bell
        (datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc), True, "EST open"),
        # 20:45 UTC == 15:45 EST (winter) -> OPEN. The frozen -4 build computed
        # 16:45 here and called it closed, an hour early, for ~5 months a year.
        (datetime(2026, 1, 15, 20, 45, tzinfo=timezone.utc), True, "EST pre-close"),
        # 21:00 UTC == 16:00 EST -> the close boundary, still open
        (datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc), True, "EST close"),
        # 21:01 UTC == 16:01 EST -> closed
        (datetime(2026, 1, 15, 21, 1, tzinfo=timezone.utc), False, "EST after close"),
        # 13:30 UTC == 09:30 EDT (summer) -> open at the bell
        (datetime(2026, 7, 15, 13, 30, tzinfo=timezone.utc), True, "EDT open"),
        # 19:45 UTC == 15:45 EDT (summer) -> open
        (datetime(2026, 7, 15, 19, 45, tzinfo=timezone.utc), True, "EDT pre-close"),
        # 20:00 UTC == 16:00 EDT -> close boundary
        (datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc), True, "EDT close"),
        # 20:01 UTC == 16:01 EDT -> closed
        (datetime(2026, 7, 15, 20, 1, tzinfo=timezone.utc), False, "EDT after close"),
        # 21:00 UTC == 17:00 EDT -> closed (the frozen -4 build agreed here,
        # which is why the bug was invisible in summer and only bit in winter)
        (datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc), False, "EDT well after close"),
        # Saturday
        (datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc), False, "weekend"),
    ],
)
def test_is_us_open_at_known_utc_moments(monkeypatch, utc_moment, expect_open, why):
    """Pins the market-hours logic against wall-clock instants in both DST halves."""
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_moment.astimezone(tz) if tz else utc_moment

    monkeypatch.setattr(mkt, "datetime", _FrozenDatetime)
    assert mkt.is_us_open() is expect_open, why


def test_minutes_to_close_uses_et_close_in_winter(monkeypatch):
    """21:00 UTC is the EST close, so 20:45 UTC is 15 minutes to close."""
    moment = datetime(2026, 1, 15, 20, 45, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz) if tz else moment

    monkeypatch.setattr(mkt, "datetime", _FrozenDatetime)
    assert mkt.minutes_to_close("us") == pytest.approx(15.0, abs=1.0)


def test_minutes_to_close_uses_et_close_in_summer(monkeypatch):
    """20:00 UTC is the EDT close, so 19:45 UTC is 15 minutes to close."""
    moment = datetime(2026, 7, 15, 19, 45, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz) if tz else moment

    monkeypatch.setattr(mkt, "datetime", _FrozenDatetime)
    assert mkt.minutes_to_close("us") == pytest.approx(15.0, abs=1.0)
