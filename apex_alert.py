"""Outbound alerting for events that are otherwise invisible until you look.

Why this exists
---------------
The agent counted its own failures — `Agent has failed N consecutive cycles` was
already in the log — but nothing watched that line. For an unattended bot the
failure modes are all silent:

* cycles fail, the log grows, no one reads it, and the bot has been not trading
  for six hours;
* worse, the agent thread dies outright. The Flask process stays up, the
  dashboard keeps serving 200s, and nothing is logged at all, because a
  background-thread death writes nothing;
* the daily-loss or drawdown kill-switch trips, which is *deliberate* and
  therefore produces no error anywhere — it just stops trading.

None of these look like a problem from outside. A webhook is the cheapest thing
that turns them into a phone notification.

Design constraints
------------------
This is called from the trading thread's failure path, so it must never raise,
never block for long, and never retry-storm. A monitoring system that takes the
service down is worse than no monitoring.

* Every failure path returns a bool and records why, rather than raising or
  silently passing. A bare `pass` here would be indistinguishable from a webhook
  that is simply not configured.
* One delivery attempt, no retries. A webhook that is down fails fast; a retry
  loop would turn an outage into a self-inflicted one.
* Escalation, not repetition. A persistent failure alerts at 1, 3, 10, 30, 100
  consecutive occurrences and then stops. Alerting every cycle for six hours
  trains the operator to ignore the channel, which is the same as having no
  alerting at all.

Configuration
-------------
``APEX_ALERT_WEBHOOK_URL``   where to POST. Slack, Discord, ntfy, PagerDuty and
                             most incident tools accept a plain JSON POST.
``APEX_ALERT_TIMEOUT``       seconds, default 5.
``APEX_ALERT_ALWAYS``        set to 1 to alert even outside a market session.

With no URL configured nothing is sent and the state is exposed by
``/healthz``, so an external monitor can poll for liveness instead. That path is
the one that always works, because it needs no third party.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Consecutive occurrences at which a repeated event is announced. The final
# entry is the cap: past that, silence is deliberate, because a condition that
# has persisted for 100 cycles is not news.
_ESCALATION = (1, 3, 10, 30, 100)

_lock = threading.Lock()
_state: dict[str, dict] = {}
_noted_unconfigured = False


def webhook_url() -> str:
    return os.getenv("APEX_ALERT_WEBHOOK_URL", "").strip()


def is_configured() -> bool:
    return bool(webhook_url())


def timeout_seconds() -> float:
    try:
        return max(1.0, min(float(os.getenv("APEX_ALERT_TIMEOUT", "5")), 30.0))
    except (TypeError, ValueError):
        return 5.0


def reset(event: str | None = None) -> None:
    """Forget escalation state, so a recurring event alerts again from the start."""
    with _lock:
        if event is None:
            _state.clear()
        else:
            _state.pop(event, None)


def status() -> dict:
    """Alerting state for /healthz. Never raises."""
    with _lock:
        entries = list(_state.values())
        last_error = next(
            (e["last_error"] for e in reversed(entries) if e.get("last_error")),
            None,
        )
        return {
            "configured": is_configured(),
            # Summed across events, not counted per event: an operator asking
            # "how many alerts has this thing sent" needs the total.
            "sent": sum(int(e.get("sent", 0)) for e in entries),
            "suppressed": sum(int(e.get("suppressed", 0)) for e in entries),
            "events": len(entries),
            "last_error": last_error,
        }


def _should_send(event: str, count: int) -> bool:
    """Escalation gate. Caller holds the lock."""
    entry = _state.setdefault(event, {"sent": 0, "suppressed": 0, "last_error": None})
    for threshold in _ESCALATION:
        if count == threshold:
            entry["sent"] += 1
            return True
    entry["suppressed"] += 1
    return False


def alert(event: str, message: str, level: str = "error", **fields) -> bool:
    """Deliver one alert. Returns True if a webhook was actually called.

    Never raises. `event` is a stable identifier used for escalation; use the
    same string for every occurrence of the same condition.
    """
    global _noted_unconfigured

    url = webhook_url()
    if not url:
        with _lock:
            first_time = not _noted_unconfigured
            _noted_unconfigured = True
        if first_time:
            # Say so once. Alerting that is silently off is the worst kind,
            # because it looks configured.
            logger.warning(
                "Alerting is DISABLED: set APEX_ALERT_WEBHOOK_URL to be notified. "
                "Liveness is still pollable at /healthz."
            )
        return False

    with _lock:
        count = _state.setdefault(event, {"sent": 0, "suppressed": 0, "last_error": None})
        count = count["sent"] + count["suppressed"] + 1
        if not _should_send(event, count):
            return False

    payload = {
        "event": event,
        "level": level,
        "message": message,
        "occurrences": count,
        "env": os.getenv("APEX_ENV", "unset"),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: v for k, v in fields.items() if _serialisable(v)},
    }

    try:
        import requests

        requests.post(
            url,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds(),
        )
    except Exception as exc:  # noqa: BLE001
        # Cannot let a monitoring failure become a trading failure. Record it
        # visibly so /healthz can report that alerting itself is broken.
        with _lock:
            _state[event]["last_error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Alert delivery failed for %s: %s", event, exc)
        return False

    logger.warning("Alert raised: %s — %s", event, message)
    return True


def _serialisable(value) -> bool:
    """Keep the payload JSON-safe without importing anything at call time."""
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))


class Heartbeat:
    """Liveness state for /healthz.

    This is the half of alerting that does not need a third party: the process
    reports whether the agent is actually cycling, and an external monitor polls
    it. A thread that dies silently leaves a stale timestamp, which is the whole
    signal — there is no error to log when a thread simply stops existing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_cycle: float | None = None
        self._started: float = time.time()

    def beat(self) -> None:
        """Record a successful cycle."""
        with self._lock:
            self._last_cycle = time.time()

    def snapshot(self, *, running: bool, status: str,
                 consecutive_failures: int = 0,
                 stale_after_s: int = 900) -> dict:
        """Liveness as a dict. `stale_after_s` is how long a cycle may be absent
        before this counts as stale — 15 minutes by default, comfortably longer
        than any legitimate sleep between cycles."""
        with self._lock:
            last = self._last_cycle
            started = self._started
        now = time.time()
        age = None if last is None else round(now - last, 1)
        stale = last is None or (now - last) > stale_after_s
        healthy = bool(running) and not stale and consecutive_failures == 0
        return {
            "healthy": healthy,
            "agent_running": bool(running),
            "agent_status": status,
            "consecutive_failures": consecutive_failures,
            "last_cycle_age_s": age,
            "stale": stale,
            "uptime_s": round(now - started, 1),
        }


heartbeat = Heartbeat()
