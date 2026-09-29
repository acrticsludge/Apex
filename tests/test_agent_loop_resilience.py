"""The agent thread must not die on a bad cycle.

A KeyError inside agent_loop killed the thread outright: the bot stopped trading
and nothing said so, because the exception escaped a background thread and only
surfaced as a pytest warning. In production that is a silent trading outage.

These tests pin the two halves of the fix: session rotation tolerates a market
that has not been loaded, and a failing cycle is logged and survived.
"""
import threading
import time
import types
from datetime import datetime

import pytest

import apex_dashboard as dash


# ── Session rotation tolerates a partially-loaded state ──────────────────────

def test_session_rotation_tolerates_a_missing_market(dash):
    """A state missing one market must not raise."""
    with dash._lock:
        dash._state["india"] = dash._empty_mstate(1000.0, datetime.now(dash.IST).strftime("%Y-%m-%d"))
        dash._state.pop("us", None)
    with dash._lock:
        dash._check_session_rotation({})     # must not raise
    dash._state.clear()


def test_session_rotation_tolerates_an_empty_state(dash):
    with dash._lock:
        dash._state.clear()
    with dash._lock:
        dash._check_session_rotation({})
    dash._state.clear()


def test_session_rotation_still_archives_an_active_session(dash, monkeypatch):
    """The tolerance must not stop the real rollover behaviour."""
    with dash._lock:
        st = dash._empty_mstate(1000.0, "1999-01-01")
        st["wins"] = 3
        dash._state["india"] = st
        dash._state["us"] = dash._empty_mstate(1000.0, datetime.now(dash.EDT).strftime("%Y-%m-%d"))

    archived = []
    monkeypatch.setattr(dash, "_close_session",
                        lambda m, p: archived.append(m) or dash._state.get(m, {}))
    with dash._lock:
        dash._check_session_rotation({})
    assert "india" in archived, "an active stale session was not archived"
    dash._state.clear()


# ── A failing cycle is survived and logged ──────────────────────────────────

def test_agent_loop_survives_a_failing_cycle(dash, monkeypatch):
    """One bad cycle must not end the trading thread."""
    dash._agent["running"] = True
    dash._agent["paused"] = False
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("cycle exploded")

    monkeypatch.setattr(dash, "is_india_open", lambda: True)
    monkeypatch.setattr(dash, "is_us_open", lambda: True)
    monkeypatch.setattr(dash, "_check_session_rotation", boom)
    monkeypatch.setattr(dash, "save_state", lambda st: None)
    monkeypatch.setattr(dash, "persist_state", lambda s: None)
    monkeypatch.setattr(dash, "fetch_cycle_data", lambda *a, **k: ([], {}))
    # Shorten only the agent's long back-off sleeps. Patching time.sleep directly
    # would also neuter this test's own wait loop, so proxy it instead.
    real_time = dash.time
    monkeypatch.setattr(
        dash, "time",
        types.SimpleNamespace(
            sleep=lambda s: real_time.sleep(min(s, 0.01)),
            monotonic=real_time.monotonic,
            time=real_time.time,
        ),
    )

    errs = []
    orig = dash.apex_log.error

    def spy(msg, *a, **k):
        errs.append(msg % a if a else msg)
        return orig(msg, *a, **k)

    monkeypatch.setattr(dash.apex_log, "error", spy)

    t = threading.Thread(target=dash.agent_loop, daemon=True)
    t.start()
    deadline = time.time() + 5
    while calls["n"] < 3 and time.time() < deadline:
        time.sleep(0.02)
    dash._agent["running"] = False
    t.join(timeout=5)

    assert calls["n"] >= 3, f"loop died after {calls['n']} cycle(s) instead of retrying"
    assert any("cycle" in e.lower() for e in errs), f"failing cycle was not logged: {errs}"
    assert not t.is_alive(), "thread did not exit after running=False"


def test_agent_loop_warns_about_sustained_failure(dash, monkeypatch, caplog):
    """Repeated failures must escalate in the log: a stalled bot is the incident
    an operator has to see, and the thread will not die to tell them."""
    dash._agent["running"] = True
    dash._agent["paused"] = False

    monkeypatch.setattr(dash, "is_india_open", lambda: True)
    monkeypatch.setattr(dash, "is_us_open", lambda: True)
    monkeypatch.setattr(
        dash, "_check_session_rotation",
        lambda *a: (_ for _ in ()).throw(RuntimeError("still broken")),
    )
    real_time = dash.time
    monkeypatch.setattr(
        dash, "time",
        types.SimpleNamespace(
            sleep=lambda s: real_time.sleep(min(s, 0.005)),
            monotonic=real_time.monotonic,
            time=real_time.time,
        ),
    )

    errors = []
    orig = dash.apex_log.error
    monkeypatch.setattr(
        dash.apex_log, "error",
        lambda msg, *a, **k: (errors.append(msg % a if a else msg), orig(msg, *a, **k))[1],
    )

    t = threading.Thread(target=dash.agent_loop, daemon=True)
    t.start()
    deadline = time.time() + 5
    while not any("consecutive" in e for e in errors) and time.time() < deadline:
        time.sleep(0.02)
    dash._agent["running"] = False
    t.join(timeout=5)

    assert any("consecutive" in e for e in errors), (
        f"sustained failure never escalated to an operator-visible warning: {errors}"
    )
    assert any("trading may be stalled" in e for e in errors), (
        f"warning did not say the bot may be stalled: {errors}"
    )
