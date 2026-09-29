"""Regression tests for keeping blocking Supabase I/O out of the state lock.

Every persist happened inside `with _lock:`. Under gunicorn with one worker and
eight threads, a single slow Supabase round-trip stalled all eight request
threads *and* the agent thread, and blew the 120 s gunicorn timeout part-way
through apply_cycle — killing a live trading cycle.
"""
import ast
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "apex_dashboard.py"

BLOCKING = {"save_state", "load_state", "save_cfg", "load_cfg", "_save_rl_decision"}


def _iter_with_lock_blocks():
    tree = ast.parse(DASH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(
            isinstance(i.context_expr, ast.Name) and i.context_expr.id == "_lock"
            for i in node.items
        ):
            continue
        calls = {
            s.func.id
            for s in ast.walk(node)
            if isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
        }
        blocking = calls & BLOCKING
        if blocking:
            yield node.lineno, sorted(blocking)


def test_no_blocking_io_inside_the_state_lock():
    """Structural guard: this defect is easy to reintroduce one line at a time."""
    offenders = list(_iter_with_lock_blocks())
    assert not offenders, (
        f"blocking I/O inside `with _lock` — persist must happen after the lock "
        f"is released: {offenders}"
    )


class _SlowTable:
    """Supabase table stub that records whether the state lock was held at the
    moment execute() was called, then blocks briefly."""

    def __init__(self, delay=0.4, lock=None):
        self.delay = delay
        self.lock = lock
        self.calls = 0
        self.lock_held_during_call = []

    def select(self, *a, **kw):
        return self

    def eq(self, *a, **kw):
        return self

    def upsert(self, *a, **kw):
        return self

    def insert(self, *a, **kw):
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def execute(self):
        self.calls += 1
        if self.lock is not None:
            self.lock_held_during_call.append(self.lock.locked())
        time.sleep(self.delay)
        return type("R", (), {"data": None})()


class _SlowSupabase:
    def __init__(self, delay=0.4, lock=None):
        self.table_stub = _SlowTable(delay, lock)

    def table(self, name):
        return self.table_stub


@pytest.fixture
def slow_sb(dash, monkeypatch):
    stub = _SlowSupabase(delay=0.4, lock=dash._lock)
    monkeypatch.setattr(dash, "_sb", stub)
    return stub


def test_slow_persist_does_not_hold_the_state_lock(dash, slow_sb, client):
    """A blocking upsert must not be issued with the state lock held.

    Deterministic: the fake client records lock.locked() at the moment execute()
    is called, so this cannot pass or fail on thread timing.
    """
    with dash._lock:
        dash._state["india"] = dash._empty_mstate(100000.0, "2026-01-01")

    r = client.post("/api/edit/state", json={"india": {"cash": 90000.0}})

    assert r.status_code == 200
    assert slow_sb.table_stub.calls >= 1, "no upsert reached Supabase"
    assert slow_sb.table_stub.lock_held_during_call, "upsert was never exercised"
    assert not any(slow_sb.table_stub.lock_held_during_call), (
        "blocking Supabase I/O was issued while holding _lock — under "
        "--workers 1 --threads 8 this stalls every request and the agent thread"
    )


def test_state_lock_is_free_during_a_persist(dash, slow_sb, client):
    """A second thread must be able to take the lock while a persist is in flight."""
    acquired = threading.Event()

    with dash._lock:
        dash._state["us"] = dash._empty_mstate(1000.0, "2026-01-01")

    t = threading.Timer(0.05, lambda: (dash._lock.acquire(), acquired.set(), dash._lock.release()))
    t.start()
    client.post("/api/edit/state", json={"us": {"cash": 500.0}})
    t.cancel()
    assert acquired.is_set(), "state lock was held for the whole duration of the persist"


def test_state_is_still_persisted(dash, slow_sb, client):
    """Moving I/O out of the lock must not skip the write."""
    client.post("/api/edit/state", json={"india": {"cash": 12345.0}})
    assert slow_sb.table_stub.calls >= 1, "no upsert reached Supabase"


def test_snapshot_does_not_alias_live_state(dash):
    """The snapshot handed to the writer must be a copy, not the live dict."""
    with dash._lock:
        dash._state["india"] = dash._empty_mstate(1000.0, "2026-01-01")
        snap = dash.snapshot_state()
        dash._state["india"]["cash"] = 777.0
    assert snap["india"]["cash"] == 1000.0, "snapshot aliases live state"
