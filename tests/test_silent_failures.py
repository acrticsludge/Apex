"""Regression tests for defects that failed silently.

A bare `except: pass` around the RL feedback path meant the online learner never
trained, and nothing anywhere said so. Separately, a dead price feed left the
last good price in place forever, so a symbol could trade on a stale quote
indefinitely.
"""
import ast
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_SKIP_DIRS = {".venv", "venv", "build", "dist", "graphify-out", "__pycache__",
              ".git", ".pytest_cache", ".ruff_cache", ".ruff", "node_modules"}


def _iter_project_py():
    """Project .py files, pruning vendored/build dirs (rglob would walk .venv)."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


# ── H7: RL feedback failures were swallowed ──────────────────────────────────

def test_record_exit_failure_is_logged_not_swallowed(dash, monkeypatch, caplog):
    """execute_sell feeds the closed trade to the online learner. A failure
    there used to hit `except: pass`, leaving _pending empty forever."""
    calls = {}

    def boom(*a, **kw):
        raise RuntimeError("learner exploded")

    import trading_agent.integration.online_learner as ol
    monkeypatch.setattr(ol, "record_exit", boom)

    mstate = dash._empty_mstate(100000.0, "2026-01-01")
    mstate["positions"]["X.NS"] = {
        "qty": 10, "entry": 100.0, "stop_loss": 95.0, "target": 110.0, "side": "long",
    }
    with caplog.at_level("WARNING", logger="apex"):
        dash.execute_sell("X.NS", 110.0, "TARGET", mstate)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("record_exit" in w or "learner" in w.lower() for w in warnings), (
        f"RL feedback failure produced no warning; records: {warnings}"
    )


def test_record_entry_failure_is_logged_not_swallowed(dash, monkeypatch, caplog):
    import trading_agent.integration.rl_signal as rs
    import trading_agent.integration.online_learner as ol
    monkeypatch.setattr(rs, "get_cached_obs", lambda sym: [0.0] * 8)
    monkeypatch.setattr(
        ol, "record_entry", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("entry boom"))
    )
    with caplog.at_level("WARNING", logger="apex"):
        try:
            dash.rl_feedback_entry("X.NS", 1, 100.0, 2.0)
        except Exception as e:
            pytest.fail(f"rl_feedback_entry propagated: {e}")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("record_entry" in w for w in warnings), (
        f"entry-feedback failure produced no warning; records: {warnings}"
    )


def test_no_broad_except_that_swallows_silently():
    """A bare `pass` under `except Exception` is how every one of these defects
    hid. Narrow handlers (ImportError for an optional dep, ValueError from
    removing an absent key) are allowed — they are precise about what they ignore.
    """
    ALLOWED = {"ImportError", "ValueError", "KeyError", "AttributeError", "TypeError"}
    offenders = []
    for path in _iter_project_py():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = node.type
            names = []
            if caught is None:
                names = ["<bare except>"]
            elif isinstance(caught, ast.Name):
                names = [caught.id]
            elif isinstance(caught, ast.Tuple):
                names = [e.id for e in caught.elts if isinstance(e, ast.Name)]
            if not ({"<bare except>"} & set(names)) and set(names) <= ALLOWED:
                continue
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} ({','.join(names)})")
    assert not offenders, f"broad excepts that swallow silently: {offenders}"


# ── H7: a dead price feed left a stale quote in place forever ───────────────

def test_price_age_is_tracked(dash):
    with dash._price_lock:
        dash._latest_prices.clear()
        dash._price_ts.clear()
        dash._record_prices({"AAPL": 100.0})
        assert "AAPL" in dash._price_ts


def test_stale_price_is_rejected(dash):
    """A symbol whose feed died kept trading on its last good price."""
    with dash._price_lock:
        dash._latest_prices.clear()
        dash._price_ts.clear()
        dash._record_prices({"AAPL": 100.0})
    assert dash.price_is_fresh("AAPL")

    # Backdate the quote past the staleness window.
    with dash._price_lock:
        dash._price_ts["AAPL"] = time.time() - (dash.PRICE_MAX_AGE_SECONDS + 60)
    assert not dash.price_is_fresh("AAPL"), "stale price still considered fresh"

    with dash._price_lock:
        dash._latest_prices.clear()
        dash._price_ts.clear()


def test_freshness_is_configurable_and_positive(dash):
    assert dash.PRICE_MAX_AGE_SECONDS > 0
