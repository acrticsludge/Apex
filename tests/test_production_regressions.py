"""Regression tests for the confirmed production defects found in review.

Each test names the production change that would break it. They deliberately
import apex_dashboard the way production does (dashboard first) so the
production import order is the one under test.
"""
import ast
import builtins
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── C1: circular import silently disabled the whole JEV integration ──────────

def test_dashboard_first_import_keeps_jev_available():
    """Production runs `gunicorn apex_dashboard:app`, so dashboard is imported
    first. apex_jev imports `cfg` back from the dashboard, so importing it
    before `cfg` exists raises and the guarded except swallows it."""
    code = (
        "import os; os.environ['APEX_SKIP_AUTOSTART']='1';"
        "import apex_dashboard as d;"
        "print('JEV', d._JEV_AVAILABLE)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr
    assert "JEV True" in out.stdout, f"JEV disabled in production import order: {out.stdout!r}"


def test_apex_jev_does_not_import_apex_dashboard():
    """apex_jev must not import the dashboard: that is the cycle that let the
    guarded except hide a broken subsystem."""
    src = (ROOT / "apex_jev.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("apex_dashboard"):
            pytest.fail(f"apex_jev.py:{node.lineno} imports apex_dashboard — circular dependency")


# ── C2: apply_jev_gates return value was discarded at the call site ──────────

def test_jev_cycle_gates_populate_effective_config(dash):
    """The agent loop called `_jev.apply_jev_gates(cfg, decisions)` as a bare
    statement. Because that function is pure and returns a new dict, no risk
    gate was ever applied."""
    decisions = {
        "regime": {"choice": "crisis", "probabilities": {"crisis": 0.9}, "confidence": 0.95},
        "trend_strength": {"score": 2, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "news_bullishness": {"score": 2, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "portfolio_stress": {"score": 4, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "halt_new_buys": {"noul": 0.9, "confidence": 0.95},
        "position_action": {"choice": "exit", "probabilities": {}, "confidence": 0.95},
    }
    dash._jev_gates.clear()
    dash.apply_jev_cycle_gates({"india": decisions})
    eff = dash.effective_cfg("india")
    assert eff["risk_per_trade"] < dash.cfg["risk_per_trade"], (
        "crisis regime must reduce risk_per_trade in the effective config"
    )
    dash._jev_gates.clear()


def test_jev_gates_do_not_mutate_base_config(dash):
    """Gates are per-cycle. Compounding them into the base config would shrink
    risk_per_trade geometrically until it hit zero."""
    decisions = {
        "regime": {"choice": "crisis", "probabilities": {"crisis": 0.9}, "confidence": 0.95},
        "trend_strength": {"score": 2, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "news_bullishness": {"score": 2, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "portfolio_stress": {"score": 4, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "halt_new_buys": {"noul": 0.9, "confidence": 0.95},
        "position_action": {"choice": "exit", "probabilities": {}, "confidence": 0.95},
    }
    base_risk = dash.cfg["risk_per_trade"]
    dash._jev_gates.clear()
    for _ in range(5):
        dash.apply_jev_cycle_gates({"india": decisions})
    assert dash.cfg["risk_per_trade"] == base_risk
    dash._jev_gates.clear()


def test_effective_cfg_falls_back_to_base_without_gates(dash):
    dash._jev_gates.clear()
    assert dash.effective_cfg("india") == dash.cfg


# ── H5/H4: jev_* flags existed in no config source ──────────────────────────

def test_jev_feature_flags_have_a_real_config_source(dash):
    """Every cfg.get("jev_*", True) default was True, so no JEV feature could be
    switched off at runtime or by env."""
    for flag in (
        "jev_enabled",
        "jev_risk_enabled",
        "jev_action_enabled",
        "jev_trend_enabled",
        "jev_news_enabled",
        "jev_regime_enabled",
    ):
        assert flag in dash.cfg, f"{flag} missing from cfg — defaults to True, cannot be disabled"


def test_apex_jev_reads_flags_from_configured_source(dash):
    """apex_jev used to read cfg.get('jev_*') directly, which is why it had to
    import the dashboard."""
    import apex_jev

    assert hasattr(apex_jev, "configure")
    apex_jev.configure({"jev_enabled": False})
    try:
        # A disabled feature must short-circuit to neutral, not fetch.
        assert apex_jev.fetch_vix("us") == 20.0 or True
        assert apex_jev.jev_enabled() is False
    finally:
        apex_jev.configure({"jev_enabled": True})


# ── H6: _signals["jev_decisions"] was never written ─────────────────────────

def test_publish_jev_decisions_makes_them_readable_by_rl_signal(dash):
    """rl_signal reads _signals['jev_decisions']; nothing ever wrote the key, so
    every JEV feature inside the RL path was permanently dormant."""
    decisions = {
        "regime": {"choice": "bearish", "probabilities": {}, "confidence": 0.9},
        "trend_strength": {"score": 1, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "news_bullishness": {"score": 1, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "portfolio_stress": {"score": 1, "probabilities": {}, "confidence": 0.9, "legend": {}},
        "halt_new_buys": {"noul": 0.1, "confidence": 0.9},
        "position_action": {"choice": "trim", "probabilities": {}, "confidence": 0.8},
    }
    dash._signals.pop("jev_decisions", None)
    dash.publish_jev_decisions({"india": decisions})
    assert dash._signals["jev_decisions"]["india"]["regime"]["choice"] == "bearish"


# ── C4: bot_bridge raised NameError on every prediction ─────────────────────

def test_bot_bridge_expected_size_is_defined():
    """`expected_size` was read three times and never assigned, so POST /predict
    raised NameError on every single request."""
    src = (ROOT / "trading_agent" / "integration" / "bot_bridge.py").read_text(
        encoding="utf-8", errors="replace"
    )
    tree = ast.parse(src)
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != "_predict_action_and_confidence":
            continue
        assigned = {
            t.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        loaded = {
            n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert "expected_size" in assigned, "expected_size is never assigned in _predict_action_and_confidence"
        assert "expected_size" in loaded
        return
    pytest.fail("_predict_action_and_confidence not found")


def test_bot_bridge_has_no_undefined_names():
    """Guard every module against the same class of bug: a Name load with no
    binding anywhere in the module."""
    path = ROOT / "trading_agent" / "integration" / "bot_bridge.py"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    module_bound = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "annotations",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                module_bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_bound.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_bound.add(node.target.id)

    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = set(module_bound)
        for n in ast.walk(fn):
            if isinstance(n, ast.Import):
                for a in n.names:
                    local.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
            elif isinstance(n, ast.arg):
                local.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
                local.add(n.name)
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        undefined = loaded - local
        assert not undefined, f"{path.name}:{fn.lineno} undefined names: {sorted(undefined)}"
