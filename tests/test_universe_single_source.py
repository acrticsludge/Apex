"""Regression tests for a single source of truth for the trading universe.

trading_agent/config.py AST-parsed apex_dashboard.py to copy `cfg` and the two
watchlists out of it without importing the Flask app. That made the RL package
silently depend on a literal dict staying literal: the moment anyone made `cfg`
computed, the scrape returned {} and the RL universe quietly became the
hand-maintained FALLBACK_* copies — a second, drifting set of symbols.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "trading_agent" / "config.py"

_skip = {".venv", "venv", "build", "dist", "graphify-out", "__pycache__", ".git"}


def _project_py():
    for dirpath, dirnames, filenames in __import__("os").walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _skip]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def test_config_does_not_ast_parse_the_dashboard():
    src = CONFIG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "literal_eval" not in calls, "config.py still literal_evals the dashboard source"
    assert "_extract_dashboard_objects" not in src, "dashboard scraper still present"


def test_dashboard_cfg_reference_is_gone():
    """dashboard_reference_cfg had zero consumers."""
    src = CONFIG.read_text(encoding="utf-8")
    assert "dashboard_reference_cfg" not in src, (
        "dead dashboard_reference_cfg field still present"
    )
    assert "DEFAULT_DASHBOARD_CFG" not in src


def test_watchlists_have_exactly_one_definition():
    """Four copies of the same 32-symbol list used to exist."""
    definitions = []
    for path in _project_py():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in tree.body:
            targets = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else ([node.target.id] if isinstance(node, ast.AnnAssign)
                      and isinstance(node.target, ast.Name) else [])
            )
            for name in targets:
                if name in {"INDIA_WATCHLIST", "US_WATCHLIST"}:
                    definitions.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    assert len(definitions) == 2, (
        f"watchlists must be defined once each, found: {definitions}"
    )


def test_rl_universe_matches_the_dashboard():
    """The RL package must train on exactly what the dashboard trades."""
    import apex_dashboard
    from trading_agent.config import settings

    assert list(settings.dashboard_india_watchlist) == list(apex_dashboard.INDIA_WATCHLIST)
    assert list(settings.dashboard_us_watchlist) == list(apex_dashboard.US_WATCHLIST)


def test_rl_active_tickers_cover_both_markets():
    from trading_agent.config import settings

    tickers = set(settings.active_tickers)
    for sym in list(apex_dashboard_india()) + list(apex_dashboard_us()):
        assert sym in tickers, f"{sym} traded by the dashboard but absent from the RL universe"


def apex_dashboard_india():
    import apex_dashboard
    return apex_dashboard.INDIA_WATCHLIST


def apex_dashboard_us():
    import apex_dashboard
    return apex_dashboard.US_WATCHLIST
