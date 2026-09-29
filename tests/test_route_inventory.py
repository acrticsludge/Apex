"""Route inventory.

During the module decomposition a refactor silently deleted the
`/api/edit/state` route. The suite caught it (the edit tests 404'd), but only
by luck of ordering. This pins the surface explicitly, so a removed or renamed
route fails loudly and names itself.
"""
import pytest

import apex_dashboard as dash

EXPECTED_ROUTES = {
    # auth
    ("/", "GET"),
    ("/login", "GET"),
    ("/login", "POST"),
    ("/logout", "GET"),
    # reads
    ("/api/state", "GET"),
    ("/api/logs", "GET"),
    ("/api/prices/stream", "GET"),
    ("/api/think", "GET"),
    ("/api/jev/status", "GET"),
    ("/api/retrain/log", "GET"),
    ("/api/rl/decisions", "GET"),
    ("/api/sessions", "GET"),
    # agent control (mutating)
    ("/api/agent/start", "POST"),
    ("/api/agent/stop", "POST"),
    ("/api/agent/pause", "POST"),
    ("/api/sell/<market>/<path:symbol>", "POST"),
    # settings + ledger (mutating)
    ("/api/config", "POST"),
    ("/api/reset/<market>", "POST"),
    ("/api/edit/state", "POST"),
    ("/api/edit/position/<market>/<path:symbol>", "POST"),
    ("/api/think/clear", "POST"),
    ("/api/sessions/close/<market>", "POST"),
    # static
    ("/static/<path:filename>", "GET"),
}


def _actual():
    out = set()
    for rule in dash.app.url_map.iter_rules():
        if rule.endpoint == "static":
            out.add(("/static/<path:filename>", "GET"))
            continue
        methods = rule.methods - {"HEAD", "OPTIONS"}
        for m in methods:
            out.add((str(rule.rule), m))
    return out


def test_no_route_disappeared():
    actual = _actual()
    missing = EXPECTED_ROUTES - actual
    assert not missing, f"routes removed or renamed: {sorted(missing)}"


def test_no_unexpected_route_appeared():
    actual = _actual()
    extra = actual - EXPECTED_ROUTES
    assert not extra, f"undocumented routes: {sorted(extra)}"


@pytest.mark.parametrize("path,method", sorted(EXPECTED_ROUTES))
def test_route_is_reachable(client, path, method):
    """Every documented route answers, and the auth gate never 500s."""
    concrete = (
        path.replace("<market>", "india")
            .replace("<path:symbol>", "TEST.NS")
            .replace("<path:filename>", "app.js")
    )
    r = client.open(concrete, method=method)
    assert r.status_code < 500, f"{method} {concrete} -> {r.status_code}"


# Every API route that is safe over GET. Anything else must be POST-only, so a
# stray link or prefetch can never move the ledger.
SAFE_GET_API_ROUTES = {
    "/api/state",
    "/api/logs",
    "/api/prices/stream",
    "/api/think",
    "/api/jev/status",
    "/api/retrain/log",
    "/api/rl/decisions",
    "/api/sessions",
}


def test_mutating_routes_do_not_answer_get(client, dash):
    """A GET must never start/stop the agent, sell, or rewrite the ledger."""
    offenders = []
    for rule in dash.app.url_map.iter_rules():
        path = str(rule.rule)
        if not path.startswith("/api/") or path in SAFE_GET_API_ROUTES:
            continue
        if "GET" in (rule.methods or set()):
            offenders.append(path)
    assert not offenders, f"state-mutating routes answer GET: {offenders}"


@pytest.mark.parametrize("path", sorted(SAFE_GET_API_ROUTES))
def test_read_routes_stay_read_only(dash, path):
    """Sanity: the read allowlist really is the set of GET-capable API routes."""
    for rule in dash.app.url_map.iter_rules():
        if str(rule.rule) == path:
            assert "GET" in (rule.methods or set())
            return
    pytest.fail(f"{path} is not registered")
