"""The SPA used to be a 1629-line HTML/CSS/JS string literal in the Python module,
re-tokenised by Jinja on every request. It now lives in templates/ and static/.

These tests pin the extraction: the served document must still contain every
marker it did before, and the assets must actually be reachable.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "templates" / "index.html"
CSS = ROOT / "static" / "app.css"
JS = ROOT / "static" / "app.js"


def test_no_large_html_literals_remain_in_the_module():
    src = (ROOT / "apex_dashboard.py").read_text(encoding="utf-8", errors="replace")
    assert "LOGIN_HTML = " not in src, "login template literal still inline"
    assert "\nHTML = " not in src, "SPA template literal still inline"
    assert "render_template_string" not in src, (
        "render_template_string re-tokenises on every request; use render_template"
    )


def test_template_and_assets_exist():
    for p in (TPL, CSS, JS, ROOT / "templates" / "login.html"):
        assert p.exists(), f"missing {p.relative_to(ROOT)}"
        assert p.stat().st_size > 0, f"{p.relative_to(ROOT)} is empty"


def test_dashboard_serves_the_template(client, dash):
    with client.session_transaction() as s:
        s["logged_in"] = True
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "<!DOCTYPE html>" in body
    assert "/static/app.css" in body
    assert "/static/app.js" in body


def test_assets_are_served(client, dash):
    for path, marker in (("/static/app.css", "app.css"), ("/static/app.js", "app.js")):
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        assert len(r.get_data()) > 100, f"{path} looks empty"


def test_login_page_renders(client, dash):
    r = client.get("/login")
    assert r.status_code == 200
    assert "form" in r.get_data(as_text=True).lower()


@pytest.mark.parametrize(
    "marker",
    [
        # structural elements the dashboard cannot work without
        "settings-toggle", "s-risk", "s-conf", "s-sl", "s-tgt",
        "s-chk", "s-idle", "s-ip", "s-up", "s-eod-h", "s-eod-e", "s-rl-exit",
        "last-upd",
    ],
)
def test_controls_survived_the_move(marker):
    """Every id the JS binds to must still exist somewhere in markup or JS."""
    blob = TPL.read_text(encoding="utf-8") + JS.read_text(encoding="utf-8")
    assert marker in blob, f"UI control {marker!r} lost in extraction"


@pytest.mark.parametrize(
    "fn",
    [
        "saveConfig", "loadCfg", "toggleSettings", "renderAll", "doRefresh",
    ],
)
def test_js_functions_survived_the_move(fn):
    assert fn in JS.read_text(encoding="utf-8"), f"{fn}() lost in extraction"


def test_js_has_no_python_string_artifacts():
    js = JS.read_text(encoding="utf-8")
    for bad in ('r"""', '"""', "\\n\\\"", "None,"):
        assert bad not in js, f"Python artifact {bad!r} leaked into app.js"


def test_static_assets_are_git_tracked_candidates():
    """templates/ and static/ must be committed or the deploy serves nothing."""
    for name in ("templates", "static"):
        assert (ROOT / name).is_dir()
        for f in (ROOT / name).iterdir():
            assert f.is_file(), f"{f} must be a plain file"
