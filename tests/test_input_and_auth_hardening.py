"""Regression tests for the unvalidated-write and auth defects found in review.

The edit endpoints wrote client-supplied values straight into the live trading
ledger. A negative qty made execute_sell subtract cash and inflate realised_pnl.
"""
from pathlib import Path

import pytest


# â”€â”€ C3: /api/edit/position accepted negative and zero values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@pytest.fixture
def seeded(dash):
    """A market state with one long position to edit."""
    with dash._lock:
        dash._state["india"] = dash._empty_mstate(100000.0, "2026-01-01")
        dash._state["india"]["positions"]["TEST.NS"] = {
            "qty": 10, "entry": 100.0, "stop_loss": 95.0,
            "target": 110.0, "atr": 2.0, "side": "long",
        }
    yield dash
    with dash._lock:
        dash._state.pop("india", None)


def test_edit_position_rejects_negative_qty(client, seeded):
    r = client.post("/api/edit/position/india/TEST.NS", json={"qty": -500})
    assert r.status_code == 400
    assert seeded._state["india"]["positions"]["TEST.NS"]["qty"] == 10


def test_edit_position_rejects_zero_qty(client, seeded):
    r = client.post("/api/edit/position/india/TEST.NS", json={"qty": 0})
    assert r.status_code == 400
    assert seeded._state["india"]["positions"]["TEST.NS"]["qty"] == 10


def test_edit_position_rejects_non_numeric_qty(client, seeded):
    r = client.post("/api/edit/position/india/TEST.NS", json={"qty": "lots"})
    assert r.status_code == 400
    assert seeded._state["india"]["positions"]["TEST.NS"]["qty"] == 10


def test_edit_position_rejects_negative_prices(client, seeded):
    for field in ("entry", "stop_loss", "target"):
        r = client.post("/api/edit/position/india/TEST.NS", json={field: -1.0})
        assert r.status_code == 400, f"{field} accepted a negative value"
    pos = seeded._state["india"]["positions"]["TEST.NS"]
    assert pos["entry"] == 100.0 and pos["stop_loss"] == 95.0 and pos["target"] == 110.0


def test_edit_position_rejects_non_finite_prices(client, seeded):
    for field in ("entry", "stop_loss", "target"):
        r = client.post("/api/edit/position/india/TEST.NS", json={field: float("inf")})
        assert r.status_code == 400, f"{field} accepted inf"


def test_edit_position_accepts_valid_update(client, seeded):
    r = client.post("/api/edit/position/india/TEST.NS", json={"qty": 25, "stop_loss": 92.0})
    assert r.status_code == 200
    pos = seeded._state["india"]["positions"]["TEST.NS"]
    assert pos["qty"] == 25 and pos["stop_loss"] == 92.0


def test_edit_state_rejects_negative_cash(client, seeded):
    r = client.post("/api/edit/state", json={"india": {"cash": -50000.0}})
    assert r.status_code == 400
    assert seeded._state["india"]["cash"] > 0


def test_edit_state_rejects_negative_win_loss_counts(client, seeded):
    r = client.post("/api/edit/state", json={"india": {"wins": -5, "losses": -3}})
    assert r.status_code == 400
    assert seeded._state["india"]["wins"] == 0
    assert seeded._state["india"]["losses"] == 0


def test_edit_state_rejects_non_finite_cash(client, seeded):
    r = client.post("/api/edit/state", json={"india": {"cash": float("nan")}})
    assert r.status_code == 400


# â”€â”€ H1: /api/config accepted arbitrary risk parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@pytest.mark.parametrize(
    "payload",
    [
        {"risk_per_trade": 50},
        {"risk_per_trade": -1},
        {"stop_loss_pct": -0.5},
        {"stop_loss_pct": 12},
        {"target_pct": -0.1},
        {"confidence_threshold": -999},
        {"confidence_threshold": 5000},
        {"india_max_positions": 1000000},
        {"india_max_positions": 0},
        {"us_max_positions": -3},
        {"rl_exit_confidence": 9000},
        {"eod_harvest_min": -5},
        {"check_interval_min": 0},
    ],
)
def test_config_rejects_out_of_range_values(client, dash, payload):
    before = dict(dash.cfg)
    r = client.post("/api/config", json=payload)
    assert r.status_code == 400, f"{payload} was accepted"
    for k in payload:
        assert dash.cfg[k] == before[k], f"{k} was mutated despite the rejection"


def test_config_rejects_non_numeric_value(client, dash):
    r = client.post("/api/config", json={"risk_per_trade": "all of it"})
    assert r.status_code == 400


def test_config_rejects_unknown_key(client, dash):
    r = client.post("/api/config", json={"not_a_real_setting": 1})
    assert r.status_code == 400
    assert "not_a_real_setting" not in dash.cfg


def test_config_accepts_sane_values(client, dash):
    before = dash.cfg["risk_per_trade"]
    r = client.post("/api/config", json={"risk_per_trade": 0.01, "confidence_threshold": 70})
    assert r.status_code == 200
    assert dash.cfg["risk_per_trade"] == 0.01
    assert dash.cfg["confidence_threshold"] == 70
    dash.cfg["risk_per_trade"] = before
    dash.cfg["confidence_threshold"] = 62


# â”€â”€ H2: settings_enabled disabled every filter with one call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_disabling_settings_requires_explicit_confirmation(client, dash):
    """settings_enabled=false drops conf_thr to 50, raises max_pos to 999 and
    skips the ATR SL/TP clamps. It must not be a bare one-field POST."""
    before = dash.cfg["settings_enabled"]
    r = client.post("/api/config", json={"settings_enabled": False})
    assert r.status_code == 400, "risk-limit bypass accepted without confirmation"
    assert dash.cfg["settings_enabled"] == before


# â”€â”€ C1/H4 security posture on the auth boundary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_login_throttles_after_repeated_failures(anon_client, dash):
    """The password was the only defence, with no lockout, on a public host."""
    dash._login_failures.clear()
    codes = [
        anon_client.post("/login", data={"u": "x", "p": "wrong"}).status_code
        for _ in range(dash.LOGIN_MAX_ATTEMPTS + 3)
    ]
    assert 429 in codes, f"no throttle after {dash.LOGIN_MAX_ATTEMPTS} failures: {codes}"


def test_login_blocks_correct_password_once_throttled(anon_client, dash):
    dash._login_failures.clear()
    for _ in range(dash.LOGIN_MAX_ATTEMPTS):
        anon_client.post("/login", data={"u": "x", "p": "wrong"})
    r = anon_client.post("/login", data={"u": dash._AUTH_USER, "p": dash._AUTH_PASS})
    assert r.status_code == 429, "throttle did not block a valid password"


def test_successful_login_clears_throttle(anon_client, dash):
    dash._login_failures.clear()
    anon_client.post("/login", data={"u": "x", "p": "wrong"})
    r = anon_client.post("/login", data={"u": dash._AUTH_USER, "p": dash._AUTH_PASS})
    assert r.status_code == 302
    assert not dash._login_failures


def test_default_credentials_are_not_accepted(dash, monkeypatch):
    """The fallback was apex/admin, documented in .env.example."""
    monkeypatch.setattr(dash, "_AUTH_USER", "apex")
    monkeypatch.setattr(dash, "_AUTH_PASS", "admin")
    with dash.app.test_client() as c:
        dash._login_failures.clear()
        r = c.post("/login", data={"u": "apex", "p": "admin"})
    assert r.status_code != 302, "default apex/admin credentials still work"


def test_password_comparison_is_constant_time():
    """Plain == leaks the password byte-by-byte."""
    src = (Path(__file__).resolve().parent.parent / "apex_dashboard.py").read_text(
        encoding="utf-8", errors="replace"
    )
    login_body = src.split("def login():", 1)[1].split("\n@app.route", 1)[0]
    assert 'request.form.get("p", "") == _AUTH_PASS' not in login_body, (
        "login still uses == for the password"
    )
    assert "compare_digest" in login_body, "login does not use a constant-time compare"


def test_session_cookie_is_secure(dash):
    assert dash.app.config.get("SESSION_COOKIE_SECURE") is True


# â”€â”€ Hashed password support â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_hashed_password_verifies(dash):
    """APEX_PASS_HASH lets the plaintext password leave the environment."""
    import hashlib
    import secrets as pysecrets

    salt = pysecrets.token_bytes(16)
    digest = hashlib.scrypt(b"correct horse", salt=salt, n=2**14, r=8, p=1, dklen=32)
    encoded = f"{salt.hex()}${digest.hex()}"
    assert dash.verify_password("correct horse", encoded) is True


def test_hashed_password_rejects_wrong_value(dash):
    import hashlib
    import secrets as pysecrets

    salt = pysecrets.token_bytes(16)
    digest = hashlib.scrypt(b"correct horse", salt=salt, n=2**14, r=8, p=1, dklen=32)
    encoded = f"{salt.hex()}${digest.hex()}"
    assert dash.verify_password("wrong horse", encoded) is False


def test_hashed_password_never_raises_on_malformed_input(dash):
    for bad in ("", "not-a-hash", "$$$", "abc", "deadbeef$"):
        assert dash.verify_password("x", bad) is False


def test_hashed_password_is_used_when_configured(dash, monkeypatch, anon_client):
    """The hash must take precedence over APEX_PASS, and throttling must apply."""
    import hashlib
    import secrets as pysecrets

    salt = pysecrets.token_bytes(16)
    digest = hashlib.scrypt(b"hashed-pw", salt=salt, n=2**14, r=8, p=1, dklen=32)
    monkeypatch.setattr(dash, "_AUTH_PASS_HASH", f"{salt.hex()}${digest.hex()}")
    dash._login_failures.clear()

    assert anon_client.post("/login", data={"u": dash._AUTH_USER, "p": "wrong"}).status_code == 200
    r = anon_client.post("/login", data={"u": dash._AUTH_USER, "p": "hashed-pw"})
    assert r.status_code == 302, "hashed-password login was not accepted"


def test_security_headers_are_set(client, dash):
    r = client.get("/api/logs")
    assert r.status_code == 200
    hdrs = {k.lower() for k in r.headers.keys()}
    assert "x-content-type-options" in hdrs
    assert "x-frame-options" in hdrs
    assert "content-security-policy" in hdrs


def test_unhandled_errors_return_json_not_html(client, dash):
    """The SPA calls r.json() unconditionally, so an HTML 500 broke the UI."""
    r = client.get("/api/nonexistent")
    assert r.is_json, f"non-JSON error body: {r.data[:120]!r}"
    assert r.status_code == 404


def test_state_endpoint_survives_missing_markets(client, dash):
    """/api/state is polled every 30s. A KeyError on an unpopulated market
    used to blank the entire dashboard."""
    with dash._lock:
        dash._state.clear()
    r = client.get("/api/state")
    assert r.status_code == 200, r.data[:200]
    body = r.get_json()
    assert "india" in body and "us" in body
    assert body["india"]["portfolio_value"] >= 0


def test_request_bodies_are_bounded(dash):
    assert dash.app.config.get("MAX_CONTENT_LENGTH"), "unbounded request body size"
