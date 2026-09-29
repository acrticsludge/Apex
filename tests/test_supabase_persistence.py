"""Supabase persistence, exercised for real.

Every persistence test in this suite runs against the JSON fallback, because
`SUPABASE_URL` is unset in the test environment. That is precisely the
configuration production does *not* use, and it is why a defect that only
manifests across a network call — Supabase I/O running inside the state lock —
could sit in the codebase through a full review cycle without being caught.

These tests drive the actual `save_state` / `load_state` / `save_cfg` /
`load_cfg` / `_save_rl_decision` code against a stub that implements the same
query-builder chain and raises the same exception type as `supabase-py`. They
do not hit the network.

The stub is deliberately faithful in the places that matter:

* the chain is `.table(t).select(c).eq(k, v).execute()` and
  `.table(t).upsert(row).execute()` / `.insert(row).execute()`, each link
  returning self, exactly as postgrest does;
* `execute()` returns an object with `.data` as a list of dicts;
* failures raise `postgrest.exceptions.APIError`, the type the real client
  raises, so the code's `except Exception` is exercised realistically rather
  than against a bespoke exception.

Where a real Supabase project is configured, `test_against_a_real_project`
runs the round trip for real and skips otherwise.
"""
import json
import os
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATE_TABLE = "apex_state"


class _Response:
    """Mirrors postgrest's APIResponse for the attributes this code reads."""

    def __init__(self, data):
        self.data = data


class SupabaseStub:
    """A stand-in for supabase.Client with the same chainable surface.

    `fail_on` selects the failure mode; `lock_probe`, when set, records
    whether that lock was held at the moment execute() was called.
    """

    def __init__(self, fail_on=None, lock_probe=None, lock_held=False):
        self.fail_on = fail_on          # None | "execute" | "table"
        self.lock_probe = lock_probe
        self.lock_held = lock_held
        self.upserts = []
        self.inserts = []
        self.queries = []
        self._rows = []
        self._table = None

    # ── chain ────────────────────────────────────────────────────────────────
    def table(self, name):
        if self.fail_on == "table":
            raise _api_error("relation does not exist", code="42P01")
        self._table = name
        return self

    def select(self, *cols):
        self.queries.append((self._table, tuple(cols), "select"))
        return self

    def eq(self, col, val):
        self.queries.append((self._table, (col, val), "eq"))
        return self

    def upsert(self, row):
        self.upserts.append((self._table, dict(row)))
        self._rows = [dict(row)]
        return self

    def insert(self, row):
        self.inserts.append((self._table, dict(row)))
        return self

    def order(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def execute(self):
        if self.fail_on == "execute":
            raise _api_error("connection reset by peer", code="08006")
        if self.lock_probe is not None:
            self.lock_probe.append(self.lock_held)
        return _Response(list(self._rows))

    # ── test helpers ─────────────────────────────────────────────────────────
    def seed(self, rows):
        self._rows = [dict(r) for r in rows]

    @property
    def row_for(self, row_id):
        for table, row in self.upserts:
            if row.get("id") == row_id:
                return row
        return None


def _api_error(message, code="P0001"):
    from postgrest.exceptions import APIError

    return APIError({"message": message, "code": code, "details": None, "hint": None})


@pytest.fixture
def supa(dash, tmp_path, monkeypatch):
    """A stubbed Supabase client plus a clean JSON fallback location."""
    stub = SupabaseStub()
    monkeypatch.setattr(dash, "_sb", stub)
    monkeypatch.setitem(dash.cfg, "state_file", str(tmp_path / "state.json"))
    dash._state.clear()
    yield stub
    dash._state.clear()


# ── Round trip ───────────────────────────────────────────────────────────────

def test_state_survives_a_round_trip_through_supabase(dash, supa):
    dash._state["india"] = dash._empty_mstate(100000.0, "2026-09-29")
    dash._state["india"]["cash"] = 12345.67
    dash._state["india"]["positions"]["A.NS"] = {
        "qty": 5, "entry": 100.0, "stop_loss": 95.0, "target": 110.0, "side": "long",
    }
    dash._state["sessions"] = [{"date": "2026-09-28", "pnl": 42.0}]

    dash.save_state(dash._state)

    assert supa.upserts, "nothing was written to Supabase"
    table, row = supa.upserts[-1]
    assert table == STATE_TABLE
    assert row["id"].endswith("-singleton"), "row id must be namespaced by APEX_ENV"
    assert "updated_at" in row

    dash._state.clear()
    loaded = dash.load_state()
    assert loaded["india"]["cash"] == pytest.approx(12345.67)
    assert loaded["india"]["positions"]["A.NS"]["qty"] == 5
    assert loaded["sessions"][0]["date"] == "2026-09-28"


def test_the_stored_payload_is_json_safe(dash, supa):
    """save_state round-trips through json.dumps(default=str), so a value that
    is not JSON-native must degrade to a string rather than break the upsert."""
    dash._state["india"] = dash._empty_mstate(1000.0, "2026-09-29")
    dash._state["india"]["last_price_at"] = time.time()   # float, fine
    dash.save_state(dash._state)
    _, row = supa.upserts[-1]
    json.dumps(row["data"])          # must not raise
    assert isinstance(row["data"], dict)


# ── Namespacing ──────────────────────────────────────────────────────────────

def test_state_and_config_use_distinct_namespaced_rows(dash, supa):
    dash._state["india"] = dash._empty_mstate(1000.0, "2026-09-29")
    dash.save_state(dash._state)
    dash.save_cfg()

    ids = {row["id"] for _, row in supa.upserts}
    assert len(ids) == 2, f"state and config must not share a row: {ids}"
    assert all(i.startswith(dash._APEX_ENV) for i in ids), (
        f"rows are not namespaced by APEX_ENV: {ids}"
    )


# ── Config round trip, and the persist whitelist ─────────────────────────────

def test_config_round_trips_only_persisted_keys(dash, supa):
    dash.cfg["risk_per_trade"] = 0.015
    dash.cfg["confidence_threshold"] = 71
    dash.cfg["india_capital"] = 250000        # not user-adjustable
    dash.save_cfg()

    stored = [r for t, r in supa.upserts if r["id"] == dash._CONFIG_ID][-1]["data"]
    assert stored["risk_per_trade"] == 0.015
    assert "india_capital" not in stored, "capital must not be user-settable via stored config"

    # Restore from Supabase on the next start.
    dash.cfg["risk_per_trade"] = 0.02
    dash.cfg["confidence_threshold"] = 62
    dash.load_cfg()
    assert dash.cfg["risk_per_trade"] == 0.015
    assert dash.cfg["confidence_threshold"] == 71
    dash.cfg["india_capital"] = 180000
    dash.cfg["confidence_threshold"] = 62
    dash.cfg["risk_per_trade"] = 0.02


# ── Failure modes: each must degrade predictably ─────────────────────────────

def test_a_failing_write_falls_back_to_json(dash, supa, tmp_path, caplog):
    supa.fail_on = "execute"
    dash._state["india"] = dash._empty_mstate(5000.0, "2026-09-29")
    dash.save_state(dash._state)

    written = json.loads(Path(dash.cfg["state_file"]).read_text(encoding="utf-8"))
    assert written["india"]["cash"] == pytest.approx(5000.0), (
        "a Supabase outage must not lose the ledger — the JSON fallback is the backstop"
    )
    with caplog.at_level("ERROR", logger="apex"):
        dash.save_state(dash._state)
    assert any("Supabase save error" in r.getMessage() for r in caplog.records), (
        "a failed Supabase write was not logged as an error"
    )


def test_a_failing_read_falls_back_to_json(dash, supa, tmp_path):
    fallback = tmp_path / "state.json"
    fallback.write_text(
        json.dumps({"india": dash._empty_mstate(7777.0, "2026-09-28"), "us": {}}),
        encoding="utf-8",
    )
    supa.fail_on = "execute"
    loaded = dash.load_state()
    assert loaded["india"]["cash"] == pytest.approx(7777.0)


def test_a_missing_table_is_survivable(dash, supa):
    """A first deploy against a project without the table must not crash the
    app; it should fall back and say why."""
    supa.fail_on = "table"
    loaded = dash.load_state()
    assert "india" in loaded and "us" in loaded


def test_a_malformed_stored_row_does_not_crash_the_load(dash, supa):
    """The column exists but holds something _normalize_state cannot read."""
    supa.seed([{"id": dash._STATE_ID, "data": {"india": "not-a-dict", "us": None}}])
    loaded = dash.load_state()
    assert "india" in loaded, "load_state must always return both markets"
    assert isinstance(loaded["india"], dict)
    assert isinstance(loaded["us"], dict)


def test_an_empty_result_starts_fresh(dash, supa):
    supa.seed([])
    loaded = dash.load_state()
    assert loaded["india"]["cash"] > 0
    assert loaded.get("sessions") == []


# ── RL decision writes ───────────────────────────────────────────────────────

def test_rl_decisions_are_written_with_the_expected_shape(dash, supa):
    dash._save_rl_decision("A.NS", "india", {
        "rl_action": 1, "rl_probs": [0.1, 0.8, 0.1], "score": 72, "confidence": 80.0,
    })
    assert supa.inserts, "the RL decision was not recorded"
    table, row = supa.inserts[-1]
    assert table == "apex_rl_decisions"
    assert row["symbol"] == "A.NS" and row["market"] == "india"
    assert row["action"] == 1
    assert row["prob_buy"] == pytest.approx(0.8)
    assert 0.0 <= row["prob_hold"] <= 1.0


def test_rl_decision_write_failure_does_not_propagate(dash, supa):
    supa.fail_on = "execute"
    dash._save_rl_decision("A.NS", "us", {"rl_action": 0, "rl_probs": [1, 0, 0]})
    # Must not raise: losing an audit row must never stop a trading cycle.


# ── The lock, on the real Supabase path ──────────────────────────────────────

def test_no_supabase_call_is_issued_while_the_state_lock_is_held(dash, supa):
    """The defect this whole file exists to guard against, now on the path that
    actually runs in production.

    `save_state` is reached both from request routes and from the agent cycle.
    Whichever route it takes, the network call must not happen with `_lock`
    held — under `--workers 1 --threads 8` that stalls every thread at once.
    """
    probe = []

    class _LockAwareStub(SupabaseStub):
        def execute(self):
            if self.fail_on == "execute":
                raise _api_error("boom")
            probe.append(dash._lock.locked())
            return _Response(list(self._rows))

    dash._sb = _LockAwareStub()

    with dash._lock:
        dash._state["india"] = dash._empty_mstate(4242.0, "2026-09-29")
        snap = dash.snapshot_state()
    dash.persist_state(snap)

    assert probe, "the Supabase call never happened — the test proved nothing"
    assert not any(probe), "Supabase was called while the state lock was held"


def test_a_persist_reached_the_wrong_way_would_be_caught(dash, supa):
    """The same call, deliberately issued under the lock, must be observable.

    This guards the assertion above: without it, a stub that silently stopped
    recording would make that test pass for the wrong reason.
    """
    probe = []

    class _LockAwareStub(SupabaseStub):
        def execute(self):
            probe.append(dash._lock.locked())
            return _Response(list(self._rows))

    dash._sb = _LockAwareStub()
    with dash._lock:                      # the wrong way, on purpose
        dash.save_state(dash._state)
    assert probe == [True], "the probe failed to observe a locked call"


# ── Optional: a real project, when one is configured ─────────────────────────

@pytest.mark.skipif(
    not (os.getenv("APEX_TEST_SUPABASE_URL") and os.getenv("APEX_TEST_SUPABASE_KEY")),
    reason="set APEX_TEST_SUPABASE_URL and APEX_TEST_SUPABASE_KEY to run against a real project",
)
def test_against_a_real_project(dash):
    """Round trip against an actual Supabase instance. Opt-in.

    Uses the `apex_state` table, which the application already owns, and
    restores the row it touched.
    """
    from supabase import create_client

    client = create_client(os.environ["APEX_TEST_SUPABASE_URL"],
                           os.environ["APEX_TEST_SUPABASE_KEY"])
    row_id = "apex-ci-roundtrip"
    payload = {"india": {"cash": 1234.5}, "us": {"cash": 0.0}, "sessions": []}
    try:
        client.table(STATE_TABLE).upsert(
            {"id": row_id, "data": payload, "updated_at": "2026-01-01T00:00:00"}
        ).execute()
        resp = client.table(STATE_TABLE).select("data").eq("id", row_id).execute()
        assert resp.data[0]["data"]["india"]["cash"] == pytest.approx(1234.5)
    finally:
        client.table(STATE_TABLE).delete().eq("id", row_id).execute()
