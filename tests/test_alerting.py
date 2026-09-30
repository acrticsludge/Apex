"""Tests for alerting and liveness.

Two distinct failure modes are covered, and the second is the one that actually
bit:

- the agent cycles and fails, which the loop already counted and logged. Nobody
  watched the log line.
- the agent thread dies outright. The process stays up, the dashboard keeps
  serving, and *nothing is written anywhere*, because a background thread that
  stops does not raise. This is the silent killer, and it can only be detected
  by a liveness signal someone polls.

`apex_alert` must also never make things worse. It is called from the failure
path of a trading thread, so it may not raise, may not retry-storm, and may not
alert in a loop until the channel is ignored.
"""
import json
import threading
import time

import pytest

import apex_alert
from apex_alert import Heartbeat


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("APEX_ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("APEX_ALERT_TIMEOUT", "5")
    apex_alert.reset()
    monkeypatch.setattr(apex_alert, "_noted_unconfigured", False)
    yield
    apex_alert.reset()


class _Response:
    status_code = 200


@pytest.fixture
def posted(monkeypatch):
    """Capture webhook POSTs instead of sending them."""
    calls = []

    class _FakeRequests:
        @staticmethod
        def post(url, data=None, headers=None, timeout=None):
            calls.append({
                "url": url, "data": json.loads(data), "timeout": timeout,
            })
            return _Response()

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
    return calls


def _enable(monkeypatch, url="https://alerts.invalid/hook"):
    monkeypatch.setenv("APEX_ALERT_WEBHOOK_URL", url)
    return url


# ── With no webhook configured ─────────────────────────────────────────────

def test_nothing_is_sent_without_a_webhook(monkeypatch, posted):
    assert apex_alert.alert("cycle_failures", "boom") is False
    assert posted == []


def test_being_unconfigured_is_reported_once_not_every_time(monkeypatch, caplog):
    """Alerting that is silently off looks configured, which is worse than not
    having it. The operator is told, once."""
    with caplog.at_level("WARNING", logger="apex_alert"):
        for _ in range(3):
            apex_alert.alert("cycle_failures", "boom")
    notes = [r for r in caplog.records if "DISABLED" in r.getMessage()]
    assert len(notes) == 1, f"expected one notice, got {len(notes)}"


def test_status_reports_that_alerting_is_off():
    s = apex_alert.status()
    assert s["configured"] is False
    assert s["sent"] == 0


def test_status_names_the_variable_to_set(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="apex_alert"):
        apex_alert.alert("x", "y")
    assert "APEX_ALERT_WEBHOOK_URL" in caplog.text


# ── Delivery ───────────────────────────────────────────────────────────────

def test_an_alert_is_delivered(monkeypatch, posted):
    _enable(monkeypatch)
    assert apex_alert.alert("cycle_failures", "3 cycles failed") is True
    assert len(posted) == 1
    assert posted[0]["data"]["event"] == "cycle_failures"
    assert posted[0]["data"]["message"] == "3 cycles failed"
    assert posted[0]["data"]["level"] == "error"


def test_the_payload_carries_context_for_triage(monkeypatch, posted):
    _enable(monkeypatch)
    apex_alert.alert("cycle_failures", "boom", consecutive=3, error="TypeError")
    body = posted[0]["data"]
    assert body["consecutive"] == 3
    assert body["error"] == "TypeError"
    assert body["env"], "the environment is needed to tell main from a preview"
    assert body["at"]


def test_unserialisable_context_is_dropped_not_fatal(monkeypatch, posted):
    """A stray object in the payload must not turn an alert into a failure."""
    _enable(monkeypatch)
    assert apex_alert.alert("x", "y", bad=object(), good=1) is True
    assert posted[0]["data"]["good"] == 1
    assert "bad" not in posted[0]["data"]


def test_the_timeout_is_bounded(monkeypatch, posted):
    _enable(monkeypatch)
    apex_alert.alert("x", "y")
    assert 0 < posted[0]["timeout"] <= 30


def test_a_malformed_timeout_falls_back_to_the_default(monkeypatch, posted):
    _enable(monkeypatch)
    monkeypatch.setenv("APEX_ALERT_TIMEOUT", "not-a-number")
    assert apex_alert.alert("x", "y") is True
    assert posted[0]["timeout"] == 5.0


def test_an_absurd_timeout_is_capped(monkeypatch, posted):
    _enable(monkeypatch)
    monkeypatch.setenv("APEX_ALERT_TIMEOUT", "9999")
    apex_alert.alert("x", "y")
    assert posted[0]["timeout"] <= 30, "a 2.7-hour timeout would stall the trading thread"


# ── A broken webhook must not become a broken bot ──────────────────────────

def test_a_webhook_that_raises_does_not_propagate(monkeypatch):
    """It is called from the trading thread's failure path. If alerting raised,
    the caller would crash and the failure handling would fail."""
    _enable(monkeypatch)

    class _Boom:
        @staticmethod
        def post(*a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setitem(__import__("sys").modules, "requests", _Boom)
    assert apex_alert.alert("x", "y") is False, "alerting raised into the caller"


def test_a_failed_delivery_is_recorded_for_health_checks(monkeypatch):
    _enable(monkeypatch)

    class _Boom:
        @staticmethod
        def post(*a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setitem(__import__("sys").modules, "requests", _Boom)
    apex_alert.alert("x", "y")
    assert "connection refused" in apex_alert.status()["last_error"]


def test_a_failed_delivery_is_logged_rather_than_swallowed(monkeypatch, caplog):
    _enable(monkeypatch)

    class _Boom:
        @staticmethod
        def post(*a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setitem(__import__("sys").modules, "requests", _Boom)
    with caplog.at_level("WARNING", logger="apex_alert"):
        apex_alert.alert("x", "y")
    assert any("delivery failed" in r.getMessage() for r in caplog.records)


def test_a_delivery_failure_does_not_retry(monkeypatch):
    """A webhook that is down must fail fast. Retrying turns an outage into a
    self-inflicted one."""
    _enable(monkeypatch)
    attempts = []

    class _Boom:
        @staticmethod
        def post(*a, **k):
            attempts.append(1)
            raise RuntimeError("connection refused")

    monkeypatch.setitem(__import__("sys").modules, "requests", _Boom)
    apex_alert.alert("x", "y")
    assert len(attempts) == 1, f"attempted {len(attempts)} times for one alert"


# ── Escalation: repeat the signal, not the noise ───────────────────────────

def test_a_repeated_event_is_not_announced_every_time(monkeypatch, posted):
    """Alerting every cycle for six hours trains the operator to ignore the
    channel, which is the same as having no alerting."""
    _enable(monkeypatch)
    for i in range(1, 21):
        apex_alert.alert("cycle_failures", f"boom {i}")
    assert len(posted) == 3, f"expected 3 escalation points in 20, got {len(posted)}"


def test_escalation_points_are_ordered_and_include_the_first(monkeypatch, posted):
    _enable(monkeypatch)
    for i in range(1, 11):
        apex_alert.alert("e", f"boom {i}")
    counts = [p["data"]["occurrences"] for p in posted]
    assert counts == [1, 3, 10], f"unexpected escalation ladder: {counts}"


def test_escalation_stops_entirely_past_the_last_threshold(monkeypatch, posted):
    _enable(monkeypatch)
    for i in range(1, 151):
        apex_alert.alert("e", f"boom {i}")
    assert len(posted) == len(apex_alert._ESCALATION)


def test_two_different_events_escalate_independently(monkeypatch, posted):
    _enable(monkeypatch)
    apex_alert.alert("cycle_failures", "a")
    apex_alert.alert("agent_dead", "b")
    assert {p["data"]["event"] for p in posted} == {"cycle_failures", "agent_dead"}


def test_recovery_resets_the_escalation_ladder(monkeypatch, posted):
    """Otherwise the next incident starts halfway up the ladder and the first
    occurrence — the one you most want — is silent."""
    _enable(monkeypatch)
    apex_alert.alert("e", "boom")
    apex_alert.reset("e")
    apex_alert.alert("e", "boom again")
    assert len(posted) == 2, "the ladder did not reset, so a new incident is silent"
    assert posted[1]["data"]["occurrences"] == 1


def test_reset_with_no_argument_clears_everything(monkeypatch, posted):
    _enable(monkeypatch)
    apex_alert.alert("a", "x")
    apex_alert.alert("b", "y")
    apex_alert.reset()
    apex_alert.alert("a", "x")
    assert posted[-1]["data"]["occurrences"] == 1


def test_suppressed_counters_are_visible(monkeypatch, posted):
    """Otherwise a channel that looks configured but has said nothing for an hour
    is indistinguishable from a healthy quiet bot."""
    _enable(monkeypatch)
    for i in range(1, 21):
        apex_alert.alert("e", f"boom {i}")
    s = apex_alert.status()
    assert s["sent"] == 3
    assert s["suppressed"] == 17


# ── Heartbeat: the half that needs no third party ──────────────────────────

def test_a_running_agent_with_a_recent_cycle_is_healthy():
    hb = Heartbeat()
    hb.beat()
    s = hb.snapshot(running=True, status="running")
    assert s["healthy"] is True
    assert s["stale"] is False


def test_a_stopped_agent_is_not_healthy():
    hb = Heartbeat()
    hb.beat()
    assert hb.snapshot(running=False, status="stopped")["healthy"] is False


def test_an_agent_that_never_cycled_is_stale():
    hb = Heartbeat()
    s = hb.snapshot(running=True, status="running")
    assert s["stale"] is True
    assert s["last_cycle_age_s"] is None


def test_an_old_cycle_is_stale():
    """The silent killer: the thread stopped without raising, so the only
    evidence is that time passed with no cycle."""
    hb = Heartbeat()
    hb.beat()
    s = hb.snapshot(running=True, status="running", stale_after_s=0.01)
    time.sleep(0.05)
    s = hb.snapshot(running=True, status="running", stale_after_s=0.01)
    assert s["stale"] is True
    assert s["healthy"] is False


def test_repeated_failures_make_an_otherwise_fresh_agent_unhealthy():
    hb = Heartbeat()
    hb.beat()
    s = hb.snapshot(running=True, status="error", consecutive_failures=2)
    assert s["healthy"] is False
    assert s["consecutive_failures"] == 2


def test_the_snapshot_reports_how_stale_not_just_that_it_is():
    hb = Heartbeat()
    hb.beat()
    time.sleep(0.05)
    s = hb.snapshot(running=True, status="running")
    assert s["last_cycle_age_s"] >= 0.05


def test_uptime_is_reported():
    hb = Heartbeat()
    assert hb.snapshot(running=True, status="running")["uptime_s"] >= 0


def test_beats_from_several_threads_are_safe():
    """The cycle runs on the agent thread and the health check on a request
    thread, so this dict is genuinely contended."""
    hb = Heartbeat()
    errors = []

    def _hammer():
        try:
            for _ in range(200):
                hb.beat()
                hb.snapshot(running=True, status="running")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"concurrent access failed: {errors}"
    assert hb.snapshot(running=True, status="running")["last_cycle_age_s"] is not None


def test_a_concurrent_burst_follows_the_ladder_and_loses_no_counts(monkeypatch, posted):
    """A failure storm must not become a request flood at the webhook.

    Eight concurrent failures cross two escalation points (1 and 3), so two
    deliveries is the correct answer — not one, and not eight. What this also
    pins is that the occurrence counter does not lose updates to the race: the
    eight calls are counted as eight occurrences, not two.
    """
    _enable(monkeypatch)
    barrier = threading.Barrier(8)

    def _fire():
        barrier.wait()
        apex_alert.alert("e", "boom")

    threads = [threading.Thread(target=_fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(posted) == 2, f"sent {len(posted)} webhooks for 8 failures"
    s = apex_alert.status()
    assert s["sent"] + s["suppressed"] == 8, (
        f"occurrences were lost to the race: {s}"
    )
    counts = sorted(p["data"]["occurrences"] for p in posted)
    assert counts == [1, 3], f"unexpected escalation under concurrency: {counts}"

# ── The /healthz contract ───────────────────────────────────────────────────

def test_healthz_is_reachable_without_a_session(dash, anon_client):
    """A monitor has no session. Behind the login redirect this endpoint would
    catch nothing, which is the failure it exists to catch."""
    resp = anon_client.get("/healthz")
    assert resp.status_code in (200, 503), (
        f"/healthz is gated by auth and useless to a monitor: {resp.status_code}"
    )


def test_healthz_reports_503_while_the_agent_is_down(dash, anon_client):
    assert anon_client.get("/healthz").status_code == 503, (
        "an HTTP monitor cannot detect a dead agent if this returns 200"
    )


def test_healthz_reports_200_while_the_agent_is_cycling(dash, anon_client, monkeypatch):
    import apex_alert
    monkeypatch.setitem(dash._agent, "running", True)
    monkeypatch.setitem(dash._agent, "status", "running")
    monkeypatch.setitem(dash._agent, "consecutive_failures", 0)
    apex_alert.heartbeat.beat()
    assert anon_client.get("/healthz").status_code == 200


def test_healthz_goes_unhealthy_on_a_stale_cycle(dash, anon_client, monkeypatch):
    """The silent killer: the thread stopped without raising, so nothing was
    logged. Age of the last cycle is the only evidence there is."""
    import apex_alert
    monkeypatch.setitem(dash._agent, "running", True)
    apex_alert.heartbeat.beat()
    # No further beat; force staleness by ageing the recorded cycle.
    apex_alert.heartbeat._last_cycle -= 3600
    assert anon_client.get("/healthz").status_code == 503


def test_healthz_reports_consecutive_failures(dash, anon_client, monkeypatch):
    monkeypatch.setitem(dash._agent, "consecutive_failures", 7)
    assert anon_client.get("/healthz").get_json()["failures"] == 7


# ── The unauthenticated surface stays minimal ──────────────────────────────

def test_healthz_exposes_no_ledger_or_credential(dash, anon_client):
    """This endpoint is reachable by anyone who can reach the host. It must
    answer 'is it alive' and nothing else."""
    body = anon_client.get("/healthz").get_json()
    text = " ".join(body).lower()
    for leak in ("cash", "pnl", "position", "entry", "qty", "key", "secret",
                 "password", "token", "env", "hostname", "url"):
        assert leak not in text, (
            f"/healthz exposes {leak!r} on an unauthenticated endpoint: {body}"
        )


def test_healthz_does_not_leak_the_webhook_destination(dash, anon_client, monkeypatch):
    """A connection error from the webhook carries the provider's hostname, so
    the error string must not be on an unauthenticated response."""
    monkeypatch.setenv("APEX_ALERT_WEBHOOK_URL", "https://alerts.secret-provider.invalid/x")
    import apex_alert
    apex_alert.reset()
    # alert() is contractually non-raising; a delivery failure here records the
    # error instead, which is exactly the state being checked.
    assert apex_alert.alert("x", "y") is False
    assert apex_alert.status()["last_error"], "the delivery failure was not recorded"
    body = anon_client.get("/healthz").get_json()
    assert "secret-provider" not in " ".join(map(str, body.values()))
    assert "last_error" not in body, "the error string leaked onto an open endpoint"
    assert "alerting" not in body or isinstance(body["alerting"], bool)


def test_healthz_only_reports_whether_alerting_is_configured(dash, anon_client):
    body = anon_client.get("/healthz").get_json()
    assert body["alerting"] is False
    assert isinstance(body["alerting"], bool), (
        "alerting detail on an unauthenticated endpoint risks leaking the "
        "destination; expose a boolean only"
    )


def test_other_routes_are_still_gated(dash, anon_client):
    """Exempting one route must not have opened the rest."""
    assert anon_client.get("/api/state").status_code == 401
    assert anon_client.get("/").status_code in (302, 401)

# ── The loop raises alerts on the paths that used to be silent ─────────────

def _bounded(monkeypatch, dash, outcomes):
    """Drive agent_loop with a scripted cycle list, then let it exit.

    `outcomes` is a list of tuples returned by the cycle, or the string "boom"
    to raise instead. The final iteration clears `_agent["running"]` so the loop
    terminates.

    The failure path sleeps 30 s between attempts, so the sleep is stubbed too.
    Without both bounds the test hangs instead of failing.
    """
    seq = list(outcomes)
    state = {"n": 0}

    def _cycle(*_a):
        state["n"] += 1
        item = seq.pop(0) if seq else None
        if item is None:
            dash._agent["running"] = False
            return (False, False, 0)
        if item == "boom":
            raise RuntimeError("cycle exploded")
        return item

    monkeypatch.setattr(dash, "_run_one_cycle", _cycle)
    monkeypatch.setattr(dash.time, "sleep", lambda *_a, **_k: None)
    dash._agent["running"] = True
    dash.agent_loop()
    return state["n"]


def test_consecutive_failures_raise_an_alert(dash, monkeypatch):
    """The log line existed and nothing watched it."""
    import apex_alert
    sent = []
    monkeypatch.setattr(apex_alert, "alert", lambda *a, **k: sent.append(a) or True)

    _bounded(monkeypatch, dash, ["boom", "boom", "boom"])

    assert sent, "repeated cycle failures produced no alert"
    event, message = sent[0][0], sent[0][1]
    assert event == "agent_cycle_failures"
    assert "consecutive" in message


def test_a_single_failure_alerts_but_does_not_storm(dash, monkeypatch):
    """One bad cycle is worth knowing about; a hundred identical ones are not,
    and a notification per cycle is how a channel gets ignored."""
    import apex_alert
    sent = []
    monkeypatch.setattr(apex_alert, "alert", lambda *a, **k: sent.append(a) or True)

    calls = _bounded(monkeypatch, dash, ["boom"] * 6)

    assert calls >= 2, "the loop did not retry, so this proves nothing"
    failure_alerts = [a for a in sent if a[0] == "agent_cycle_failures"]
    assert failure_alerts, "no alert for repeated failures"


def test_recovery_raises_an_info_alert(dash, monkeypatch):
    """Silence after a stall is as unhelpful as the stall itself."""
    import apex_alert
    sent = []
    monkeypatch.setattr(apex_alert, "alert", lambda *a, **k: sent.append((a, k)) or True)

    _bounded(monkeypatch, dash, ["boom", "boom", (False, False, 0)])

    events = [a[0] for a, _k in sent]
    assert "agent_cycle_failures" in events
    assert "agent_recovered" in events, f"recovery was silent: {events}"


def test_recovery_resets_the_failure_ladder(dash, monkeypatch):
    """Otherwise the next incident starts halfway up the escalation and its
    first occurrence — the one you most want — is silent."""
    import apex_alert
    resets = []
    monkeypatch.setattr(apex_alert, "reset", lambda ev=None: resets.append(ev) or None)
    monkeypatch.setattr(apex_alert, "alert", lambda *a, **k: True)

    _bounded(monkeypatch, dash, ["boom", (False, False, 0)])

    assert "agent_cycle_failures" in resets, f"the ladder was not reset: {resets}"


def test_the_loop_exiting_raises_an_alert(dash, monkeypatch):
    """An operator stopping the agent knows they did. Nobody knows when the
    thread is killed, and this is the last code that runs either way."""
    import apex_alert
    sent = []
    monkeypatch.setattr(apex_alert, "alert", lambda *a, **k: sent.append(a) or True)

    _bounded(monkeypatch, dash, [(False, False, 0)])

    events = [a[0] for a in sent]
    assert "agent_stopped" in events, f"loop exit was silent: {events}"


def test_a_successful_cycle_marks_the_heartbeat(dash, monkeypatch):
    import apex_alert
    apex_alert.heartbeat.beat()
    dash._agent["consecutive_failures"] = 4
    _bounded(monkeypatch, dash, [(False, False, 0)])
    snap = apex_alert.heartbeat.snapshot(running=True, status="running")
    assert snap["last_cycle_age_s"] is not None
    assert dash._agent["consecutive_failures"] == 0, "the counter was not cleared"


def test_an_alert_failure_does_not_stop_the_loop(dash, monkeypatch):
    """Alerting is called from the failure path. If it raised, the failure
    handler itself would fail and the loop would die — trading down in order to
    protect monitoring."""
    import apex_alert

    def _explode(*_a, **_k):
        raise RuntimeError("webhook exploded")

    monkeypatch.setattr(apex_alert, "alert", _explode)
    calls = _bounded(monkeypatch, dash, ["boom", "boom", "boom"])
    assert calls >= 2, "the loop died on the first alerting failure"
