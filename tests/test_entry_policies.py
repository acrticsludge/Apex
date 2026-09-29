"""Characterisation tests for the entry side of apply_cycle.

The exit side of apply_cycle was extracted into run_exit_policies and is covered
by test_exit_policies.py. The entry side is still one interlocking if/elif/else
decision spanning roughly a hundred lines, and it decided whether the system
opens real positions. It had almost no direct coverage: the only tests touching
apply_cycle passed an empty analysis list, so they exercised exits only.

These tests pin the entry decision as it behaves today. They are deliberately
not a refactor and assert nothing about how the code is structured — the value
is that any future change to the gates has to confront the current behaviour
rather than rediscover it.

The shape of the decision is: a symbol reaches a long entry, a short entry, or
is logged as a watch. A long is gated on confidence, a positive score, either
an RL action or a passing index, remaining position capacity, enough cash for
two shares, and the JEV action gate. A short is gated on shorting being
enabled, the RL agent being long-only, being outside the open window, bearish
confidence, a negative score, a bull-blocked index, capacity, cash, and the
same JEV action gate.

Note the index gate appears in both branches with opposite polarity: longs need
the index not strongly down, shorts need the index not strongly up. That
asymmetry is intentional and is pinned here.
"""
import copy

import pytest

import apex_dashboard as dash


def _mstate(cash=100_000.0, positions=None):
    return {
        "cash": cash, "positions": positions or {},
        "realised_pnl": 0.0, "wins": 0, "losses": 0,
        "peak_portfolio": 100_000.0, "max_drawdown": 0.0,
        "session_start_cash": 100_000.0, "trade_log": [],
        "wins_total": 0, "losses_total": 0, "cooldown_until": {},
    }


def _a(symbol="A.NS", score=40, confidence=75.0, price=100.0, **extra):
    """An analysis row with only the fields the entry side reads set."""
    a = {"symbol": symbol, "score": score, "confidence": confidence,
         "price": price, "adx": 30.0, "atr": 2.0, "rl_action": None}
    a.update(extra)
    return a


# The short branch gates on `bearish_conf = 100 - confidence`, so a short needs
# a *low* confidence together with a negative score. Using a high confidence on a
# bearish row blocks the short for the wrong reason and makes a test pass
# vacuously. These builders keep each row on the correct side of every gate it
# is not meant to be testing.
_LONG = {"score": 40, "confidence": 75.0}
_SHORT = {"score": -40, "confidence": 25.0}      # bearish_conf = 75


def _long(symbol="A.NS", **extra):
    return _a(symbol=symbol, **{**_LONG, **extra})


def _short(symbol="A.NS", **extra):
    return _a(symbol=symbol, **{**_SHORT, **extra})


@pytest.fixture
def entries(dash, monkeypatch):
    """A cycle in which only the entry side can act, with trades recorded.

    Exits are made inert by holding no positions, and the recorders capture
    which branch each symbol took so a test can assert the decision rather than
    the resulting position.
    """
    monkeypatch.setattr(dash, "_settings_active", lambda: True)
    monkeypatch.setattr(dash, "minutes_to_close", lambda k: None)      # not EOD
    monkeypatch.setattr(dash, "is_india_open", lambda: True)
    monkeypatch.setattr(dash, "is_us_open", lambda: True)
    monkeypatch.setattr(dash, "get_price", lambda s: None)
    monkeypatch.setattr(dash, "portfolio_value", lambda m, p: m.get("cash", 0.0))
    monkeypatch.setattr(dash, "unrealised_pnl", lambda m, p: 0.0)
    monkeypatch.setattr(dash, "rl_feedback_entry", lambda *a, **k: None)
    monkeypatch.setattr(dash, "_JEV_AVAILABLE", False)
    dash._signals["jev_decisions"] = {}
    dash._jev_gates.clear()
    yield dash
    dash._signals["jev_decisions"] = {}
    dash._jev_gates.clear()


def _cycle(dash, analyses, mstate, prices=None, market="india",
           max_pos=10, index_pct=0.0, cfg_over=None, decisions=None):
    """Run one cycle and return the state it produced."""
    prices = prices if prices is not None else {a["symbol"]: a["price"] for a in analyses}
    base = dict(dash.cfg)
    dash._index_trend[market] = index_pct
    try:
        if cfg_over:
            dash.cfg.update(cfg_over)
        with dash._lock:
            dash._state[market] = mstate
        payload = {market: decisions} if decisions is not None else {}
        with dash._lock:
            dash.apply_cycle(market, analyses, prices, max_pos, payload)
        return dash._state[market]
    finally:
        dash._index_trend[market] = 0.0
        dash.cfg.clear()
        dash.cfg.update(base)


def _recorded(dash, monkeypatch, ret_long=True, ret_short=True):
    """Record which entry branch fired, without executing a real fill."""
    calls = []
    monkeypatch.setattr(
        dash, "paper_buy",
        lambda s, q, p, m, atr=None, rcfg=None: (calls.append(("long", s)) or ret_long)
    )
    monkeypatch.setattr(
        dash, "paper_short",
        lambda s, q, p, m, atr=None, rcfg=None: (calls.append(("short", s)) or ret_short)
    )
    return calls


# ── The long branch ──────────────────────────────────────────────────────────

def test_a_strong_positive_signal_opens_a_long(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(score=40, confidence=75.0)], _mstate())
    assert calls == [("long", "A.NS")], f"expected one long, got {calls}"


def test_confidence_below_the_threshold_watches_instead_of_trading(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(score=40, confidence=40.0)], _mstate())
    assert calls == [], f"a low-confidence signal must not trade, got {calls}"


def test_a_negative_score_does_not_open_a_long(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_short=False)
    _cycle(entries, [_a(score=-40, confidence=75.0)], _mstate(),
           cfg_over={"short_selling_enabled": False})
    assert calls == [], f"a bearish score must not long, got {calls}"


def test_the_long_gate_moves_with_the_configured_threshold(entries, monkeypatch):
    """The gate reads cfg at call time; raising the bar must change the
    decision for the same signal."""
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(score=40, confidence=75.0)], _mstate(),
           cfg_over={"confidence_threshold": 90})
    assert calls == [], f"conf 75 should fail a 90 gate, got {calls}"

    calls2 = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(score=40, confidence=95.0)], _mstate(),
           cfg_over={"confidence_threshold": 90})
    assert calls2 == [("long", "A.NS")], f"conf 95 should pass a 90 gate, got {calls2}"


def test_cash_below_two_shares_blocks_the_entry(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(price=100.0)], _mstate(cash=150.0))
    assert calls == [], f"cash under 2x price must block, got {calls}"


def test_the_position_cap_blocks_further_entries(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    held = {"B.NS": {"qty": 1, "entry": 10.0, "side": "long",
                     "stop_loss": 1.0, "target": 20.0, "atr": 1.0}}
    _cycle(entries, [_a()], _mstate(positions=held), max_pos=1)
    assert calls == [], f"a full book must not open another, got {calls}"


# ── The index gate, and its asymmetry ────────────────────────────────────────

def test_a_falling_index_blocks_longs(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_short=False)
    _cycle(entries, [_a(score=40)], _mstate(), index_pct=-2.0,
           cfg_over={"short_selling_enabled": False, "index_min_pct": -0.5})
    assert calls == [], f"a down index must suppress longs, got {calls}"


def test_a_rising_index_does_not_block_longs(entries, monkeypatch):
    """The index gate is a floor, not a two-way switch: a bull day is fine for
    longs. Only shorts care about the upper bound."""
    calls = _recorded(entries, monkeypatch, ret_short=False)
    _cycle(entries, [_a(score=40)], _mstate(), index_pct=1.5,
           cfg_over={"short_selling_enabled": False, "index_min_pct": -0.5})
    assert calls == [("long", "A.NS")], f"a bull index must allow longs, got {calls}"


def test_a_strongly_bull_index_blocks_shorts(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short()], _mstate(), index_pct=1.5,
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [], f"a bull index must block shorts, got {calls}"


def test_the_index_gate_has_opposite_polarity_for_each_direction(entries, monkeypatch):
    """The asymmetry is the point: the same index level blocks shorts and
    permits longs."""
    bull_calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short()], _mstate(), index_pct=1.5,
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert bull_calls == []

    bear_calls = _recorded(entries, monkeypatch, ret_short=False)
    _cycle(entries, [_long()], _mstate(), index_pct=-2.0,
           cfg_over={"short_selling_enabled": True, "index_max_pct_for_short": 0.5,
                     "index_min_pct": -0.5})
    assert bear_calls == [], f"a down index must still block longs, got {bear_calls}"


# ── The short branch ─────────────────────────────────────────────────────────

def test_a_bearish_signal_opens_a_short_when_enabled(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short()], _mstate(),
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [("short", "A.NS")], f"expected one short, got {calls}"


def test_shorts_are_off_unless_explicitly_enabled(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short()], _mstate(),
           cfg_over={"short_selling_enabled": False})
    assert calls == [], f"shorting must be opt-in, got {calls}"


def test_a_bearish_signal_without_the_index_filter_also_opens_a_short(entries, monkeypatch):
    """A control for the test above, so the opt-in assertion is not passing
    because some unrelated gate is also blocking."""
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short()], _mstate(),
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [("short", "A.NS")], f"the control should short, got {calls}"


def test_bearish_confidence_below_the_short_threshold_watches(entries, monkeypatch):
    """confidence 85 gives bearish_conf 15, under a 50 threshold."""
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short(confidence=85.0)], _mstate(),
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [], f"weak bearish conviction must not short, got {calls}"


def test_a_bullish_score_does_not_open_a_short(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short(score=40)], _mstate(),
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [], f"a positive score must not short, got {calls}"


def test_the_rl_agent_is_long_only(entries, monkeypatch):
    """An RL action short-circuits the short branch entirely: the RL model
    does not produce short actions, so a row carrying one must not short."""
    calls = _recorded(entries, monkeypatch, ret_long=False)
    _cycle(entries, [_short(symbol="RL.NS", rl_action=2)], _mstate(),
           cfg_over={"short_selling_enabled": True, "short_confidence_threshold": 50,
                     "index_max_pct_for_short": 0.5})
    assert calls == [], f"the RL agent must not short, got {calls}"


# ── The RL long bypass ───────────────────────────────────────────────────────

def test_an_rl_action_satisfies_the_index_gate_for_longs(entries, monkeypatch):
    """A row carrying an RL action bypasses the index check on the long side.
    This is the documented exception, and it is easy to lose in a refactor."""
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_long(symbol="RL.NS", score=10, rl_action=1)],
           _mstate(), index_pct=-3.0,
           cfg_over={"index_min_pct": -0.5, "short_selling_enabled": False})
    assert calls == [("long", "RL.NS")], (
        f"an RL action should bypass the index gate on the long side, got {calls}"
    )


# ── The ADX trend gate ───────────────────────────────────────────────────────

def test_a_trendless_market_blocks_entries(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch, ret_short=False)
    _cycle(entries, [_a(score=40, adx=8.0)], _mstate(), cfg_over={"adx_min": 20})
    assert calls == [], f"ADX below the floor must block, got {calls}"


def test_the_adx_floor_is_configurable(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a(score=40, adx=8.0)], _mstate(), cfg_over={"adx_min": 5})
    assert calls == [("long", "A.NS")], f"a lower floor should admit ADX 8, got {calls}"


# ── The re-entry cooldown ────────────────────────────────────────────────────

def test_a_symbol_in_cooldown_is_not_re_entered(entries, monkeypatch):
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds")
    m = _mstate()
    m["cooldown_until"] = {"A.NS": future}
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a()], m)
    assert calls == [], f"cooldown must block re-entry, got {calls}"


def test_an_expired_cooldown_is_cleared_and_no_longer_blocks(entries, monkeypatch):
    from datetime import datetime, timedelta, timezone
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    m = _mstate()
    m["cooldown_until"] = {"A.NS": past}
    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_a()], m)
    assert calls == [("long", "A.NS")], f"an expired cooldown must not block, got {calls}"
    assert "A.NS" not in m["cooldown_until"], "the expired entry should be cleared"


# ── Held positions never take the entry branch ───────────────────────────────

def test_a_symbol_already_held_does_not_re_enter(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    held = {"A.NS": {"qty": 5, "entry": 100.0, "side": "long",
                     "stop_loss": 1.0, "target": 9999.0, "atr": 1.0}}
    _cycle(entries, [_a(score=40, confidence=95.0)], _mstate(positions=held))
    assert calls == [], f"a held symbol must not re-enter, got {calls}"


# ── Session-level kill switches block every entry ────────────────────────────

def test_the_daily_loss_limit_halts_new_entries(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    m = _mstate()
    m["realised_pnl"] = -20_000.0          # -20% on a 100k session
    _cycle(entries, [_a(score=40)], m, cfg_over={"daily_loss_limit_pct": 0.05})
    assert calls == [], f"a breached daily loss limit must halt entries, got {calls}"
    assert m.get("trading_halted") is True


def test_the_drawdown_kill_switch_halts_new_entries(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    m = _mstate()
    m["peak_portfolio"] = 200_000.0       # peak far above current cash
    _cycle(entries, [_a(score=40)], m, cfg_over={"max_drawdown_pct": 0.08})
    assert calls == [], f"a breached drawdown limit must halt entries, got {calls}"
    assert m.get("trading_halted") is True


def test_the_halt_flag_is_cleared_on_a_healthy_session(entries, monkeypatch):
    """trading_halted is set every cycle, so a prior halt must not persist."""
    calls = _recorded(entries, monkeypatch)
    m = _mstate()
    m["trading_halted"] = True
    _cycle(entries, [_a(score=40)], m)
    assert m.get("trading_halted") is False
    assert calls == [("long", "A.NS")], f"a healthy session should trade, got {calls}"


def test_eod_harvest_blocks_new_entries(entries, monkeypatch):
    calls = _recorded(entries, monkeypatch)
    monkeypatch.setattr(entries, "minutes_to_close", lambda k: 20)
    _cycle(entries, [_a(score=40)], _mstate())
    assert calls == [], f"EOD harvest must block new entries, got {calls}"


# ── The JEV halt gate, which sits above the entry decision ───────────────────

# The shape an actual gate publishes. should_halt is stubbed below, but the
# exit path still reads trend_strength from the same object, so a partial dict
# raises KeyError before the halt branch is ever reached.
_GATE = {
    "regime": {"choice": "normal", "probabilities": {}, "confidence": 0.9},
    "trend_strength": {"score": 0.0, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "news_bullishness": {"score": 0, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "portfolio_stress": {"score": 0, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "halt_new_buys": {"noul": 0.0, "confidence": 0.9},
    "position_action": {"choice": "hold", "probabilities": {}, "confidence": 0.9},
}


def _gate(noul, action="buy"):
    """A published gate. `action` matters independently of `noul`: when JEV is
    active and a gate is present, the long branch additionally requires
    position_action to be buy or add, so a "hold" action suppresses entries by
    a different route than the halt does."""
    d = copy.deepcopy(_GATE)
    d["halt_new_buys"] = {"noul": noul, "confidence": 0.9}
    d["position_action"] = {"choice": action, "probabilities": {}, "confidence": 0.9}
    return d


def test_a_jev_halt_blocks_every_new_entry(entries, monkeypatch):
    """should_halt is the only gate that returns before the entry loop runs, so
    it is the one place where a single decision suppresses all entries at once.
    """
    monkeypatch.setattr(entries, "_JEV_AVAILABLE", True)
    monkeypatch.setattr(entries._jev, "should_halt", lambda d: True)

    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_long(), _short(symbol="B.NS")], _mstate(),
           decisions=_gate(1.0), cfg_over={"jev_risk_enabled": True})

    assert calls == [], f"a JEV halt must suppress all entries, got {calls}"


def test_without_a_halt_entries_proceed_under_the_same_jev_settings(entries, monkeypatch):
    """The control for the test above: with should_halt returning False under
    identical config, entries must flow, so the halt assertion is not passing
    because some unrelated gate is also blocking."""
    monkeypatch.setattr(entries, "_JEV_AVAILABLE", True)
    monkeypatch.setattr(entries._jev, "should_halt", lambda d: False)

    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_long()], _mstate(),
           decisions=_gate(0.0), cfg_over={"jev_risk_enabled": True})

    assert calls == [("long", "A.NS")], f"the control should trade, got {calls}"


def test_a_hold_position_action_suppresses_entries_without_a_halt(entries, monkeypatch):
    """A second, independent way a JEV gate blocks entries.

    should_halt is False, so the cycle proceeds normally, but position_action is
    "hold", which the long branch rejects. This is separate from the halt gate
    and would be missed by a test that only checked should_halt.
    """
    monkeypatch.setattr(entries, "_JEV_AVAILABLE", True)
    monkeypatch.setattr(entries._jev, "should_halt", lambda d: False)

    calls = _recorded(entries, monkeypatch)
    _cycle(entries, [_long()], _mstate(),
           decisions=_gate(0.0, action="hold"), cfg_over={"jev_risk_enabled": True})

    assert calls == [], f"a hold action must suppress entries, got {calls}"
