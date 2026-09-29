"""Characterisation tests for the exit policies in apply_cycle.

apply_cycle was 544 lines containing two copies of the EOD / trailing / SL-TP
exit logic (the JEV-halt early-return path and the normal path). The copies had
already diverged:

  * only the halt path applied get_trailing_dist_mult()
  * only the main path counted _dc["sell"]
  * only the main path logged the trail adjustment
  * the halt path ran trailing and SL/TP as separate passes

These tests pin the intended behaviour of each policy, and assert the two paths
agree — which is the property the duplication had lost.
"""
import copy

import pytest

import apex_dashboard as dash


def _pos(**over):
    """A position with SL/TP far enough out that it survives an unrelated pass,
    so a test can inspect the mutated stop rather than a closed book. Defaults
    are side-aware because a short's target sits below the price."""
    side = over.get("side", "long")
    p = {"qty": 10, "entry": 100.0, "atr": 2.0, "side": side}
    if side == "short":
        p.update({"stop_loss": 200.0, "target": 50.0})
    else:
        p.update({"stop_loss": 50.0, "target": 200.0})
    p.update(over)
    return p


def _mstate(positions, cash=0.0):
    return {
        "cash": cash, "positions": positions,
        "realised_pnl": 0.0, "wins": 0, "losses": 0,
        "peak_portfolio": 1000.0, "max_drawdown": 0.0,
        "session_start_cash": 1000.0, "trade_log": [],
        "wins_total": 0, "losses_total": 0, "cooldown_until": {},
    }


@pytest.fixture
def runtime(dash, monkeypatch):
    """Neutralise the cycle's external dependencies; record trade calls."""
    dash._jev_gates.clear()
    monkeypatch.setattr(dash, "_settings_active", lambda: True)
    monkeypatch.setattr(dash, "is_india_open", lambda: True)
    monkeypatch.setattr(dash, "is_us_open", lambda: True)
    monkeypatch.setattr(dash, "portfolio_value", lambda m, p: m.get("cash", 0.0))
    monkeypatch.setattr(dash, "unrealised_pnl", lambda m, p: 0.0)
    monkeypatch.setattr(dash, "get_price", lambda s: None)
    monkeypatch.setattr(dash, "minutes_to_close", lambda k: None)
    monkeypatch.setattr(dash, "fetch_cycle_data", lambda *a, **k: ([], {}))
    # No new entries: the analysis list is empty, so only exits run.
    dash._signals["jev_decisions"] = {}
    yield dash
    dash._jev_gates.clear()
    dash._signals["jev_decisions"] = {}


def _run(dash, market, positions, prices, mtc=None, decisions=None):
    """Run one apply_cycle and return the resulting positions."""
    with dash._lock:
        dash._state[market] = _mstate(positions)
        dash._signals["jev_decisions"] = {market: decisions} if decisions else {}
    with dash._lock:
        dash.apply_cycle(market, [], prices, 10, dash._signals.get("jev_decisions", {}))
    return dash._state[market]["positions"]


# ── EOD policy ───────────────────────────────────────────────────────────────

def test_eod_harvest_liquidates_only_winners(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "minutes_to_close", lambda k: 20)   # harvest window
    pos = _run(runtime, "india", {"WIN.NS": _pos(), "LOSE.NS": _pos()},
               {"WIN.NS": 130.0, "LOSE.NS": 95.0})
    assert "WIN.NS" not in pos, "profitable position should be harvested"
    assert "LOSE.NS" in pos, "losing position must be held through the harvest window"


def test_eod_force_exit_liquidates_everything(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "minutes_to_close", lambda k: 10)   # force-exit window
    pos = _run(runtime, "india",
               {"A.NS": _pos(entry=100.0, target=200.0), "B.NS": _pos(entry=100.0, target=50.0)},
               {"A.NS": 130.0, "B.NS": 95.0})
    assert pos == {}, "force exit must flatten the book regardless of P&L"


def test_eod_is_inert_when_market_is_closed(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(target=200.0)}, {"A.NS": 130.0}, mtc=None)
    assert "A.NS" in pos


# ── Trailing stop policy ─────────────────────────────────────────────────────

def test_long_trailing_stop_ratchets_up(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(entry=100.0, atr=2.0, stop_loss=50.0)},
               {"A.NS": 130.0})
    assert "A.NS" in pos, "the trailed long must survive its own pass"
    assert pos["A.NS"]["running_high"] == 130.0
    # activation 100 + 2*1.0 = 102; new SL = 130 - 2*1.5 = 127
    assert pos["A.NS"]["stop_loss"] == pytest.approx(127.0, abs=0.01)


def test_trailing_stop_does_not_activate_before_threshold(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(entry=100.0, atr=2.0, stop_loss=50.0)},
               {"A.NS": 101.0})   # below activation of 102
    assert pos["A.NS"]["stop_loss"] == 50.0
    assert pos["A.NS"].get("running_high", 101.0) == 101.0


def test_long_trailing_stop_never_loosens(runtime):
    """A short whose computed trail is looser than the existing SL keeps the SL."""
    pos = _run(runtime, "india", {"A.NS": _pos(side="short", entry=100.0, atr=2.0,
                                               stop_loss=73.0, running_low=70.0)},
               {"A.NS": 70.0})
    assert "A.NS" in pos
    assert pos["A.NS"]["stop_loss"] == 73.0, "trailing stop must only tighten"


def test_short_trailing_stop_ratchets_down(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(side="short", entry=100.0, atr=2.0,
                                               stop_loss=110.0)},
               {"A.NS": 70.0})
    assert "A.NS" in pos, "the trailed short must survive its own pass"
    assert pos["A.NS"]["running_low"] == 70.0
    # activation 100 - 2*1.0 = 98; new SL = 70 + 2*1.5 = 73
    assert pos["A.NS"]["stop_loss"] == pytest.approx(73.0, abs=0.01)


# ── SL / target policy ───────────────────────────────────────────────────────

def test_long_stop_loss_closes(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(stop_loss=90.0)}, {"A.NS": 85.0})
    assert "A.NS" not in pos


def test_long_target_closes(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(target=115.0)}, {"A.NS": 120.0})
    assert "A.NS" not in pos


def test_short_stop_loss_is_inverted(runtime):
    """A short loses when price rises: sl_hit must be price >= stop_loss."""
    pos = _run(runtime, "india", {"A.NS": _pos(side="short", stop_loss=110.0)},
               {"A.NS": 115.0})
    assert "A.NS" not in pos


def test_short_target_is_inverted(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(side="short", target=90.0)},
               {"A.NS": 80.0})
    assert "A.NS" not in pos


def test_short_rising_price_does_not_hit_target(runtime):
    pos = _run(runtime, "india", {"A.NS": _pos(side="short", target=90.0, stop_loss=200.0)},
               {"A.NS": 100.0})
    assert "A.NS" in pos


# ── The parity property the duplication had lost ─────────────────────────────

HALT_DECISIONS = {
    "regime": {"choice": "crisis", "probabilities": {"crisis": 0.9}, "confidence": 0.95},
    # trend 0 with high confidence widens the trail to 2.0x, i.e. a different
    # stop from the base 1.5x. score=3.0 would collapse to 1.0x and hide the bug.
    "trend_strength": {"score": 0.0, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "news_bullishness": {"score": 3, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "portfolio_stress": {"score": 4, "probabilities": {}, "confidence": 0.9, "legend": {}},
    "halt_new_buys": {"noul": 0.95, "confidence": 0.95},
    "position_action": {"choice": "exit", "probabilities": {}, "confidence": 0.95},
}


@pytest.fixture
def halting(runtime, monkeypatch):
    import apex_jev
    monkeypatch.setattr(runtime, "_JEV_AVAILABLE", True, raising=False)
    # The halt verdict is per-run; the decisions are supplied to both runs.
    runtime._should_halt = True
    monkeypatch.setattr(
        apex_jev, "should_halt", lambda d: getattr(runtime, "_should_halt", False)
    )
    return runtime


def _comparable(pos):
    return {s: {k: v for k, v in p.items() if k != "running_high"} for s, p in pos.items()}


def _both_paths(dash, market, book, prices, mtc, decisions):
    """Run the same inputs with the halt flag off, then on."""
    dash._should_halt = False
    normal = _run(dash, market, copy.deepcopy(book), dict(prices), mtc, decisions)
    dash._should_halt = True
    halted = _run(dash, market, copy.deepcopy(book), dict(prices), mtc, decisions)
    dash._should_halt = False
    return _comparable(normal), _comparable(halted)


def test_halt_path_matches_normal_path_for_trailing_stop(halting, monkeypatch):
    """The halt flag must change whether new entries happen, not how an open
    position is managed. This is what the duplicated blocks got wrong."""
    monkeypatch.setattr(halting, "minutes_to_close", lambda k: None)
    book = {"A.NS": _pos(entry=100.0, atr=2.0, stop_loss=50.0)}
    prices = {"A.NS": 130.0}
    normal, halted = _both_paths(halting, "india", book, prices, None, HALT_DECISIONS)
    assert normal, "test book was fully closed; it must survive to compare stops"
    assert halted == normal, (
        f"trailing stop diverged between paths:\n  normal={normal}\n  halted={halted}"
    )


def test_halt_path_matches_normal_path_for_eod(halting, monkeypatch):
    monkeypatch.setattr(halting, "minutes_to_close", lambda k: 10)   # force-exit window
    book = {"A.NS": _pos(), "B.NS": _pos(side="short")}
    prices = {"A.NS": 130.0, "B.NS": 130.0}
    normal, halted = _both_paths(halting, "india", book, prices, 10, HALT_DECISIONS)
    assert halted == normal, f"EOD diverged: normal={normal} halted={halted}"


def test_halt_path_matches_normal_path_for_sl_tp(halting, monkeypatch):
    monkeypatch.setattr(halting, "minutes_to_close", lambda k: None)
    book = {"A.NS": _pos(stop_loss=90.0), "B.NS": _pos(side="short", stop_loss=110.0)}
    prices = {"A.NS": 85.0, "B.NS": 120.0}
    normal, halted = _both_paths(halting, "india", book, prices, None, HALT_DECISIONS)
    assert halted == normal, f"SL/TP diverged: normal={normal} halted={halted}"
    assert normal == {}, "both SL positions should have been closed"


def test_jev_trend_widens_the_trailing_stop(halting, monkeypatch):
    """The JEV trend adjustment must apply on the normal path too — previously
    only the halt path consulted it, so the feature was effectively dead."""
    monkeypatch.setattr(halting, "minutes_to_close", lambda k: None)
    book = {"A.NS": _pos(entry=100.0, atr=2.0, stop_loss=50.0)}
    prices = {"A.NS": 130.0}
    halting._should_halt = False

    with_jev = _run(halting, "india", copy.deepcopy(book), prices, None, HALT_DECISIONS)
    no_jev = _run(halting, "india", copy.deepcopy(book), prices, None, None)
    halting._should_halt = False

    # trend 0 -> 2.0x distance (SL 124) instead of the base 1.5x (SL 127)
    assert with_jev["A.NS"]["stop_loss"] == pytest.approx(124.0, abs=0.01)
    assert no_jev["A.NS"]["stop_loss"] == pytest.approx(127.0, abs=0.01)
    assert with_jev["A.NS"]["stop_loss"] < no_jev["A.NS"]["stop_loss"], (
        "a low-trend JEV read must produce a looser (lower) trailing stop"
    )
