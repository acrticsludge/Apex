"""Tests that every cost a trade pays reaches reported P&L.

Found while adding slippage: entry commission was deducted from cash when a
position opened, but `realised_pnl` was computed from the exit side alone —
`net_proceeds - entry * qty`. The entry cost never reached P&L.

That is not a cosmetic accounting preference. `daily_loss_pct` is
`realised_pnl / session_start_cash`, and the drawdown kill-switch reads the same
ledger, so a cost that is missing from P&L makes **both** of them fire later than
they should. The safety mechanism is under-sensitive in exactly the direction
that loses money. Slippage does not have this problem because it is baked into
the fill price; this is the cost that has to be carried explicitly.

Positions persisted before this change have no `entry_commission` key, so the
lookup is `.get(..., 0.0)` and an old ledger opens without error.
"""
import pytest

import apex_dashboard as dash


def _mstate(cash=1_000_000.0, positions=None):
    return {
        "cash": cash, "positions": positions or {},
        "realised_pnl": 0.0, "wins": 0, "losses": 0,
        "peak_portfolio": cash, "max_drawdown": 0.0,
        "session_start_cash": cash, "trade_log": [],
        "wins_total": 0, "losses_total": 0, "cooldown_until": {},
    }


@pytest.fixture
def costs(dash, monkeypatch):
    """Commission on, slippage off, so this tests commission accounting alone."""
    base = dict(dash.cfg)
    dash.cfg["commission_pct"] = 0.001
    dash.cfg["slippage_pct"] = 0.0
    dash.cfg["settings_enabled"] = True
    dash.cfg["use_atr_exits"] = True
    monkeypatch.setattr(dash, "rl_feedback_entry", lambda *a, **k: None)
    monkeypatch.setattr(dash, "rl_feedback_exit", lambda *a, **k: None)
    yield dash
    dash.cfg.clear()
    dash.cfg.update(base)


def test_the_entry_commission_is_recorded_on_the_position(costs):
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    pos = m["positions"]["A.NS"]
    assert "entry_commission" in pos, (
        "the entry cost is not stored, so it cannot be charged at exit"
    )
    assert pos["entry_commission"] == pytest.approx(0.001 * pos["entry"] * pos["qty"])


def test_the_short_entry_commission_is_recorded_too(costs):
    m = _mstate()
    costs.execute_short("A.NS", 100.0, m, atr=2.0)
    pos = m["positions"]["A.NS"]
    assert pos["entry_commission"] == pytest.approx(0.001 * pos["entry"] * pos["qty"])


def test_a_long_flat_round_trip_loses_both_sides_of_commission(costs):
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    qty = m["positions"]["A.NS"]["qty"]
    entry_commission = m["positions"]["A.NS"]["entry_commission"]
    costs.execute_sell("A.NS", 100.0, "TEST", m)

    exit_commission = 100.0 * qty * 0.001
    assert m["realised_pnl"] == pytest.approx(-(entry_commission + exit_commission)), (
        f"a flat round trip reported {m['realised_pnl']}, but commission was "
        f"{entry_commission} on entry and {exit_commission} on exit"
    )


def test_a_short_flat_round_trip_loses_both_sides_of_commission(costs):
    m = _mstate()
    costs.execute_short("A.NS", 100.0, m, atr=2.0)
    qty = m["positions"]["A.NS"]["qty"]
    entry_commission = m["positions"]["A.NS"]["entry_commission"]
    costs.execute_cover("A.NS", 100.0, "TEST", m)

    exit_commission = 100.0 * qty * 0.001
    assert m["realised_pnl"] == pytest.approx(-(entry_commission + exit_commission))


def test_commission_is_charged_once_not_twice(costs):
    """The fix must not also start charging entry commission on the exit, which
    would double-count it."""
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    pos = dict(m["positions"]["A.NS"])
    exit_price = 100.0
    costs.execute_sell("A.NS", exit_price, "TEST", m)

    expected = -(pos["entry_commission"]
                 + 0.001 * exit_price * pos["qty"])
    assert m["realised_pnl"] == pytest.approx(expected)
    assert abs(m["realised_pnl"]) < abs(2 * expected), (
        "commission was charged more than twice per side"
    )


def test_the_daily_loss_limit_becomes_more_sensitive(costs):
    """The point of the fix. A cost missing from P&L delays the halt, because
    `daily_loss_pct = realised_pnl / session_start_cash` reads high while a cost
    is unaccounted for."""
    m = _mstate()
    exit_price = 100.001          # a 0.1% gross gain on a 100.00 entry
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    qty = m["positions"]["A.NS"]["qty"]
    entry_commission = m["positions"]["A.NS"]["entry_commission"]

    gross = qty * (exit_price - 100.0)
    exit_commission = 0.001 * exit_price * qty
    assert gross > 0, "the test needs a trade that looks profitable before costs"

    costs.execute_sell("A.NS", exit_price, "TEST", m)

    assert m["realised_pnl"] == pytest.approx(gross - entry_commission - exit_commission)
    assert m["realised_pnl"] < 0, (
        "the trade reported a profit despite paying commission on both sides — "
        "so the loss limit would not see it"
    )


def test_an_old_position_without_the_key_still_closes(costs):
    """State persisted before this change has no entry_commission. It must open
    and close rather than raising on a missing field."""
    m = _mstate(positions={"OLD.NS": {"side": "long", "qty": 10, "entry": 100.0,
                                        "stop_loss": 1.0, "target": 9999.0,
                                        "atr": 1.0}})
    costs.execute_sell("OLD.NS", 100.0, "TEST", m)
    assert "OLD.NS" not in m["positions"]
    assert m["realised_pnl"] == pytest.approx(-1.0)   # exit commission only


def test_an_old_short_without_the_key_still_closes(costs):
    m = _mstate(positions={"OLD.NS": {"side": "short", "qty": 10, "entry": 100.0,
                                        "stop_loss": 9999.0, "target": 1.0,
                                        "atr": 1.0}})
    costs.execute_cover("OLD.NS", 100.0, "TEST", m)
    assert "OLD.NS" not in m["positions"]
    assert m["realised_pnl"] == pytest.approx(-1.0)


def test_the_position_carries_the_entry_cost_it_paid(costs):
    """A position is closed whole on this path, so the whole entry cost is
    charged at exit. Recorded on the position rather than recomputed, because
    the price actually paid is what commission was charged on."""
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    pos = m["positions"]["A.NS"]
    assert pos["entry_commission"] > 0
    assert pos["entry_commission"] == pytest.approx(
        pos["entry"] * pos["qty"] * 0.001
    )


def test_the_win_loss_counters_see_the_commission(costs):
    """A trade that grossed positive and netted negative is a loss."""
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    qty = m["positions"]["A.NS"]["qty"]
    # Gross +0.1% exactly offsets nothing; commission is 0.2% of notional.
    costs.execute_sell("A.NS", 100.1, "TEST", m)
    assert qty > 0
    assert m["realised_pnl"] < 0
    assert m["losses"] == 1 and m["wins"] == 0


def test_zero_commission_leaves_pnl_untouched(costs):
    costs.cfg["commission_pct"] = 0.0
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    costs.execute_sell("A.NS", 100.0, "TEST", m)
    assert m["realised_pnl"] == pytest.approx(0.0)
    assert m["positions"] == {}


def test_a_profitable_trade_still_reports_profit(costs):
    """The fix must not turn winners into losers."""
    m = _mstate()
    costs.execute_buy("A.NS", 100.0, m, atr=2.0)
    costs.execute_sell("A.NS", 105.0, "TARGET", m)
    assert m["realised_pnl"] > 0
    assert m["wins"] == 1