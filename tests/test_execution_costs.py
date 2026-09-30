"""Tests for trade execution costs: commission and slippage.

The paper layer deducted `commission_pct` on every side but filled every order
at the quoted price. Real fills do not happen at the quote, so reported P&L was
optimistic by roughly the round-trip spread — and position sizing, the daily
loss limit and the drawdown kill-switch are all calibrated against that
optimistic number.

Slippage is applied by recording the *fill* price rather than the quote, so it
propagates consistently: entry, stop, target, running high/low and cash all
derive from what was actually paid or received. A long round trip therefore pays
slippage twice — once paying up on entry, once receiving less on exit — which is
what the market would charge.

These tests pin the direction of each side, because getting it backwards would
make slippage a profit rather than a cost, and every P&L number would look
better instead of worse.

The RL learner's feedback keeps using quoted prices. A constant percentage drag
applied to every trade largely cancels under normalised advantages, so it does
not change which signals look good, and mixing a quoted entry with a slipped exit
would bias the reward in one direction. The ledger has to be honest; the
learner's relative ranking does not.
"""
import pytest

import apex_dashboard as dash


BUY, SELL = "buy", "sell"


def _mstate(cash=1_000_000.0, positions=None):
    return {
        "cash": cash, "positions": positions or {},
        "realised_pnl": 0.0, "wins": 0, "losses": 0,
        "peak_portfolio": cash, "max_drawdown": 0.0,
        "session_start_cash": cash, "trade_log": [],
        "wins_total": 0, "losses_total": 0, "cooldown_until": {},
    }


def _long_round_trip(fills, entry_quote=100.0, exit_quote=None, atr=2.0):
    """Open and close a long, returning (pnl, qty, entry_fill, exit_fill).

    Derives everything from the position that was actually opened rather than
    assuming a share count. Position sizing depends on risk_per_trade and the
    position cap, and these tests are about cost arithmetic, not sizing.
    """
    exit_quote = entry_quote if exit_quote is None else exit_quote
    m = _mstate()
    fills.execute_buy("A.NS", entry_quote, m, atr=atr)
    pos = dict(m["positions"]["A.NS"])
    fills.execute_sell("A.NS", exit_quote, "TEST", m)
    return m["realised_pnl"], pos["qty"], pos["entry"]


def _short_round_trip(fills, entry_quote=100.0, exit_quote=None, atr=2.0):
    exit_quote = entry_quote if exit_quote is None else exit_quote
    m = _mstate()
    fills.execute_short("A.NS", entry_quote, m, atr=atr)
    pos = dict(m["positions"]["A.NS"])
    fills.execute_cover("A.NS", exit_quote, "TEST", m)
    return m["realised_pnl"], pos["qty"], pos["entry"]


@pytest.fixture
def fills(dash, monkeypatch):
    """Execution with slippage under test and commission isolated.

    Commission defaults to zero unless a test sets it, so a slippage assertion
    cannot pass or fail because of an unrelated cost.
    """
    base = dict(dash.cfg)
    dash.cfg["slippage_pct"] = 0.0
    dash.cfg["commission_pct"] = 0.0
    dash.cfg["settings_enabled"] = True
    dash.cfg["use_atr_exits"] = True
    monkeypatch.setattr(dash, "rl_feedback_entry", lambda *a, **k: None)
    monkeypatch.setattr(dash, "rl_feedback_exit", lambda *a, **k: None)
    yield dash
    dash.cfg.clear()
    dash.cfg.update(base)


# ── The fill price moves against you on every side ──────────────────────────

def test_a_buy_fills_above_the_quote(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    assert m["positions"]["A.NS"]["entry"] == pytest.approx(100.1), (
        "a buy must fill above the quote — slippage is a cost, not a rebate"
    )


def test_a_sell_fills_below_the_quote(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate(positions={"A.NS": {"side": "long", "qty": 10, "entry": 100.0,
                                     "stop_loss": 1.0, "target": 9999.0,
                                     "atr": 1.0}})
    fills.execute_sell("A.NS", 100.0, "TEST", m)
    # Received 99.90 against an entry of 100.00: -0.10 per share, -1.00 total.
    assert m["realised_pnl"] == pytest.approx(-1.0)


def test_a_short_open_fills_below_the_quote(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_short("A.NS", 100.0, m, atr=2.0)
    assert m["positions"]["A.NS"]["entry"] == pytest.approx(99.9), (
        "sell-to-open must fill below the quote"
    )


def test_a_cover_fills_above_the_quote(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate(positions={"A.NS": {"side": "short", "qty": 10, "entry": 100.0,
                                     "stop_loss": 9999.0, "target": 1.0,
                                     "atr": 1.0}})
    fills.execute_cover("A.NS", 100.0, "TEST", m)
    # Bought back at 100.10 against an entry of 100.00: -1.00 total.
    assert m["realised_pnl"] == pytest.approx(-1.0)


# ── A flat round trip must cost money ───────────────────────────────────────

def test_a_long_round_trip_at_a_flat_price_loses_exactly_the_slippage(fills):
    """The clearest statement of the property: no price movement, still a loss.
    Without this, slippage can be 'working' in the wrong direction."""
    fills.cfg["slippage_pct"] = 0.001
    pnl, qty, entry = _long_round_trip(fills)
    # Paid 100.10, received 99.90: exactly 2 x 0.10 per share.
    assert entry == pytest.approx(100.1)
    assert pnl == pytest.approx(-0.2 * qty)


def test_a_long_round_trip_records_a_loss_not_a_win(fills):
    """A slippage-only loss still has to reach the win/loss counters, or the
    win rate flatters the strategy."""
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    fills.execute_sell("A.NS", 100.0, "TEST", m)
    assert m["realised_pnl"] < 0
    assert m["losses"] == 1
    assert m["wins"] == 0


def test_a_short_round_trip_at_a_flat_price_loses_the_same(fills):
    fills.cfg["slippage_pct"] = 0.001
    pnl, qty, entry = _short_round_trip(fills)
    # Sold at 99.90, bought back at 100.10: exactly 2 x 0.10 per share.
    assert entry == pytest.approx(99.9)
    assert pnl == pytest.approx(-0.2 * qty)


def test_the_round_trip_drag_is_twice_the_per_side_setting(fills):
    """Not 1% on entry and 1% again on the exit notional — the drag is 2x the
    per-side rate on a flat price, not compounding."""
    fills.cfg["slippage_pct"] = 0.01
    pnl, qty, _ = _long_round_trip(fills)
    notional = 100.0 * qty
    assert pnl == pytest.approx(-0.02 * notional)


def test_zero_slippage_reproduces_the_previous_behaviour_exactly(fills):
    """Backwards compatibility: with slippage off, a flat round trip is free."""
    fills.cfg["slippage_pct"] = 0.0
    pnl, _qty, _entry = _long_round_trip(fills)
    assert pnl == pytest.approx(0.0)


def test_slippage_costs_more_as_it_rises(fills):
    fills.cfg["slippage_pct"] = 0.0005
    small = _long_round_trip(fills)[0]
    fills.cfg["slippage_pct"] = 0.005
    large = _long_round_trip(fills)[0]
    assert large < small, "a 10x slippage setting did not cost 10x"
    assert large == pytest.approx(small * 10, rel=0.05)


# ── It propagates to everything derived from the entry ─────────────────────

def test_the_stop_and_target_are_measured_from_the_fill(fills):
    """A stop placed relative to the quote rather than the fill would sit one
    slippage-width away from where the position actually is.

    ATR drives both distances, so the fill price cancels out here; what is
    asserted is that they remain the intended ATR multiples of the *recorded*
    entry, which is the fill.
    """
    fills.cfg["slippage_pct"] = 0.01
    atr = 2.0
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=atr)
    pos = m["positions"]["A.NS"]
    sl_mult, tp_mult = fills.cfg["atr_sl_mult"], fills.cfg["atr_tp_mult"]
    assert pos["stop_loss"] == pytest.approx(pos["entry"] - atr * sl_mult, abs=0.01)
    assert pos["target"] == pytest.approx(pos["entry"] + atr * tp_mult, abs=0.01)


def test_the_stop_moves_with_the_fill_price(fills):
    """With slippage on, the stop sits below the fill by the ATR distance. With
    it off, it sits below the quote by the same amount — so the stop is
    unambiguously derived from whatever price was recorded as the entry."""
    fills.cfg["slippage_pct"] = 0.0
    off = _mstate()
    fills.execute_buy("A.NS", 100.0, off, atr=2.0)
    sl_off = off["positions"]["A.NS"]["entry"] - off["positions"]["A.NS"]["stop_loss"]

    fills.cfg["slippage_pct"] = 0.01
    on = _mstate()
    fills.execute_buy("A.NS", 100.0, on, atr=2.0)
    sl_on = on["positions"]["A.NS"]["entry"] - on["positions"]["A.NS"]["stop_loss"]

    assert sl_on == pytest.approx(sl_off), (
        "the stop distance does not follow the recorded entry price"
    )
    assert on["positions"]["A.NS"]["stop_loss"] == pytest.approx(
        on["positions"]["A.NS"]["entry"] - sl_off, abs=0.01
    )


def test_the_running_high_starts_at_the_fill_not_the_quote(fills):
    """Seeded at the quote, the trailing stop would immediately trail on the
    first tick even before the position has moved."""
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    assert m["positions"]["A.NS"]["running_high"] == pytest.approx(100.1)


def test_the_running_low_starts_at_the_fill_for_a_short(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_short("A.NS", 100.0, m, atr=2.0)
    assert m["positions"]["A.NS"]["running_low"] == pytest.approx(99.9)


# ── Cash accounting must not assume a better price than you get ────────────

def test_a_buy_cannot_spend_cash_it_will_not_receive(fills):
    """The guard has to use the fill price. Checking it against the quote lets a
    position open with more cash than exists once the fill is paid for."""
    fills.cfg["slippage_pct"] = 0.01
    m = _mstate(cash=10_100.0)
    qty = fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    if qty is None:
        return   # correctly refused; nothing to assert
    assert m["cash"] >= 0, f"cash went negative: {m['cash']}"
    spent = 10_100.0 - m["cash"]
    assert spent <= 10_100.0 + 1e-6
    assert m["positions"]["A.NS"]["qty"] * m["positions"]["A.NS"]["entry"] <= spent


def test_the_cash_guard_refuses_when_even_one_share_will_not_fit(fills):
    """Enough cash for the quoted price of a single share, but not for the fill.

    The guard uses the fill price throughout. When the position cap can still
    size an order down to something affordable the trade goes ahead smaller,
    which is correct; this asserts the remaining case, where nothing fits and
    the order must be refused with cash untouched.
    """
    fills.cfg["slippage_pct"] = 0.01
    m = _mstate(cash=100.0)          # exactly one share at the 100.00 quote
    qty = fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    assert qty is None, "opened a position the account could not afford at the fill"
    assert "A.NS" not in m["positions"]
    assert m["cash"] == pytest.approx(100.0), "cash moved on a refused buy"


def test_the_position_cap_sizes_down_using_the_fill_price(fills):
    """Where the cap can be met by buying fewer shares, it must size on what the
    shares actually cost rather than what they were quoted at."""
    fills.cfg["slippage_pct"] = 0.01
    m = _mstate(cash=10_050.0)
    qty = fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    if qty is None:
        return
    cost = m["positions"]["A.NS"]["entry"] * qty
    assert cost <= m["cash"], f"sized {qty} shares costing {cost} with only {m['cash']} free"
    assert m["cash"] >= 0


# ── It stacks with commission rather than replacing it ─────────────────────

def test_slippage_and_commission_are_both_charged(fills):
    """Slippage alone on a flat round trip costs 2 x the per-side drag.

    Commission is compared separately rather than stacked into the same run,
    because commission reduces the cash available and therefore changes the
    position size — so a combined figure is not comparable against a
    slippage-only run that sized differently.
    """
    fills.cfg["commission_pct"] = 0.0
    fills.cfg["slippage_pct"] = 0.001
    slip_only, qty, _ = _long_round_trip(fills)
    assert slip_only == pytest.approx(-0.2 * qty)

    fills.cfg["commission_pct"] = 0.001
    with_commission, qty2, _ = _long_round_trip(fills)
    assert with_commission < slip_only, (
        f"commission did not stack with slippage: {slip_only} -> {with_commission}"
    )


def test_slippage_is_counted_on_both_sides(fills):
    """Why slippage is modelled as a fill price rather than a fee: the entry is
    stored as what was paid and the exit uses what was received, so both sides
    land in P&L without any extra bookkeeping."""
    fills.cfg["commission_pct"] = 0.0
    fills.cfg["slippage_pct"] = 0.001
    pnl, qty, entry = _long_round_trip(fills)
    assert entry == pytest.approx(100.1)                 # paid up on entry
    assert pnl == pytest.approx(-0.1 * qty - 0.1 * qty)  # entry and exit drag both present


def test_entry_commission_is_missing_from_realised_pnl(fills):
    """A pre-existing accounting asymmetry, pinned so it cannot change silently.

    Entry commission is deducted from cash when the position opens, but
    realised_pnl is computed only from the exit side:
    `net_proceeds - entry * qty`. So the entry cost never reaches P&L, the win
    /loss counters, the daily loss limit or the drawdown kill-switch — all four
    read as one side cheaper than the trade actually was.

    Slippage does not have this problem, which is precisely why it is modelled
    as a fill price rather than as a fee.
    """
    fills.cfg["commission_pct"] = 0.001
    fills.cfg["slippage_pct"] = 0.0
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    qty = m["positions"]["A.NS"]["qty"]
    entry_commission = 100.0 * qty * 0.001

    # Cash was charged for it...
    spent = 1_000_000.0 - m["cash"] - qty * 100.0
    assert spent == pytest.approx(entry_commission)

    # ...but P&L only reflects the exit side of a flat round trip.
    fills.execute_sell("A.NS", 100.0, "TEST", m)
    assert m["realised_pnl"] == pytest.approx(-entry_commission)
    assert m["realised_pnl"] != pytest.approx(-2 * entry_commission), (
        "this test documents the asymmetry; if it now fails, P&L has started "
        "charging both sides and the fix should land with it"
    )


# ── It has to be visible, or nobody can tell it is on ──────────────────────

def test_the_trade_log_records_the_fill_price_not_the_quote(fills):
    fills.cfg["slippage_pct"] = 0.001
    m = _mstate()
    fills.execute_buy("A.NS", 100.0, m, atr=2.0)
    message = m["trade_log"][-1]["message"]
    assert "100.10" in message, (
        f"the logged fill does not show the slipped price: {message!r}"
    )
    assert " @ 100.00 " not in message, "the quote was logged instead of the fill"


def test_the_fill_helper_is_the_single_place_slippage_is_applied(fills):
    """Four call sites each computing their own adjustment is how the direction
    eventually gets inverted on one of them."""
    fills.cfg["slippage_pct"] = 0.001
    assert fills._fill_price(100.0, BUY) > 100.0
    assert fills._fill_price(100.0, SELL) < 100.0


def test_the_fill_helper_is_a_no_op_when_slippage_is_zero(fills):
    fills.cfg["slippage_pct"] = 0.0
    assert fills._fill_price(100.0, BUY) == pytest.approx(100.0)
    assert fills._fill_price(100.0, SELL) == pytest.approx(100.0)


def test_a_negative_slippage_setting_cannot_pay_the_trader(fills):
    """A negative value would make every trade profitable. Rejected, not clamped
    silently, so the mistake is visible in the log."""
    fills.cfg["slippage_pct"] = -0.01
    assert fills._fill_price(100.0, BUY) >= 100.0, (
        "a negative slippage setting paid the trader"
    )


def test_slippage_never_produces_a_non_positive_price(fills):
    """A large setting against a tiny quote must not produce a zero or negative
    fill, which would corrupt every downstream calculation."""
    fills.cfg["slippage_pct"] = 0.5
    assert fills._fill_price(0.0001, SELL) > 0
    assert fills._fill_price(0.0001, BUY) > 0