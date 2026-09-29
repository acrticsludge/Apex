"""JEV state v2 — builder tests for expanded context (additive, no breaking changes)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import apex_jev as jev


def _base_indicators(**over):
    d = {"price": 100.0, "rsi": 55.0, "macd_hist": 0.5, "adx": 25.0,
         "atr": 2.0, "vol_ratio": 1.2}
    d.update(over)
    return d


def test_new_indicator_keys_present_with_defaults():
    s = json.loads(jev.build_market_state("AAPL", _base_indicators(), [],
                                          {"cash": 1}, {"spy_trend_pct": 0}))
    ind = s["indicators"]
    assert ind["atr_pct"] == 0.0
    assert ind["ema_gap_pct"] == 0.0
    assert ind["bb_pos"] == 0.5  # default preserved


def test_new_indicator_values_pass_through():
    s = json.loads(jev.build_market_state(
        "AAPL", _base_indicators(bb_pos=0.8, ema_gap_pct=1.5, atr_pct=0.02,
                                 ema_9_21="bullish"),
        [], {"cash": 1}, {"spy_trend_pct": 0}))
    ind = s["indicators"]
    assert ind["bb_pos"] == 0.8
    assert ind["ema_gap_pct"] == 1.5
    assert ind["atr_pct"] == 0.02
    assert ind["ema_9_21"] == "bullish"  # old key untouched


def test_old_callers_without_new_keys_still_work():
    # callers passing only the original keys must not break
    s = json.loads(jev.build_market_state("RELIANCE.NS", {"price": 2500.0},
                                          [], {}, {}))
    assert s["symbol"] == "RELIANCE.NS"
    assert s["indicators"]["rsi"] == 50.0  # defaults intact


def test_calc_breadth_empty_is_neutral():
    assert jev.calc_breadth([]) == 0.5


def test_calc_breadth_all_bullish():
    assert jev.calc_breadth([10, 5, 1]) == 1.0


def test_calc_breadth_all_bearish():
    assert jev.calc_breadth([-10, -5, -1]) == 0.0


def test_calc_breadth_mixed():
    assert jev.calc_breadth([10, -10, 5, -5]) == 0.5
    assert jev.calc_breadth([10, 10, -10]) == jev.calc_breadth([10, 10, -10])


def test_fetch_vix_unknown_market_falls_back_without_network():
    assert jev.fetch_vix("unknown-market") == 20.0


def test_fetch_vix_never_raises():
    assert isinstance(jev.fetch_vix("us"), float)
