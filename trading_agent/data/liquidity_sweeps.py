"""Liquidity sweep and swing-level features for the RL trading agent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity_sweeps(df: pd.DataFrame, swing_lookback: int = 20) -> pd.DataFrame:
    """
    Add liquidity sweep and structural features derived from daily OHLCV bars.

    All reference levels are shifted so no future information leaks into a row.

    Columns added:
        ls_pdh_dist        – (high - prev_day_high) / close
        ls_pdl_dist        – (low  - prev_day_low)  / close
        ls_swing_high_dist – (close - N-day swing high) / close
        ls_swing_low_dist  – (close - N-day swing low)  / close
        ls_sweep           –  1 = bullish SSL sweep, -1 = bearish BSL sweep, 0 = none
        ls_choch           –  1 = bullish CHoCH confirmed, -1 = bearish, 0 = none
    """
    result = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    pdh = high.shift(1)
    pdl = low.shift(1)

    # Swing levels: rolling max/min of the N days *before* the current bar
    swing_high = high.shift(1).rolling(swing_lookback, min_periods=swing_lookback).max()
    swing_low = low.shift(1).rolling(swing_lookback, min_periods=swing_lookback).min()

    result["ls_pdh_dist"] = (high - pdh) / close.replace(0, np.nan)
    result["ls_pdl_dist"] = (low - pdl) / close.replace(0, np.nan)
    result["ls_swing_high_dist"] = (close - swing_high) / close.replace(0, np.nan)
    result["ls_swing_low_dist"] = (close - swing_low) / close.replace(0, np.nan)

    # Sweep candle: wick pierces the swing level but the daily close rejects back
    bullish_sweep = (low < swing_low) & (close > swing_low)
    bearish_sweep = (high > swing_high) & (close < swing_high)

    ls_sweep = pd.Series(0.0, index=df.index)
    ls_sweep[bullish_sweep] = 1.0
    ls_sweep[bearish_sweep] = -1.0
    result["ls_sweep"] = ls_sweep

    # Change of Character: sweep on previous bar + close breaks the opposite swing level today
    prev_bullish = bullish_sweep.astype(float).shift(1).fillna(0).astype(bool)
    prev_bearish = bearish_sweep.astype(float).shift(1).fillna(0).astype(bool)
    prev_swing_high = swing_high.shift(1)
    prev_swing_low = swing_low.shift(1)

    choch_bull = prev_bullish & (close > prev_swing_high)
    choch_bear = prev_bearish & (close < prev_swing_low)

    ls_choch = pd.Series(0.0, index=df.index)
    ls_choch[choch_bull] = 1.0
    ls_choch[choch_bear] = -1.0
    result["ls_choch"] = ls_choch

    return result
