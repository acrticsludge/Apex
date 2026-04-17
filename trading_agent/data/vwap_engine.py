"""Rolling daily VWAP and band features for the RL trading agent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_daily_vwap(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """
    Add rolling VWAP and 2σ band features computed from daily OHLCV bars.

    This is a daily-bar approximation: typical_price × volume accumulated over
    a rolling window, not a true intraday session VWAP.

    Columns added:
        vwap_dist       – signed % distance of close from VWAP
        vwap_above      – 1 if close > VWAP, else 0
        vwap_band_2_dist – signed % distance of close from upper 2σ band
                          (negative = below band, i.e. not overextended to upside)
    """
    result = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = typical * df["volume"]

    rolling_tpv = tpv.rolling(lookback, min_periods=lookback)
    rolling_vol = df["volume"].rolling(lookback, min_periods=lookback)

    vwap = rolling_tpv.sum() / rolling_vol.sum()
    vwap_std = typical.rolling(lookback, min_periods=lookback).std()
    vwap_upper_2 = vwap + 2.0 * vwap_std

    result["vwap_dist"] = (df["close"] - vwap) / vwap.replace(0, np.nan)
    result["vwap_above"] = (df["close"] > vwap).astype(np.float32)
    result["vwap_band_2_dist"] = (df["close"] - vwap_upper_2) / vwap.replace(0, np.nan)

    return result
