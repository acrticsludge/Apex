"""Rolling Volume Profile features (POC, VAH, VAL) for the RL trading agent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_vp_row(prices: pd.Series, volumes: pd.Series) -> tuple[float, float, float]:
    """Return (poc, vah, val) for a window of (typical_price, volume) pairs."""
    total_vol = volumes.sum()
    if total_vol == 0:
        mid = prices.mean()
        return mid, mid, mid

    target = 0.70 * total_vol
    sorted_idx = volumes.argsort()[::-1]
    sorted_prices = prices.iloc[sorted_idx].values
    sorted_vols = volumes.iloc[sorted_idx].values

    accumulated = 0.0
    included_prices: list[float] = []
    for price, vol in zip(sorted_prices, sorted_vols):
        accumulated += vol
        included_prices.append(float(price))
        if accumulated >= target:
            break

    poc = float(sorted_prices[0])
    vah = max(included_prices)
    val = min(included_prices)
    return poc, vah, val


def add_volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add rolling Volume Profile features computed from daily OHLCV bars.

    Each row's profile is built from the prior `lookback` days so there is no
    lookahead bias.

    Columns added:
        vp_poc_dist      – (close - POC) / close — proximity to Point of Control
        vp_in_value_area – 1 if VAL ≤ close ≤ VAH, else 0
        vp_above_poc     – 1 if close > POC, else 0
    """
    result = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"]
    close = df["close"]

    n = len(df)
    poc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        window_prices = typical.iloc[i - lookback + 1 : i + 1]
        window_vols = volume.iloc[i - lookback + 1 : i + 1]
        poc_arr[i], vah_arr[i], val_arr[i] = _compute_vp_row(window_prices, window_vols)

    poc = pd.Series(poc_arr, index=df.index)
    vah = pd.Series(vah_arr, index=df.index)
    val = pd.Series(val_arr, index=df.index)

    result["vp_poc_dist"] = (close - poc) / close.replace(0, np.nan)
    result["vp_in_value_area"] = ((close >= val) & (close <= vah)).astype(np.float32)
    result["vp_above_poc"] = (close > poc).astype(np.float32)

    return result
