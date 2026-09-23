"""Technical indicator generation built on top of pandas-ta."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_technical_indicators(price_frame: pd.DataFrame, jev_trend_strength: float | None = None) -> pd.DataFrame:
    """
    Add the required technical indicators to an OHLCV dataframe.

    The function expects lower-case ``open/high/low/close/volume`` columns and
    returns a copy so the raw input is never mutated upstream.
    """
    frame = price_frame.copy()

    # Momentum
    frame["rsi_14"] = ta.rsi(frame["close"], length=14)

    # Trend and momentum crossover family
    macd = ta.macd(frame["close"], fast=12, slow=26, signal=9)
    frame["macd"] = macd.iloc[:, 0]
    frame["macd_hist"] = macd.iloc[:, 1]
    frame["macd_signal"] = macd.iloc[:, 2]

    # Volatility envelope
    bollinger = ta.bbands(frame["close"], length=20, std=2.0)
    frame["bb_lower"] = bollinger.iloc[:, 0]
    frame["bb_mid"] = bollinger.iloc[:, 1]
    frame["bb_upper"] = bollinger.iloc[:, 2]

    # Trend filters
    frame["ema_20"] = ta.ema(frame["close"], length=20)
    frame["ema_50"] = ta.ema(frame["close"], length=50)

    # Volatility and participation
    frame["atr_14"] = ta.atr(frame["high"], frame["low"], frame["close"], length=14)
    frame["obv"] = ta.obv(frame["close"], frame["volume"])

    # JEV trend strength (Task 21)
    if jev_trend_strength is not None:
        frame["jev_trend_strength"] = jev_trend_strength

    return frame
