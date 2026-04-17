"""Historical market data pipeline for the standalone RL trading agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from trading_agent.config import Settings, settings
from trading_agent.data.indicator_engine import add_technical_indicators
from trading_agent.data.sentiment_engine import add_sentiment_feature


logger = logging.getLogger(__name__)


@dataclass
class PreparedDataBundle:
    """Container holding split datasets and the fitted scaler."""

    train_frames: dict[str, pd.DataFrame]
    validation_frames: dict[str, pd.DataFrame]
    test_frames: dict[str, pd.DataFrame]
    feature_columns: list[str]
    scaler: MinMaxScaler
    tickers: list[str]
    metadata: dict[str, Any]


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance outputs so downstream code always sees simple columns."""
    flattened = frame.copy()
    flattened.columns = [column[0] if isinstance(column, tuple) else column for column in flattened.columns]
    flattened.columns = [str(column).lower() for column in flattened.columns]
    return flattened


def download_ohlcv_history(
    ticker: str,
    current_settings: Settings = settings,
) -> pd.DataFrame:
    """
    Download ten years of daily OHLCV data for one ticker using yfinance.

    ``auto_adjust=True`` keeps the series split/dividend adjusted, which is more
    appropriate for training a policy on long-horizon historical data.
    """
    end_date = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=current_settings.lookback_years)

    history = yf.download(
        tickers=ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=current_settings.interval,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=False,
    )

    if history.empty:
        raise ValueError(f"No OHLCV data returned for ticker '{ticker}'.")

    history = _flatten_yfinance_columns(history)
    required_columns = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in history.columns]
    if missing:
        raise ValueError(f"Ticker '{ticker}' is missing expected OHLCV columns: {missing}")

    history = history[required_columns].dropna().sort_index()
    history.index = pd.DatetimeIndex(history.index).tz_localize(None).normalize()
    return history


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe chronologically to avoid lookahead bias.

    Training data always comes first in time, followed by validation, then test.
    """
    total_rows = len(frame)
    train_end = int(total_rows * train_ratio)
    validation_end = train_end + int(total_rows * validation_ratio)

    train_frame = frame.iloc[:train_end].copy()
    validation_frame = frame.iloc[train_end:validation_end].copy()
    test_frame = frame.iloc[validation_end:].copy()

    return train_frame, validation_frame, test_frame


def _preserve_raw_columns(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Keep raw feature values for evaluation while observations use scaled columns."""
    preserved = frame.copy()
    for column in feature_columns:
        preserved[f"{column}_raw"] = preserved[column].astype(float)
    return preserved


def _scale_frame(frame: pd.DataFrame, scaler: MinMaxScaler, feature_columns: list[str]) -> pd.DataFrame:
    """Apply the training-only scaler to one dataframe split."""
    scaled = _preserve_raw_columns(frame, feature_columns)
    scaled_values = scaler.transform(frame[feature_columns])
    scaled[feature_columns] = scaled_values.astype(np.float32)
    scaled["ticker"] = str(frame["ticker"].iloc[0])
    return scaled


def save_preprocessing_artifacts(bundle: PreparedDataBundle, current_settings: Settings = settings) -> None:
    """Persist fitted preprocessing state so serving and later retraining can reuse it."""
    current_settings.ensure_directories()
    joblib.dump(bundle.scaler, current_settings.scaler_path)
    current_settings.feature_columns_path.write_text(
        json.dumps(bundle.feature_columns, indent=2),
        encoding="utf-8",
    )
    current_settings.dataset_metadata_path.write_text(
        json.dumps(bundle.metadata, indent=2),
        encoding="utf-8",
    )


def load_saved_feature_columns(current_settings: Settings = settings) -> list[str]:
    """Load the saved feature ordering expected by the model bridge."""
    if current_settings.feature_columns_path.exists():
        return json.loads(current_settings.feature_columns_path.read_text(encoding="utf-8"))
    return list(current_settings.feature_columns)


def prepare_datasets(
    tickers: list[str] | None = None,
    current_settings: Settings = settings,
    refresh_sentiment: bool = False,
) -> PreparedDataBundle:
    """
    Build the end-to-end feature store used by training and evaluation.

    The scaler is fit only on the concatenated training splits. Validation and
    test splits are transformed with the same scaler so future information never
    leaks into earlier periods.
    """
    current_settings.ensure_directories()
    current_settings.save_runtime_snapshot()

    resolved_tickers = current_settings.resolved_tickers(tickers)
    processed_frames: dict[str, pd.DataFrame] = {}
    skipped_tickers: dict[str, str] = {}

    for ticker in resolved_tickers:
        try:
            ohlcv_frame = download_ohlcv_history(ticker=ticker, current_settings=current_settings)
            feature_frame = add_technical_indicators(ohlcv_frame)
            feature_frame = add_sentiment_feature(
                price_frame=feature_frame,
                ticker=ticker,
                current_settings=current_settings,
                refresh_cache=refresh_sentiment,
            )
            feature_frame["ticker"] = ticker
            feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()

            if len(feature_frame) < current_settings.minimum_rows_after_features:
                raise ValueError(
                    f"Only {len(feature_frame)} usable rows remained after feature generation."
                )

            processed_frames[ticker] = feature_frame
        except Exception as exc:  # pragma: no cover - network-dependent failures
            logger.warning("Skipping ticker %s because preprocessing failed: %s", ticker, exc)
            skipped_tickers[ticker] = str(exc)

    if not processed_frames:
        raise RuntimeError("No tickers produced a valid training dataset.")

    train_frames: dict[str, pd.DataFrame] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    test_frames: dict[str, pd.DataFrame] = {}

    for ticker, frame in processed_frames.items():
        train_frame, validation_frame, test_frame = split_frame(
            frame=frame,
            train_ratio=current_settings.train_split,
            validation_ratio=current_settings.validation_split,
        )

        if min(len(train_frame), len(validation_frame), len(test_frame)) < current_settings.min_episode_length:
            logger.warning(
                "Skipping ticker %s because one split is too short (train=%s, validation=%s, test=%s).",
                ticker,
                len(train_frame),
                len(validation_frame),
                len(test_frame),
            )
            skipped_tickers[ticker] = "One or more chronological splits were too short."
            continue

        train_frames[ticker] = train_frame
        validation_frames[ticker] = validation_frame
        test_frames[ticker] = test_frame

    if not train_frames:
        raise RuntimeError("No ticker had enough rows to survive the train/validation/test split.")

    train_matrix = pd.concat(
        [frame[current_settings.feature_columns] for frame in train_frames.values()],
        axis=0,
    )

    scaler = MinMaxScaler()
    scaler.fit(train_matrix)

    scaled_train_frames = {
        ticker: _scale_frame(frame, scaler, current_settings.feature_columns)
        for ticker, frame in train_frames.items()
    }
    scaled_validation_frames = {
        ticker: _scale_frame(frame, scaler, current_settings.feature_columns)
        for ticker, frame in validation_frames.items()
    }
    scaled_test_frames = {
        ticker: _scale_frame(frame, scaler, current_settings.feature_columns)
        for ticker, frame in test_frames.items()
    }

    bundle = PreparedDataBundle(
        train_frames=scaled_train_frames,
        validation_frames=scaled_validation_frames,
        test_frames=scaled_test_frames,
        feature_columns=list(current_settings.feature_columns),
        scaler=scaler,
        tickers=list(scaled_train_frames.keys()),
        metadata={
            "tickers": list(scaled_train_frames.keys()),
            "skipped_tickers": skipped_tickers,
            "feature_columns": list(current_settings.feature_columns),
            "rows_per_ticker": {
                ticker: {
                    "train": len(scaled_train_frames[ticker]),
                    "validation": len(scaled_validation_frames[ticker]),
                    "test": len(scaled_test_frames[ticker]),
                }
                for ticker in scaled_train_frames
            },
            "settings": current_settings.to_dict(),
        },
    )
    save_preprocessing_artifacts(bundle=bundle, current_settings=current_settings)
    return bundle
