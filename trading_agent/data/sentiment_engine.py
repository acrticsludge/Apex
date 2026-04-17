"""Finnhub-powered sentiment feature engineering with local caching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trading_agent.config import Settings, settings


logger = logging.getLogger(__name__)
_ANALYZER = SentimentIntensityAnalyzer()


def _cache_path_for_ticker(ticker: str, current_settings: Settings) -> Path:
    """Create a filesystem-safe sentiment cache path for one ticker."""
    safe_name = ticker.replace(".", "_").replace("/", "_")
    return current_settings.sentiment_cache_dir / f"{safe_name}_sentiment.csv"


def _score_articles(articles: list[dict]) -> pd.Series:
    """Turn Finnhub company-news articles into a daily compound sentiment series."""
    records: list[dict[str, object]] = []

    for article in articles:
        published_at = article.get("datetime")
        headline = str(article.get("headline", "")).strip()
        summary = str(article.get("summary", "")).strip()
        text = f"{headline}. {summary}".strip(". ").strip()
        if not published_at or not text:
            continue

        article_date = pd.to_datetime(int(published_at), unit="s", utc=True).tz_convert(None).normalize()
        score = _ANALYZER.polarity_scores(text)["compound"]
        records.append({"date": article_date, "sentiment": score})

    if not records:
        return pd.Series(dtype="float64", name="sentiment")

    scored = pd.DataFrame(records)
    daily_scores = scored.groupby("date")["sentiment"].mean().sort_index()
    daily_scores.name = "sentiment"
    return daily_scores


def _load_cached_sentiment(cache_path: Path) -> pd.Series:
    """Load a previously cached daily sentiment series if one exists."""
    if not cache_path.exists():
        return pd.Series(dtype="float64", name="sentiment")

    cached = pd.read_csv(cache_path, parse_dates=["date"])
    if cached.empty:
        return pd.Series(dtype="float64", name="sentiment")

    cached = cached.drop_duplicates(subset=["date"]).sort_values("date")
    series = cached.set_index("date")["sentiment"].astype(float)
    series.name = "sentiment"
    return series


def _save_cached_sentiment(cache_path: Path, sentiment_series: pd.Series) -> None:
    """Persist the fetched sentiment series so future runs avoid re-downloading it."""
    cache_frame = sentiment_series.rename("sentiment").reset_index(names="date")
    cache_frame.to_csv(cache_path, index=False)


def fetch_daily_sentiment(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_settings: Settings = settings,
    refresh_cache: bool = False,
) -> pd.Series:
    """
    Fetch and cache daily sentiment derived from Finnhub company news.

    The Finnhub free tier is rate-limited, so the implementation downloads news
    in chunks and stores the resulting daily series locally. If the API key is
    missing or the request fails, the caller receives a neutral series later.
    """
    cache_path = _cache_path_for_ticker(ticker, current_settings)

    if not refresh_cache:
        cached = _load_cached_sentiment(cache_path)
        if not cached.empty and cached.index.min() <= start_date and cached.index.max() >= end_date:
            return cached.loc[start_date:end_date]

    if not current_settings.enable_sentiment or not current_settings.finnhub_api_key:
        logger.info("Finnhub sentiment disabled or API key missing for %s; using neutral sentiment.", ticker)
        return pd.Series(dtype="float64", name="sentiment")

    articles: list[dict] = []
    chunk_start = start_date.normalize()

    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + pd.Timedelta(days=current_settings.news_chunk_days - 1),
            end_date.normalize(),
        )
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": chunk_start.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
                "token": current_settings.finnhub_api_key,
            },
            timeout=current_settings.finnhub_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            articles.extend(payload)

        time.sleep(current_settings.finnhub_sleep_seconds)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    scored_series = _score_articles(articles)
    if not scored_series.empty:
        _save_cached_sentiment(cache_path, scored_series)
    return scored_series.loc[start_date:end_date]


def add_sentiment_feature(
    price_frame: pd.DataFrame,
    ticker: str,
    current_settings: Settings = settings,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Add a daily ``sentiment`` column in the raw ``[-1, +1]`` range.

    Gaps are forward-filled for a few days to keep nearby news influence alive,
    then any remaining missing values are set to neutral sentiment.
    """
    frame = price_frame.copy()
    date_index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if len(date_index) == 0:
        frame["sentiment"] = 0.0
        return frame

    try:
        daily_sentiment = fetch_daily_sentiment(
            ticker=ticker,
            start_date=date_index.min(),
            end_date=date_index.max(),
            current_settings=current_settings,
            refresh_cache=refresh_cache,
        )
    except requests.RequestException as exc:
        logger.warning("Finnhub sentiment request failed for %s: %s", ticker, exc)
        daily_sentiment = pd.Series(dtype="float64", name="sentiment")

    aligned = (
        daily_sentiment.reindex(date_index)
        .ffill(limit=current_settings.sentiment_forward_fill_limit)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )
    frame["sentiment"] = aligned.to_numpy(dtype="float64")
    return frame
