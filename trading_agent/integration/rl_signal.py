"""RL inference bridge — converts the trained PPO model into analyse()-compatible signals."""

from __future__ import annotations

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)

# ── Singletons loaded once on first call ─────────────────────────────────────
_model = None
_scaler = None
_feature_columns: list[str] = []

# ── Per-ticker observation cache (TTL = 1 hour, daily bars don't change faster) ──
_obs_cache: dict[str, tuple[float, np.ndarray, float]] = {}  # (ts, obs, raw_atr)
_OBS_TTL = 3600


def _ensure_loaded() -> bool:
    global _model, _scaler, _feature_columns
    if _model is not None:
        return True

    try:
        import joblib
        from stable_baselines3 import PPO

        from trading_agent.config import settings
        from trading_agent.data.data_fetcher import load_saved_feature_columns

        if not settings.best_model_path.exists():
            logger.warning("RL model not found at %s", settings.best_model_path)
            return False

        _model = PPO.load(str(settings.best_model_path))
        _feature_columns = load_saved_feature_columns(settings)

        if settings.scaler_path.exists():
            _scaler = joblib.load(str(settings.scaler_path))

        logger.info("RL model loaded — %d features", len(_feature_columns))
        return True

    except Exception as exc:
        logger.error("RL model load failed: %s", exc)
        return False


def _build_observation(ticker: str) -> tuple[np.ndarray, float] | None:
    now = time.time()
    cached = _obs_cache.get(ticker)
    if cached and (now - cached[0]) < _OBS_TTL:
        return cached[1], cached[2]

    try:
        import pandas as pd
        import yfinance as yf

        from trading_agent.config import settings
        from trading_agent.data.indicator_engine import add_technical_indicators
        from trading_agent.data.sentiment_engine import add_sentiment_feature

        end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
        start = end - pd.DateOffset(years=2)

        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            actions=False,
        )
        if raw.empty or len(raw) < 60:
            return None

        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw.columns = [str(c).lower() for c in raw.columns]
        raw = raw.dropna()

        frame = add_technical_indicators(raw)

        if settings.enable_vwap:
            from trading_agent.data.vwap_engine import add_daily_vwap
            frame = add_daily_vwap(frame, lookback=settings.vwap_lookback)
        if settings.enable_volume_profile:
            from trading_agent.data.volume_profile import add_volume_profile
            frame = add_volume_profile(frame, lookback=settings.vp_lookback)
        if settings.enable_liquidity_sweeps:
            from trading_agent.data.liquidity_sweeps import add_liquidity_sweeps
            frame = add_liquidity_sweeps(frame, swing_lookback=settings.swing_lookback)
        if settings.enable_sentiment:
            frame = add_sentiment_feature(frame, ticker, settings)

        frame = frame.dropna(subset=_feature_columns)
        if frame.empty:
            return None

        raw_atr = float(frame["atr_14"].iloc[-1]) if "atr_14" in frame.columns else 0.0

        last_row = frame[_feature_columns].iloc[[-1]]
        obs = (_scaler.transform(last_row) if _scaler is not None else last_row.values)[0]
        obs = obs.astype(np.float32)

        _obs_cache[ticker] = (now, obs, raw_atr)
        return obs, raw_atr

    except Exception as exc:
        logger.warning("Observation build failed for %s: %s", ticker, exc)
        return None


def _get_extreme_features(obs: np.ndarray, cols: list[str]) -> list[dict]:
    """Return top-3 features furthest from neutral (0.5 in MinMax-scaled space)."""
    deviations = [(abs(float(v) - 0.5), i) for i, v in enumerate(obs)]
    deviations.sort(reverse=True)
    result = []
    for dev, idx in deviations[:3]:
        val = float(obs[idx])
        result.append({
            "feature":   cols[idx] if idx < len(cols) else f"feat_{idx}",
            "value":     round(val, 3),
            "direction": "high" if val > 0.5 else "low",
        })
    return result


def get_cached_obs(ticker: str) -> np.ndarray | None:
    """Return the most-recently-built observation array for a ticker, or None."""
    entry = _obs_cache.get(ticker)
    if entry is None:
        return None
    return entry[1].copy()


def get_rl_signal(symbol: str, live_price: float) -> dict | None:
    """
    Return an analyse()-compatible dict driven by the PPO policy.

    Extra keys:
      rl_action  — 0=hold  1=buy  2=sell
      rl_probs   — raw softmax probabilities [hold, buy, sell]
    """
    if not _ensure_loaded():
        return None

    result = _build_observation(symbol)
    if result is None:
        return None
    obs, raw_atr = result

    try:
        import torch

        obs_tensor, _ = _model.policy.obs_to_tensor(obs.reshape(1, -1))
        with torch.no_grad():
            dist = _model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]

        action = int(np.argmax(probs))
        top_prob = float(probs[action])

        entropy  = -sum(float(p) * math.log(float(p) + 1e-9) for p in probs)
        sorted_p = sorted(float(p) for p in probs)
        margin   = sorted_p[-1] - sorted_p[-2]
        extremes = _get_extreme_features(obs, _feature_columns)

        # Gate: skip signals where the model is uncertain
        if entropy > math.log(3) * 0.90 or margin < 0.08:
            return None

        # Scale confidence/score by conviction — a 35% plurality BUY is not the same
        # as a 90% BUY. Normalize top_prob out of [1/3 (random), 1.0].
        rand_floor = 1.0 / 3.0
        norm_prob  = min(1.0, max(0.0, (top_prob - rand_floor) / (1.0 - rand_floor)))

        if action == 1:   # BUY
            score      = round(40 + norm_prob * 60.0)
            confidence = round(55.0 + norm_prob * 40.0, 1)
        elif action == 2: # SELL / bearish
            score      = -round(40 + norm_prob * 60.0)
            confidence = round(45.0 - norm_prob * 40.0, 1)
        else:             # HOLD
            score, confidence = 0, 50.0

        # Use the actual ATR from the fetched data for correct SL/TP sizing
        atr = raw_atr if raw_atr > 0 else live_price * 0.015

        label = ["HOLD", "BUY", "SELL"][action]
        return {
            "symbol":          symbol,
            "price":           live_price,
            "score":           score,
            "confidence":      confidence,
            "signals":         {"RL": {"value": f"{label} {top_prob:.0%}", "signal": label}},
            "rsi":             50.0,
            "bb_upper":        live_price * 1.05,
            "bb_lower":        live_price * 0.95,
            "atr":             atr,
            "adx":             30.0,
            "hist_win_days":   0,
            "hist_total_days": 0,
            "hist_bias":       0.0,
            "news_score":      0.0,
            "news_count":      0,
            "rl_action":       action,
            "rl_probs":        probs.tolist(),
            "rl_entropy":      round(entropy, 4),
            "rl_margin":       round(margin, 4),
            "rl_extremes":     extremes,
        }

    except Exception as exc:
        logger.warning("RL inference failed for %s: %s", symbol, exc)
        return None
