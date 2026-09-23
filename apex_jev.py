"""
Apex JEV (TypeSafe) Integration Module
=======================================
Provides structured judgment overlay for trading decisions.

JEV Primitives Used:
- Choice: regime, position_action
- Score: trend_strength, news_bullishness, portfolio_stress
- Noul: halt_new_buys

All questions evaluated in parallel per TypeSafe System One architecture.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Literal, TypedDict

import requests

from apex_dashboard import cfg

logger = logging.getLogger("apex.jev")

# ─── Type Definitions ──────────────────────────────────────────────────────────

class JEVRegime(TypedDict):
    choice: Literal["bullish", "bearish", "choppy", "crisis"]
    probabilities: dict[str, float]
    confidence: float

class JEVScore(TypedDict):
    score: float
    probabilities: dict[int, float]
    confidence: float
    legend: dict[int, str]

class JEVNoul(TypedDict):
    noul: float
    confidence: float

class JEVAction(TypedDict):
    choice: Literal["buy", "add", "hold", "trim", "exit"]
    probabilities: dict[str, float]
    confidence: float

class JEVDecisions(TypedDict):
    regime: JEVRegime
    trend_strength: JEVScore
    news_bullishness: JEVScore
    portfolio_stress: JEVScore
    halt_new_buys: JEVNoul
    position_action: JEVAction

# ─── Configuration ─────────────────────────────────────────────────────────────

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")

# Confidence thresholds (overridable via env)
REGIME_CONF_THRESHOLD = float(os.getenv("JEV_REGIME_CONFIDENCE_THRESHOLD", "0.65"))
ACTION_CONF_THRESHOLD = float(os.getenv("JEV_ACTION_CONFIDENCE_THRESHOLD", "0.65"))
TREND_CONF_THRESHOLD = float(os.getenv("JEV_TREND_CONFIDENCE_THRESHOLD", "0.70"))
NEWS_CONF_THRESHOLD = float(os.getenv("JEV_NEWS_CONFIDENCE_THRESHOLD", "0.70"))
HALT_NOUl_THRESHOLD = float(os.getenv("JEV_HALT_NOUl_THRESHOLD", "0.80"))

# Cache TTL (seconds)
CACHE_TTL_SECONDS = int(os.getenv("JEV_CACHE_TTL_SECONDS", "300"))

# Max calls per cycle
MAX_CALLS_PER_CYCLE = int(os.getenv("JEV_MAX_CALLS_PER_CYCLE", "10"))

# ─── Frozen Question Schema ────────────────────────────────────────────────────

JEV_QUESTIONS: dict[str, dict[str, Any]] = {
    "regime": {
        "type": "choice",
        "instructions": "What is the current market regime for this symbol?",
        "criteria": {
            "bullish": "Uptrend, higher highs/lows, positive breadth, VIX < 15",
            "bearish": "Downtrend, lower highs/lows, negative breadth, VIX > 25",
            "choppy": "Range-bound, mixed signals, ADX < 20, whipsaws",
            "crisis": "Sharp drawdown > 3% in 5d, VIX > 30, panic volume",
        },
    },
    "trend_strength": {
        "type": "score",
        "instructions": "How strong is the current trend?",
        "criteria": [
            "No trend (ADX < 15, EMAs flat/converging, volume declining)",
            "Weak trend (ADX 15-25, EMAs slightly separated, mixed volume)",
            "Strong trend (ADX 25-40, EMAs well separated, confirming volume)",
            "Explosive trend (ADX > 40, parabolic, extreme volume)",
        ],
    },
    "news_bullishness": {
        "type": "score",
        "instructions": "How bullish is recent news for this symbol?",
        "criteria": [
            "Very bearish (miss, guidance cut, fraud, bankruptcy risk)",
            "Bearish (negative sentiment, sector headwinds, downgrades)",
            "Neutral (mixed/no material news, routine filings)",
            "Bullish (beats, upgrades, positive catalysts, buybacks)",
            "Very bullish (transformational M&A, breakthrough, major deal)",
        ],
    },
    "portfolio_stress": {
        "type": "score",
        "instructions": "How stressed is the portfolio?",
        "criteria": [
            "Calm (DD < 2%, cash > 30%, winners > losers, low correlation)",
            "Cautious (DD 2-5%, cash 15-30%, some correlated losers)",
            "Stressed (DD 5-10%, cash < 15%, multiple losers, high correlation)",
            "Critical (DD > 10%, risk limits hit, forced selling risk)",
        ],
    },
    "halt_new_buys": {
        "type": "noul",
        "instructions": "Should new position entry be halted given current portfolio risk?",
    },
    "position_action": {
        "type": "choice",
        "instructions": "What action should be taken on this position?",
        "criteria": {
            "buy": "No position; strong setup; favorable R:R; regime supportive",
            "add": "Existing long; trend confirming; pullback to support; pyramid",
            "hold": "Thesis intact; no catalyst; wait for target/SL",
            "trim": "Extended move; take partial profits; reduce risk; trail rest",
            "exit": "Thesis broken; SL hit; regime change; news shock; EOD",
        },
    },
}

# ─── Circuit Breaker & Cache ──────────────────────────────────────────────────

_jev_cache: dict[str, tuple[float, JEVDecisions]] = {}  # market_key -> (ts, decisions)
_failure_count = 0
_circuit_open = False

# ─── State Builder ─────────────────────────────────────────────────────────────

def build_market_state(
    symbol: str,
    indicators: dict[str, Any],
    news: list[dict[str, Any]],
    portfolio: dict[str, Any],
    market: dict[str, Any],
) -> str:
    """
    Build JSON state string for JEV API call.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL", "RELIANCE.NS")
        indicators: Technical indicators dict (RSI, MACD, BB, ADX, ATR, etc.)
        news: List of news dicts with title, sentiment, recency_h
        portfolio: Portfolio state (cash, drawdown_pct, open_positions, daily_pnl_pct)
        market: Market context (spy_trend_pct, vix, breadth)
    
    Returns:
        JSON string suitable for JEV state field
    """
    state = {
        "symbol": symbol,
        "price": indicators.get("price", 0.0),
        "indicators": {
            "rsi": indicators.get("rsi", 50.0),
            "macd_hist": indicators.get("macd_hist", 0.0),
            "bb_pos": indicators.get("bb_pos", 0.5),  # position within BB [0,1]
            "ema_9_21": indicators.get("ema_9_21", "neutral"),
            "adx": indicators.get("adx", 20.0),
            "atr": indicators.get("atr", 0.0),
            "vol_ratio": indicators.get("vol_ratio", 1.0),
        },
        "news": news[:5],  # Top 5 most recent
        "portfolio": {
            "cash": portfolio.get("cash", 0.0),
            "drawdown_pct": portfolio.get("drawdown_pct", 0.0),
            "open_positions": portfolio.get("open_positions", 0),
            "daily_pnl_pct": portfolio.get("daily_pnl_pct", 0.0),
        },
        "market": {
            "spy_trend_pct": market.get("spy_trend_pct", 0.0),
            "vix": market.get("vix", 20.0),
            "breadth": market.get("breadth", 0.5),
        },
    }
    return json.dumps(state, separators=(",", ":"))


def build_system_state(
    market_key: str,
    portfolio: dict[str, Any],
    market: dict[str, Any],
) -> str:
    """Build state for system-level questions (regime, portfolio_stress, halt_new_buys)."""
    state = {
        "market": market_key,
        "portfolio": {
            "cash": portfolio.get("cash", 0.0),
            "drawdown_pct": portfolio.get("drawdown_pct", 0.0),
            "open_positions": portfolio.get("open_positions", 0),
            "daily_pnl_pct": portfolio.get("daily_pnl_pct", 0.0),
        },
        "market_context": {
            "spy_trend_pct": market.get("spy_trend_pct", 0.0),
            "vix": market.get("vix", 20.0),
            "breadth": market.get("breadth", 0.5),
        },
    }
    return json.dumps(state, separators=(",", ":"))


# ─── JEV API Client ────────────────────────────────────────────────────────────

def _call_jev_api(state: str, questions: dict[str, dict[str, Any]]) -> JEVDecisions | None:
    """Call TypeSafe JEV API with state and questions."""
    global _failure_count, _circuit_open
    
    api_key = os.getenv("TYPE_SAFE_API_KEY")
    if not api_key:
        logger.warning("TYPE_SAFE_API_KEY not set — JEV disabled")
        return None
    
    if _circuit_open:
        logger.debug("JEV circuit open — skipping call")
        return None
    
    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": questions,
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        response = requests.post(JEV_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        _failure_count = 0  # Reset on success
        
        # Parse responses into typed dict
        answers = data.get("answers", {})
        decisions: JEVDecisions = {
            "regime": _parse_choice(answers.get("regime")),
            "trend_strength": _parse_score(answers.get("trend_strength")),
            "news_bullishness": _parse_score(answers.get("news_bullishness")),
            "portfolio_stress": _parse_score(answers.get("portfolio_stress")),
            "halt_new_buys": _parse_noul(answers.get("halt_new_buys")),
            "position_action": _parse_choice(answers.get("position_action")),
        }
        return decisions
        
    except requests.Timeout:
        _failure_count += 1
        logger.warning(f"JEV API timeout (failure {_failure_count}/3)")
    except requests.HTTPError as e:
        _failure_count += 1
        logger.warning(f"JEV API HTTP error: {e} (failure {_failure_count}/3)")
    except requests.RequestException as e:
        _failure_count += 1
        logger.warning(f"JEV API error: {e} (failure {_failure_count}/3)")
    
    # Open circuit after 3 failures
    if _failure_count >= 3:
        _circuit_open = True
        logger.error("JEV circuit opened after 3 consecutive failures")
    
    return None


def _parse_choice(answer: dict[str, Any] | None) -> JEVRegime | JEVAction:
    if not answer:
        return {"choice": "hold", "probabilities": {}, "confidence": 0.0}  # type: ignore
    return {
        "choice": answer.get("choice", "hold"),
        "probabilities": answer.get("probabilities", {}),
        "confidence": answer.get("confidence", 0.0),
    }


def _parse_score(answer: dict[str, Any] | None) -> JEVScore:
    if not answer:
        return {"score": 0.0, "probabilities": {}, "confidence": 0.0, "legend": {}}
    return {
        "score": answer.get("score", 0.0),
        "probabilities": {int(k): v for k, v in answer.get("probabilities", {}).items()},
        "confidence": answer.get("confidence", 0.0),
        "legend": {int(k): v for k, v in answer.get("legend", {}).items()},
    }


def _parse_noul(answer: dict[str, Any] | None) -> JEVNoul:
    if not answer:
        return {"noul": 0.0, "confidence": 0.0}
    return {
        "noul": answer.get("noul", 0.0),
        "confidence": answer.get("confidence", 0.0),
    }


# ─── High-Level Interface ──────────────────────────────────────────────────────

def get_jev_decisions(
    symbol: str,
    indicators: dict[str, Any],
    news: list[dict[str, Any]],
    portfolio: dict[str, Any],
    market: dict[str, Any],
    market_key: str,
) -> JEVDecisions | None:
    """
    Get JEV decisions for a specific symbol.
    Uses cache for system-level questions (regime, portfolio_stress, halt_new_buys).
    """
    
    # Check cache for system-level decisions
    cache_key = f"system_{market_key}"
    now = time.time()
    
    if cache_key in _jev_cache:
        cached_ts, cached_decisions = _jev_cache[cache_key]
        if now - cached_ts < CACHE_TTL_SECONDS:
            logger.debug(f"JEV cache hit for {market_key}")
            # Still need symbol-specific questions
            symbol_state = build_market_state(symbol, indicators, news, portfolio, market)
            symbol_questions = {
                "news_bullishness": JEV_QUESTIONS["news_bullishness"],
                "position_action": JEV_QUESTIONS["position_action"],
            }
            symbol_result = _call_jev_api(symbol_state, symbol_questions)
            if symbol_result:
                # Merge cached system decisions with fresh symbol decisions
                return {
                    "regime": cached_decisions["regime"],
                    "trend_strength": cached_decisions["trend_strength"],
                    "news_bullishness": symbol_result["news_bullishness"],
                    "portfolio_stress": cached_decisions["portfolio_stress"],
                    "halt_new_buys": cached_decisions["halt_new_buys"],
                    "position_action": symbol_result["position_action"],
                }
            return cached_decisions
    
    # Cache miss or expired — full call
    if not cfg.get("jev_enabled", True):
        return None
    
    # Build combined state for all questions
    # System state for regime, trend, portfolio_stress, halt_new_buys
    # Symbol state for news_bullishness, position_action
    # TypeSafe evaluates all questions in parallel anyway, so we can combine
    full_state = build_market_state(symbol, indicators, news, portfolio, market)
    full_questions = JEV_QUESTIONS
    
    decisions = _call_jev_api(full_state, full_questions)
    
    if decisions:
        _jev_cache[cache_key] = (now, decisions)
        logger.info(f"JEV decisions for {symbol}: regime={decisions['regime']['choice']} "
                    f"conf={decisions['regime']['confidence']:.2f}")
    
    return decisions


def get_system_decisions(market_key: str, portfolio: dict, market: dict) -> JEVDecisions | None:
    """Get only system-level JEV decisions (regime, trend, stress, halt)."""
    
    cache_key = f"system_{market_key}"
    now = time.time()
    
    if cache_key in _jev_cache:
        cached_ts, cached_decisions = _jev_cache[cache_key]
        if now - cached_ts < CACHE_TTL_SECONDS:
            return cached_decisions
    
    if not cfg.get("jev_enabled", True):
        return None
    
    state = build_system_state(market_key, portfolio, market)
    questions = {
        "regime": JEV_QUESTIONS["regime"],
        "trend_strength": JEV_QUESTIONS["trend_strength"],
        "portfolio_stress": JEV_QUESTIONS["portfolio_stress"],
        "halt_new_buys": JEV_QUESTIONS["halt_new_buys"],
    }
    
    decisions = _call_jev_api(state, questions)
    
    if decisions:
        _jev_cache[cache_key] = (now, decisions)
        logger.info(f"JEV system decisions for {market_key}: regime={decisions['regime']['choice']}")
    
    return decisions


# ─── Gate Functions ────────────────────────────────────────────────────────────

def should_halt(jev: JEVDecisions) -> bool:
    """Check if new entries should be halted."""
    if not cfg.get("jev_risk_enabled", True):
        return False
    return jev["halt_new_buys"]["noul"] >= HALT_NOUl_THRESHOLD


def get_position_action(jev: JEVDecisions) -> tuple[str | None, float]:
    """
    Get position action if confidence exceeds threshold.
    Returns (action, confidence) or (None, 0.0).
    """
    if not cfg.get("jev_action_enabled", True):
        return None, 0.0
    action = jev["position_action"]["choice"]
    conf = jev["position_action"]["confidence"]
    if conf >= ACTION_CONF_THRESHOLD:
        return action, conf
    return None, 0.0


def get_regime_scaling(jev: JEVDecisions) -> dict[str, float]:
    """
    Get risk/position scaling factors based on regime.
    Returns dict with risk_mult, pos_delta, sl_mult.
    """
    if not cfg.get("jev_regime_enabled", True):
        return {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0}
    
    regime = jev["regime"]["choice"]
    conf = jev["regime"]["confidence"]
    
    if conf < REGIME_CONF_THRESHOLD:
        return {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0}
    
    scaling = {
        "bullish": {"risk_mult": 1.0, "pos_delta": 2, "sl_mult": 1.0},
        "bearish": {"risk_mult": 0.8, "pos_delta": -2, "sl_mult": 1.2},
        "choppy":  {"risk_mult": 0.7, "pos_delta": -1, "sl_mult": 1.1},
        "crisis":  {"risk_mult": 0.5, "pos_delta": -4, "sl_mult": 1.5},
    }
    return scaling.get(regime, {"risk_mult": 1.0, "pos_delta": 0, "sl_mult": 1.0})


def get_portfolio_stress_scaling(jev: JEVDecisions) -> float:
    """Get risk reduction factor from portfolio stress (0.5 to 1.0)."""
    if not cfg.get("jev_risk_enabled", True):
        return 1.0
    stress = jev["portfolio_stress"]["score"]
    conf = jev["portfolio_stress"]["confidence"]
    if conf < TREND_CONF_THRESHOLD:  # Reuse trend threshold for stress
        return 1.0
    # stress 0-1 → 1.0, 1-2 → 0.85, 2-3 → 0.7, 3+ → 0.5
    if stress <= 1.0:
        return 1.0
    elif stress <= 2.0:
        return 0.85
    elif stress <= 3.0:
        return 0.7
    else:
        return 0.5


def get_effective_adx_min(base_adx_min: float, jev: JEVDecisions) -> float:
    """Calculate effective ADX minimum with JEV trend_strength adjustment."""
    if not cfg.get("jev_trend_enabled", True):
        return base_adx_min
    trend = jev["trend_strength"]
    if trend["confidence"] < TREND_CONF_THRESHOLD:
        return base_adx_min
    # score 0→0.7x, 1.5→1.0x, 3→1.3x
    multiplier = 1.0 + (trend["score"] - 1.5) * 0.2
    return base_adx_min * multiplier


def get_trailing_dist_mult(base_dist_mult: float, jev: JEVDecisions) -> float:
    """Calculate trailing stop distance multiplier with JEV trend adjustment."""
    if not cfg.get("jev_trend_enabled", True):
        return base_dist_mult
    trend = jev["trend_strength"]
    if trend["confidence"] < TREND_CONF_THRESHOLD:
        return base_dist_mult
    # Strong trend (3.0) → 1.0x (tighter); No trend (0) → 2.0x (wider)
    return base_dist_mult * (2.0 - trend["score"] / 3.0)


def get_open_window_minutes(jev: JEVDecisions, default: int = 15) -> int:
    """Get market open observation window minutes by regime."""
    if not cfg.get("jev_regime_enabled", True):
        return default
    regime = jev["regime"]["choice"]
    conf = jev["regime"]["confidence"]
    if conf < REGIME_CONF_THRESHOLD:
        return default
    windows = {
        "crisis": 30,
        "bullish": 5,
        "bearish": 15,
        "choppy": 15,
    }
    return windows.get(regime, default)


def get_news_weight(jev: JEVDecisions) -> float:
    """Get news scoring weight based on JEV confidence."""
    if not cfg.get("jev_news_enabled", True):
        return 1.0
    news = jev["news_bullishness"]
    if news["confidence"] < NEWS_CONF_THRESHOLD:
        return 0.33  # Low confidence → 1/3 weight
    return 1.0  # High confidence → full weight


def apply_jev_gates(base_cfg: dict, jev: JEVDecisions) -> dict:
    """
    Apply JEV decisions to config, returning new config dict.
    Pure function — does not mutate base_cfg.
    """
    new_cfg = base_cfg.copy()
    
    # Regime scaling
    regime_scale = get_regime_scaling(jev)
    new_cfg["risk_per_trade"] = base_cfg["risk_per_trade"] * regime_scale["risk_mult"]
    new_cfg["stop_loss_pct"] = base_cfg["stop_loss_pct"] * regime_scale["sl_mult"]
    new_cfg["india_max_positions"] = max(1, base_cfg["india_max_positions"] + regime_scale["pos_delta"])
    new_cfg["us_max_positions"] = max(1, base_cfg["us_max_positions"] + regime_scale["pos_delta"])
    
    # Portfolio stress scaling
    stress_scale = get_portfolio_stress_scaling(jev)
    new_cfg["risk_per_trade"] = new_cfg["risk_per_trade"] * stress_scale
    
    # Graduated daily loss limit
    stress = jev["portfolio_stress"]["score"]
    if stress <= 1.0:
        new_cfg["daily_loss_limit_pct"] = 0.05
    elif stress <= 2.0:
        new_cfg["daily_loss_limit_pct"] = 0.03
    elif stress <= 3.0:
        new_cfg["daily_loss_limit_pct"] = 0.015
    else:
        new_cfg["daily_loss_limit_pct"] = 0.005
    
    # Graduated max drawdown
    if stress <= 1.0:
        new_cfg["max_drawdown_pct"] = 0.08
    elif stress <= 2.0:
        new_cfg["max_drawdown_pct"] = 0.06
    elif stress <= 3.0:
        new_cfg["max_drawdown_pct"] = 0.04
    else:
        new_cfg["max_drawdown_pct"] = 0.02
    
    # Open window
    new_cfg["open_filter_min"] = get_open_window_minutes(jev, base_cfg.get("open_filter_min", 15))
    
    return new_cfg


# ─── JEV Feature Vector for RL ─────────────────────────────────────────────────

JEV_FEATURE_COLUMNS = [
    "jev_regime_bullish",
    "jev_regime_bearish",
    "jev_regime_choppy",
    "jev_regime_crisis",
    "jev_trend_strength",
    "jev_news_bullishness",
    "jev_portfolio_stress",
    "jev_halt_noul",
]


def extract_jev_features(jev: JEVDecisions) -> list[float]:
    """Extract normalized JEV features for RL observation vector."""
    regime_probs = jev["regime"]["probabilities"]
    return [
        regime_probs.get("bullish", 0.0),
        regime_probs.get("bearish", 0.0),
        regime_probs.get("choppy", 0.0),
        regime_probs.get("crisis", 0.0),
        jev["trend_strength"]["score"] / 3.0,      # Normalize to [0,1]
        jev["news_bullishness"]["score"] / 4.0,    # Normalize to [0,1]
        jev["portfolio_stress"]["score"] / 3.0,    # Normalize to [0,1]
        jev["halt_new_buys"]["noul"],              # Already [0,1]
    ]


# ─── Utility ───────────────────────────────────────────────────────────────────

def reset_circuit() -> None:
    """Manually reset circuit breaker (for testing/recovery)."""
    global _failure_count, _circuit_open
    _failure_count = 0
    _circuit_open = False
    logger.info("JEV circuit breaker manually reset")


def clear_cache() -> None:
    """Clear JEV decision cache."""
    _jev_cache.clear()
    logger.info("JEV cache cleared")


def get_jev_status() -> dict[str, Any]:
    """Get JEV integration status for monitoring."""
    return {
        "enabled": cfg.get("jev_enabled", True),
        "circuit_open": _circuit_open,
        "failure_count": _failure_count,
        "cache_size": len(_jev_cache),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "model": JEV_MODEL,
    }