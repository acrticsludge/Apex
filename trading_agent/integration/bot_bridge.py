"""FastAPI bridge that exposes the trained PPO policy to external bots."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

from trading_agent.agent.evaluate import run_evaluation
from trading_agent.agent.train import run_training
from trading_agent.config import Settings, settings
from trading_agent.data.data_fetcher import load_saved_feature_columns


logger = logging.getLogger(__name__)
app = FastAPI(title="RL Trading Agent Bridge", version="1.0.0")


class PredictionRequest(BaseModel):
    """Schema for external trading bots that want an action from the PPO model."""

    observation: list[float] = Field(..., description="Normalized feature vector in saved feature order.")
    jev_features: list[float] | None = Field(None, description="Optional JEV features (8 dims: regime_bullish, regime_bearish, regime_choppy, regime_crisis, trend_strength, news_bullishness, portfolio_stress, halt_noul).")


class PredictionResponse(BaseModel):
    """Action plus a confidence score derived from the policy distribution."""

    action: int
    confidence: float
    jev_regime_probs: dict[str, float] | None = Field(None, description="JEV regime probabilities if JEV features provided.")


class RetrainRequest(BaseModel):
    """Optional retraining controls for the bridge endpoint."""

    tickers: list[str] | None = None
    refresh_sentiment: bool = False


class BridgeState:
    """Mutable runtime state tracked by the FastAPI service."""

    def __init__(self) -> None:
        self.model: PPO | None = None
        self.model_path: str | None = None
        self.feature_columns: list[str] = []
        self.last_prediction: dict[str, Any] | None = None
        self.retraining: bool = False
        self.last_retrain: dict[str, Any] | None = None


bridge_state = BridgeState()


def _resolve_model_path(current_settings: Settings) -> Path | None:
    """Use the best saved checkpoint when available, otherwise fall back to latest."""
    if current_settings.best_model_path.exists():
        return current_settings.best_model_path
    if current_settings.latest_model_path.exists():
        return current_settings.latest_model_path
    return None


def load_model_bundle(current_settings: Settings = settings) -> bool:
    """Load the PPO model plus feature metadata into the bridge runtime."""
    model_path = _resolve_model_path(current_settings)
    if model_path is None:
        bridge_state.model = None
        bridge_state.model_path = None
        bridge_state.feature_columns = load_saved_feature_columns(current_settings)
        return False

    bridge_state.model = PPO.load(str(model_path))
    bridge_state.model_path = str(model_path)
    bridge_state.feature_columns = load_saved_feature_columns(current_settings)
    return True


def _predict_action_and_confidence(observation: list[float]) -> tuple[int, float, dict[str, float] | None]:
    """Run a forward pass through the PPO policy and expose action probabilities."""
    if bridge_state.model is None:
        raise HTTPException(status_code=503, detail="No trained model is loaded.")

    # Base feature size (without JEV features)
    base_feature_size = len(bridge_state.feature_columns) - 8  # 30 - 8 = 22
    
    if len(observation) == expected_size + 8:
        # Full observation with JEV features
        base_observation = observation[:base_feature_size]
        jev_features = observation[base_feature_size:]
    elif len(observation) == expected_size:
        # Full observation without JEV features (backward compatible)
        base_observation = observation
        jev_features = None
    elif len(observation) == base_feature_size:
        # Base observation only
        base_observation = observation
        jev_features = None
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Observation length mismatch. Expected {base_feature_size} (base) or {expected_size} (with JEV), received {len(observation)}.",
        )

    observation_array = np.asarray(base_observation, dtype=np.float32).reshape(1, -1)
    obs_tensor, _ = bridge_state.model.policy.obs_to_tensor(observation_array)

    with torch.no_grad():
        distribution = bridge_state.model.policy.get_distribution(obs_tensor)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]

    action = int(np.argmax(probabilities))
    confidence = float(probabilities[action])
    
    # Return JEV regime probabilities if available
    jev_regime_probs = None
    if jev_features is not None and len(jev_features) >= 4:
        jev_regime_probs = {
            "bullish": float(jev_features[0]),
            "bearish": float(jev_features[1]),
            "choppy": float(jev_features[2]),
            "crisis": float(jev_features[3]),
        }
    
    return action, confidence, jev_regime_probs


def _retrain_worker(
    tickers: list[str] | None,
    refresh_sentiment: bool,
    current_settings: Settings,
) -> None:
    """Background retraining task triggered by the FastAPI endpoint."""
    bridge_state.retraining = True
    try:
        training_result = run_training(
            tickers=tickers,
            current_settings=current_settings,
            refresh_sentiment=refresh_sentiment,
        )
        evaluation_summary = run_evaluation(
            tickers=tickers,
            prepared_data=training_result.bundle,
            model_path=training_result.model_path,
            current_settings=current_settings,
            refresh_sentiment=refresh_sentiment,
        )
        load_model_bundle(current_settings)
        bridge_state.last_retrain = {
            "status": "completed",
            "model_path": str(training_result.model_path),
            "validation_sharpe": training_result.validation_summary["overall_metrics"]["sharpe_ratio"],
            "test_sharpe": evaluation_summary["overall_metrics"]["sharpe_ratio"],
        }
        logger.info("Background retraining completed successfully.")
    except Exception as exc:  # pragma: no cover - runtime endpoint path
        logger.exception("Background retraining failed: %s", exc)
        bridge_state.last_retrain = {"status": "failed", "error": str(exc)}
    finally:
        bridge_state.retraining = False


@app.on_event("startup")
def _startup_load_model() -> None:
    """Attempt to load a saved model when the API process starts."""
    load_model_bundle(settings)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Accept a normalized observation vector and return the PPO action."""
    action, confidence, jev_regime_probs = _predict_action_and_confidence(request.observation)
    bridge_state.last_prediction = {
        "action": action,
        "confidence": confidence,
        "jev_regime_probs": jev_regime_probs,
    }
    return PredictionResponse(action=action, confidence=confidence, jev_regime_probs=jev_regime_probs)


@app.get("/status")
def status() -> dict[str, Any]:
    """Return bridge health, model metadata, and the latest prediction snapshot."""
    return {
        "model_loaded": bridge_state.model is not None,
        "model_path": bridge_state.model_path,
        "expected_observation_size": len(bridge_state.feature_columns),
        "feature_columns": bridge_state.feature_columns,
        "last_prediction": bridge_state.last_prediction,
        "retraining": bridge_state.retraining,
        "last_retrain": bridge_state.last_retrain,
        "jev_version": "1.0",
        "jev_feature_columns": [
            "jev_regime_bullish", "jev_regime_bearish", "jev_regime_choppy", "jev_regime_crisis",
            "jev_trend_strength", "jev_news_bullishness", "jev_portfolio_stress", "jev_halt_noul"
        ],
    }


@app.post("/retrain")
def retrain(request: RetrainRequest | None = None) -> dict[str, Any]:
    """Kick off a background retraining cycle using fresh market data."""
    if bridge_state.retraining:
        raise HTTPException(status_code=409, detail="Retraining is already in progress.")

    payload = request or RetrainRequest()
    worker = threading.Thread(
        target=_retrain_worker,
        kwargs={
            "tickers": payload.tickers,
            "refresh_sentiment": payload.refresh_sentiment,
            "current_settings": settings,
        },
        daemon=True,
    )
    worker.start()

    return {
        "status": "started",
        "tickers": payload.tickers or settings.active_tickers,
        "refresh_sentiment": payload.refresh_sentiment,
    }


def serve(current_settings: Settings = settings) -> None:
    """Run the FastAPI bridge with Uvicorn."""
    uvicorn.run(app, host=current_settings.api_host, port=current_settings.api_port)
