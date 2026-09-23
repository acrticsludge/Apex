"""PPO training loop and validation-based model selection for the trading agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_agent.agent.evaluate import evaluate_model_on_frames
from trading_agent.config import Settings, settings
from trading_agent.data.data_fetcher import PreparedDataBundle, prepare_datasets
from trading_agent.environment.trading_env import TradingEnv


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Summary object returned after a full PPO training run."""

    model_path: Path
    bundle: PreparedDataBundle
    validation_summary: dict[str, Any]


class ValidationSharpeEvalCallback(EvalCallback):
    """
    EvalCallback variant that chooses the best checkpoint by validation portfolio quality.

    The base Stable-Baselines3 callback saves by mean reward. In trading, the
    shaped reward is useful for learning, but checkpoint selection should stay
    tied to chronological validation behavior: Sharpe, excess return, drawdown,
    and whether the policy actually trades.
    """

    def __init__(
        self,
        validation_frames: dict[str, Any],
        feature_columns: list[str],
        current_settings: Settings,
        eval_freq: int,
        best_model_save_path: Path,
        dataset_metadata: dict[str, Any] | None = None,
        verbose: int = 1,
    ) -> None:
        dummy_eval_env = DummyVecEnv(
            [
                lambda: Monitor(
                    TradingEnv(
                        data_by_ticker=validation_frames,
                        feature_columns=feature_columns,
                        initial_cash=current_settings.initial_cash,
                        transaction_cost=current_settings.transaction_cost,
                        sharpe_window=current_settings.sharpe_window,
                        sharpe_reward_weight=current_settings.sharpe_reward_weight,
                        benchmark_reward_weight=current_settings.benchmark_reward_weight,
                        benchmark_opportunity_cost_weight=current_settings.benchmark_opportunity_cost_weight,
                        flat_position_penalty=current_settings.flat_position_penalty,
                        flat_penalty_after_steps=current_settings.flat_penalty_after_steps,
                        invalid_action_penalty=current_settings.invalid_action_penalty,
                        random_start=False,
                        episode_length=None,
                        min_episode_length=current_settings.min_episode_length,
                        seed=current_settings.random_seed,
                    )
                )
            ]
        )
        super().__init__(
            eval_env=dummy_eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            deterministic=True,
            best_model_save_path=str(best_model_save_path.parent),
            log_path=str(best_model_save_path.parent),
            verbose=verbose,
        )

        self.validation_frames = validation_frames
        self.feature_columns = feature_columns
        self.current_settings = current_settings
        self.best_model_path = best_model_save_path
        self.dataset_metadata = dataset_metadata or {}
        self.best_validation_sharpe = -np.inf
        self.best_validation_score = -np.inf
        self.latest_summary: dict[str, Any] = {}

    def _selection_score(self, summary: dict[str, Any]) -> float:
        """Score validation runs using outperformance, Sharpe, drawdown, and trade coverage."""
        overall_metrics = summary["overall_metrics"]
        benchmark_metrics = summary["benchmark_metrics"]
        diagnostics = summary.get("diagnostics", {})

        excess_return = float(overall_metrics["cumulative_return"]) - float(
            benchmark_metrics["cumulative_return"]
        )
        sharpe_ratio = float(overall_metrics["sharpe_ratio"])
        max_drawdown = float(overall_metrics["max_drawdown"])
        trade_coverage_ratio = float(diagnostics.get("trade_coverage_ratio", 0.0))

        base_score = (
            (self.current_settings.validation_sharpe_weight * sharpe_ratio)
            + (self.current_settings.validation_excess_return_weight * excess_return)
            - (self.current_settings.validation_drawdown_weight * max_drawdown)
            + (self.current_settings.validation_trade_coverage_weight * trade_coverage_ratio)
        )

        # Task 33: Add regime-stratified component
        regime_metrics = summary.get("regime_metrics", {})
        if regime_metrics:
            regime_weights = {
                "bullish": 0.3,
                "bearish": 0.3,
                "choppy": 0.2,
                "crisis": 0.2,
            }
            regime_score = 0.0
            for regime, metrics in regime_metrics.items():
                weight = regime_weights.get(regime, 0.0)
                if weight > 0 and "metrics" in metrics:
                    regime_sharpe = metrics["metrics"].get("sharpe_ratio", 0.0)
                    regime_excess = metrics["metrics"].get("cumulative_return", 0.0) - metrics.get("benchmark_metrics", {}).get("cumulative_return", 0.0)
                    regime_score += weight * (regime_sharpe + regime_excess)
            # Blend: 70% overall, 30% regime-stratified
            return 0.7 * base_score + 0.3 * regime_score
        
        return base_score

    def _on_step(self) -> bool:
        """Run a full validation backtest every ``eval_freq`` environment steps."""
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        summary = evaluate_model_on_frames(
            model=self.model,
            split_frames=self.validation_frames,
            feature_columns=self.feature_columns,
            current_settings=self.current_settings,
            split_name="validation",
            save_artifacts=False,
            dataset_metadata=self.dataset_metadata,
        )

        self.latest_summary = summary
        validation_sharpe = float(summary["overall_metrics"]["sharpe_ratio"])
        validation_return = float(summary["overall_metrics"]["cumulative_return"])
        benchmark_return = float(summary["benchmark_metrics"]["cumulative_return"])
        validation_excess_return = validation_return - benchmark_return
        validation_score = self._selection_score(summary)
        trade_coverage_ratio = float(summary.get("diagnostics", {}).get("trade_coverage_ratio", 0.0))

        self.logger.record("validation/sharpe_ratio", validation_sharpe)
        self.logger.record("validation/cumulative_return", validation_return)
        self.logger.record("validation/excess_return", validation_excess_return)
        self.logger.record("validation/max_drawdown", float(summary["overall_metrics"]["max_drawdown"]))
        self.logger.record("validation/trade_coverage_ratio", trade_coverage_ratio)
        self.logger.record("validation/selection_score", validation_score)

        if validation_score > self.best_validation_score:
            self.best_validation_sharpe = validation_sharpe
            self.best_validation_score = validation_score
            self.model.save(str(self.best_model_path))
            self.current_settings.validation_metrics_path.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

            if self.verbose >= 1:
                logger.info(
                    "New best validation score %.4f (Sharpe %.4f, excess return %.4f). Saved model to %s",
                    validation_score,
                    validation_sharpe,
                    validation_excess_return,
                    self.best_model_path,
                )

        return True


def _make_train_env(bundle: PreparedDataBundle, current_settings: Settings) -> DummyVecEnv:
    """Create the vectorized training environment consumed by SB3 PPO."""
    return DummyVecEnv(
        [
            lambda: Monitor(
                TradingEnv(
                    data_by_ticker=bundle.train_frames,
                    feature_columns=bundle.feature_columns,
                    initial_cash=current_settings.initial_cash,
                    transaction_cost=current_settings.transaction_cost,
                    sharpe_window=current_settings.sharpe_window,
                    sharpe_reward_weight=current_settings.sharpe_reward_weight,
                    benchmark_reward_weight=current_settings.benchmark_reward_weight,
                    benchmark_opportunity_cost_weight=current_settings.benchmark_opportunity_cost_weight,
                    flat_position_penalty=current_settings.flat_position_penalty,
                    flat_penalty_after_steps=current_settings.flat_penalty_after_steps,
                    invalid_action_penalty=current_settings.invalid_action_penalty,
                    random_start=True,
                    episode_length=current_settings.episode_length,
                    min_episode_length=current_settings.min_episode_length,
                    seed=current_settings.random_seed,
                )
            )
        ]
    )


def _build_model(train_env: DummyVecEnv, current_settings: Settings, resume: bool) -> PPO:
    """Create a new PPO model or resume from the latest local checkpoint."""
    if resume and current_settings.latest_model_path.exists():
        logger.info("Resuming PPO training from %s", current_settings.latest_model_path)
        return PPO.load(str(current_settings.latest_model_path), env=train_env, device="auto")

    return PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=current_settings.learning_rate,
        n_steps=current_settings.n_steps,
        batch_size=current_settings.batch_size,
        n_epochs=current_settings.n_epochs,
        gamma=current_settings.gamma,
        gae_lambda=current_settings.gae_lambda,
        policy_kwargs={"net_arch": current_settings.policy_hidden_layers},
        verbose=1,
        tensorboard_log=str(current_settings.tensorboard_dir),
        seed=current_settings.random_seed,
        device="auto",
    )


def run_training(
    tickers: list[str] | None = None,
    prepared_data: PreparedDataBundle | None = None,
    current_settings: Settings = settings,
    refresh_sentiment: bool = False,
    resume: bool = False,
) -> TrainingResult:
    """Train a PPO agent and save both the latest and best-performing checkpoints."""
    current_settings.ensure_directories()
    set_random_seed(current_settings.random_seed)
    np.random.seed(current_settings.random_seed)
    torch.manual_seed(current_settings.random_seed)

    bundle = prepared_data or prepare_datasets(
        tickers=tickers,
        current_settings=current_settings,
        refresh_sentiment=refresh_sentiment,
    )

    train_env = _make_train_env(bundle=bundle, current_settings=current_settings)
    validation_callback = ValidationSharpeEvalCallback(
        validation_frames=bundle.validation_frames,
        feature_columns=bundle.feature_columns,
        current_settings=current_settings,
        eval_freq=current_settings.eval_freq,
        best_model_save_path=current_settings.best_model_path,
        dataset_metadata=bundle.metadata,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=current_settings.checkpoint_freq,
        save_path=str(current_settings.checkpoint_dir),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callback = CallbackList([validation_callback, checkpoint_callback])

    model = _build_model(train_env=train_env, current_settings=current_settings, resume=resume)

    logger.info(
        "Starting PPO training for %s timesteps on tickers: %s",
        current_settings.total_timesteps,
        bundle.tickers,
    )
    model.learn(total_timesteps=current_settings.total_timesteps, callback=callback, progress_bar=False)
    model.save(str(current_settings.latest_model_path))

    best_model_available = current_settings.best_model_path.exists()
    selected_model_path = current_settings.best_model_path if best_model_available else current_settings.latest_model_path

    if not validation_callback.latest_summary:
        validation_callback.latest_summary = evaluate_model_on_frames(
            model=model,
            split_frames=bundle.validation_frames,
            feature_columns=bundle.feature_columns,
            current_settings=current_settings,
            split_name="validation",
            save_artifacts=False,
            dataset_metadata=bundle.metadata,
        )
        current_settings.validation_metrics_path.write_text(
            json.dumps(validation_callback.latest_summary, indent=2),
            encoding="utf-8",
        )
        validation_callback.best_validation_sharpe = float(
            validation_callback.latest_summary["overall_metrics"]["sharpe_ratio"]
        )
        validation_callback.best_validation_score = validation_callback._selection_score(
            validation_callback.latest_summary
        )
        model.save(str(current_settings.best_model_path))
        selected_model_path = current_settings.best_model_path

    training_summary = {
        "model_path": str(selected_model_path),
        "latest_model_path": str(current_settings.latest_model_path),
        "best_model_path": str(current_settings.best_model_path),
        "tickers": bundle.tickers,
        "feature_columns": bundle.feature_columns,
        "total_timesteps": current_settings.total_timesteps,
        "resume_enabled": resume,
        "storage_root": str(current_settings.storage_root),
        "checkpoint_dir": str(current_settings.checkpoint_dir),
        "best_validation_sharpe": float(validation_callback.best_validation_sharpe),
        "best_validation_selection_score": float(validation_callback.best_validation_score),
        "validation_summary": validation_callback.latest_summary,
    }
    current_settings.training_summary_path.write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    train_env.close()
    validation_callback.eval_env.close()
    return TrainingResult(
        model_path=selected_model_path,
        bundle=bundle,
        validation_summary=validation_callback.latest_summary,
    )
