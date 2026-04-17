"""Backtesting and metrics for the RL trading agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from trading_agent.config import Settings, settings
from trading_agent.data.data_fetcher import PreparedDataBundle, prepare_datasets
from trading_agent.environment.trading_env import TradingEnv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


@dataclass
class EpisodeEvaluation:
    """Evaluation output for a single ticker episode."""

    ticker: str
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trade_log: list[dict[str, Any]]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]


def _safe_float(value: Any) -> float:
    """Convert pandas/numpy scalars into JSON-safe Python floats."""
    if value is None:
        return 0.0
    return float(value)


def compute_performance_metrics(
    equity_curve: pd.Series,
    trade_log: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute the headline backtest metrics requested by the project brief.

    Sharpe uses daily returns with a 252-trading-day annualization factor.
    Max drawdown is returned as a positive percentage magnitude.
    """
    equity_curve = equity_curve.astype(float).dropna()
    if equity_curve.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0.0,
            "ending_value": 0.0,
        }

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0 or daily_returns.empty:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252.0)

    running_peak = equity_curve.cummax()
    drawdown_series = equity_curve / running_peak - 1.0
    max_drawdown = abs(drawdown_series.min()) if not drawdown_series.empty else 0.0

    closed_trade_pnls = [
        float(entry.get("pnl", 0.0))
        for entry in trade_log
        if entry.get("side") == "SELL" and "pnl" in entry
    ]
    winning_trades = [pnl for pnl in closed_trade_pnls if pnl > 0]
    losing_trades = [pnl for pnl in closed_trade_pnls if pnl < 0]
    total_trades = len(closed_trade_pnls)
    win_rate = (len(winning_trades) / total_trades) if total_trades else 0.0
    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "cumulative_return": _safe_float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
        "sharpe_ratio": _safe_float(sharpe_ratio),
        "max_drawdown": _safe_float(max_drawdown),
        "win_rate": _safe_float(win_rate),
        "profit_factor": _safe_float(profit_factor),
        "total_trades": _safe_float(total_trades),
        "ending_value": _safe_float(equity_curve.iloc[-1]),
    }


def evaluate_model_on_env(
    model: BaseAlgorithm,
    env: TradingEnv,
    deterministic: bool = True,
) -> EpisodeEvaluation:
    """Run one full episode and collect portfolio curves plus trading metrics."""
    observation, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, _, terminated, truncated, _ = env.step(int(action))

    performance_frame = env.get_performance_frame()
    strategy_curve = performance_frame["portfolio_value"]
    benchmark_curve = performance_frame["benchmark_value"]

    return EpisodeEvaluation(
        ticker=str(env.current_ticker),
        equity_curve=strategy_curve,
        benchmark_curve=benchmark_curve,
        trade_log=list(env.trade_log),
        metrics=compute_performance_metrics(strategy_curve, env.trade_log),
        benchmark_metrics=compute_performance_metrics(benchmark_curve, []),
    )


def _aggregate_curves(curves: list[pd.Series]) -> pd.Series:
    """Average multiple equity curves into one equal-weight aggregate curve."""
    if not curves:
        return pd.Series(dtype="float64")

    combined = pd.concat(curves, axis=1).sort_index().ffill()
    aggregated = combined.mean(axis=1)
    aggregated.name = "aggregate"
    return aggregated


def _plot_cumulative_return_chart(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series,
    output_path: Path,
) -> None:
    """Save the cumulative return comparison chart requested by the brief."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_strategy = strategy_curve / strategy_curve.iloc[0]
    normalized_benchmark = benchmark_curve / benchmark_curve.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(normalized_strategy.index, normalized_strategy.values, label="RL Strategy", linewidth=2.0)
    plt.plot(normalized_benchmark.index, normalized_benchmark.values, label="Buy and Hold", linewidth=2.0)
    plt.title("Cumulative Return Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _json_ready_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so pandas/numpy objects are converted cleanly."""
    return json.loads(json.dumps(summary, default=float))


def evaluate_model_on_frames(
    model: BaseAlgorithm,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    current_settings: Settings = settings,
    split_name: str = "test",
    save_artifacts: bool = False,
) -> dict[str, Any]:
    """Evaluate one model across a full validation or test split."""
    episode_results: list[EpisodeEvaluation] = []

    for ticker, frame in split_frames.items():
        evaluation_env = TradingEnv(
            data_by_ticker={ticker: frame},
            feature_columns=feature_columns,
            initial_cash=current_settings.initial_cash,
            transaction_cost=current_settings.transaction_cost,
            sharpe_window=current_settings.sharpe_window,
            sharpe_reward_weight=current_settings.sharpe_reward_weight,
            random_start=False,
            episode_length=None,
            fixed_ticker=ticker,
            min_episode_length=current_settings.min_episode_length,
            seed=current_settings.random_seed,
        )
        episode_results.append(evaluate_model_on_env(model=model, env=evaluation_env, deterministic=True))

    aggregate_strategy = _aggregate_curves([result.equity_curve for result in episode_results])
    aggregate_benchmark = _aggregate_curves([result.benchmark_curve for result in episode_results])
    all_trade_logs = [trade for result in episode_results for trade in result.trade_log]

    summary: dict[str, Any] = {
        "split": split_name,
        "tickers": [result.ticker for result in episode_results],
        "overall_metrics": compute_performance_metrics(aggregate_strategy, all_trade_logs),
        "benchmark_metrics": compute_performance_metrics(aggregate_benchmark, []),
        "per_ticker": {
            result.ticker: {
                "strategy_metrics": result.metrics,
                "benchmark_metrics": result.benchmark_metrics,
                "trade_count": len(result.trade_log),
            }
            for result in episode_results
        },
    }

    summary = _json_ready_summary(summary)

    if save_artifacts and not aggregate_strategy.empty and not aggregate_benchmark.empty:
        _plot_cumulative_return_chart(
            strategy_curve=aggregate_strategy,
            benchmark_curve=aggregate_benchmark,
            output_path=current_settings.chart_path,
        )

        target_path = (
            current_settings.validation_metrics_path
            if split_name.lower() == "validation"
            else current_settings.evaluation_metrics_path
        )
        target_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _resolve_model_path(model_path: str | Path | None, current_settings: Settings) -> Path:
    """Pick the best available saved model when the caller does not provide one."""
    if model_path:
        return Path(model_path)
    if current_settings.best_model_path.exists():
        return current_settings.best_model_path
    if current_settings.latest_model_path.exists():
        return current_settings.latest_model_path
    raise FileNotFoundError("No trained PPO model was found in the model directory.")


def run_evaluation(
    tickers: list[str] | None = None,
    prepared_data: PreparedDataBundle | None = None,
    model_path: str | Path | None = None,
    current_settings: Settings = settings,
    refresh_sentiment: bool = False,
) -> dict[str, Any]:
    """Load the trained PPO model and run the requested test evaluation."""
    bundle = prepared_data or prepare_datasets(
        tickers=tickers,
        current_settings=current_settings,
        refresh_sentiment=refresh_sentiment,
    )
    resolved_model_path = _resolve_model_path(model_path=model_path, current_settings=current_settings)

    logger.info("Loading model from %s for evaluation.", resolved_model_path)
    model = PPO.load(str(resolved_model_path))

    evaluation_summary = evaluate_model_on_frames(
        model=model,
        split_frames=bundle.test_frames,
        feature_columns=bundle.feature_columns,
        current_settings=current_settings,
        split_name="test",
        save_artifacts=True,
    )
    return evaluation_summary
