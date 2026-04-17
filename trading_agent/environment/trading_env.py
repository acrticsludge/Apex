"""Custom Gymnasium environment for single-asset RL trading episodes."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Long-only trading environment with benchmark-aware reward shaping.

    The environment samples one ticker per episode and gives the agent a flat
    normalized observation vector. Actions map to:

    - ``0`` -> hold current state
    - ``1`` -> buy if currently flat
    - ``2`` -> sell if currently long
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_by_ticker: dict[str, pd.DataFrame],
        feature_columns: list[str],
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        sharpe_window: int = 30,
        sharpe_reward_weight: float = 0.01,
        benchmark_reward_weight: float = 1.0,
        benchmark_opportunity_cost_weight: float = 0.35,
        flat_position_penalty: float = 0.0002,
        flat_penalty_after_steps: int = 8,
        invalid_action_penalty: float = 0.0005,
        random_start: bool = True,
        episode_length: int | None = 252,
        fixed_ticker: str | None = None,
        min_episode_length: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.data_by_ticker = {
            ticker: frame.copy().sort_index()
            for ticker, frame in data_by_ticker.items()
            if not frame.empty
        }
        if not self.data_by_ticker:
            raise ValueError("TradingEnv requires at least one non-empty ticker dataframe.")

        self.feature_columns = list(feature_columns)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.sharpe_window = int(sharpe_window)
        self.sharpe_reward_weight = float(sharpe_reward_weight)
        self.benchmark_reward_weight = float(benchmark_reward_weight)
        self.benchmark_opportunity_cost_weight = float(benchmark_opportunity_cost_weight)
        self.flat_position_penalty = float(flat_position_penalty)
        self.flat_penalty_after_steps = int(flat_penalty_after_steps)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.random_start = bool(random_start)
        self.episode_length = episode_length
        self.fixed_ticker = fixed_ticker
        self.min_episode_length = int(min_episode_length)
        self._default_seed = seed

        # Test and validation splits can move outside [0, 1] after MinMax scaling.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.current_ticker: str | None = None
        self.episode_frame: pd.DataFrame | None = None
        self.current_step: int = 0
        self.start_step: int = 0
        self.end_step: int = 0
        self.position: int = 0
        self.cash: float = self.initial_cash
        self.shares_held: float = 0.0
        self.entry_price: float = 0.0
        self.trade_log: list[dict[str, Any]] = []
        self.daily_returns: deque[float] = deque(maxlen=self.sharpe_window)
        self.portfolio_dates: list[pd.Timestamp] = []
        self.portfolio_history: list[float] = []
        self.benchmark_history: list[float] = []
        self.benchmark_shares: float = 0.0
        self.action_counts: dict[str, int] = {}
        self.invalid_action_count: int = 0
        self.flat_steps: int = 0
        self.position_steps: int = 0
        self.flat_steps_in_row: int = 0
        self.holding_periods: list[int] = []
        self.last_buy_step: int | None = None

        self._validate_frames()

    def _validate_frames(self) -> None:
        """Fail fast if required model inputs are missing from the dataset."""
        required_columns = set(self.feature_columns + ["close_raw"])
        for ticker, frame in self.data_by_ticker.items():
            missing = required_columns.difference(frame.columns)
            if missing:
                raise ValueError(f"Ticker '{ticker}' is missing required columns: {sorted(missing)}")
            if len(frame) < self.min_episode_length + 1:
                raise ValueError(
                    f"Ticker '{ticker}' needs at least {self.min_episode_length + 1} rows "
                    f"but only has {len(frame)}."
                )

    def _get_observation(self) -> np.ndarray:
        """Return the normalized feature vector for the current timestep."""
        assert self.episode_frame is not None
        row = self.episode_frame.iloc[self.current_step]
        return row[self.feature_columns].to_numpy(dtype=np.float32)

    def _portfolio_value(self, price: float) -> float:
        """Mark the account to market at the supplied price."""
        if self.position == 1:
            return self.shares_held * price
        return self.cash

    def _benchmark_value(self, price: float) -> float:
        """Mark the simple buy-and-hold benchmark to market."""
        return self.benchmark_shares * price

    def _compute_sharpe_bonus(self) -> float:
        """Compute a rolling Sharpe estimate from recent daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0

        returns = np.asarray(self.daily_returns, dtype=np.float32)
        std = float(np.std(returns))
        if std == 0.0:
            return 0.0

        mean = float(np.mean(returns))
        return (mean / std) * np.sqrt(252.0)

    def _reset_episode_state(self, first_price: float, first_date: pd.Timestamp) -> None:
        """Reset account state for a newly sampled episode."""
        self.position = 0
        self.cash = self.initial_cash
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.trade_log = []
        self.daily_returns = deque(maxlen=self.sharpe_window)
        self.portfolio_dates = [first_date]
        self.portfolio_history = [self.initial_cash]
        self.action_counts = {"hold": 0, "buy": 0, "sell": 0}
        self.invalid_action_count = 0
        self.flat_steps = 0
        self.position_steps = 0
        self.flat_steps_in_row = 0
        self.holding_periods = []
        self.last_buy_step = None

        # Benchmark is a simple buy-and-hold position opened at the episode start.
        self.benchmark_shares = self.initial_cash / first_price
        self.benchmark_history = [self.initial_cash]

    def _close_position(self, price: float, date: pd.Timestamp, reason: str) -> float:
        """Close an open position and record both trade PnL and holding period."""
        gross_value = self.shares_held * price
        net_value = gross_value * (1.0 - self.transaction_cost)
        pnl = net_value - (self.shares_held * self.entry_price)
        self.cash = net_value
        self.shares_held = 0.0
        self.position = 0

        if self.last_buy_step is not None:
            holding_period = max(self.current_step - self.last_buy_step, 1)
            self.holding_periods.append(holding_period)
            self.last_buy_step = None

        self.trade_log.append(
            {
                "date": date.isoformat(),
                "ticker": self.current_ticker,
                "side": "SELL",
                "price": price,
                "shares": 0.0,
                "pnl": pnl,
                "reason": reason,
            }
        )
        return pnl

    def get_episode_diagnostics(self) -> dict[str, Any]:
        """Expose agent behavior diagnostics for evaluation and debugging."""
        total_steps = self.flat_steps + self.position_steps
        closed_trades = sum(1 for trade in self.trade_log if trade.get("side") == "SELL")
        avg_holding_period = (
            float(np.mean(self.holding_periods))
            if self.holding_periods
            else 0.0
        )
        return {
            "action_counts": dict(self.action_counts),
            "invalid_action_count": self.invalid_action_count,
            "flat_steps": self.flat_steps,
            "position_steps": self.position_steps,
            "flat_steps_in_row": self.flat_steps_in_row,
            "time_in_market_ratio": (self.position_steps / total_steps) if total_steps else 0.0,
            "closed_trade_count": closed_trades,
            "average_holding_period_days": avg_holding_period,
            "zero_trade_episode": closed_trades == 0,
        }

    def _sample_episode_bounds(self) -> tuple[int, int]:
        """Select a valid [start, end] window for the upcoming episode."""
        assert self.episode_frame is not None
        available_steps = len(self.episode_frame) - 1
        desired_length = min(self.episode_length or available_steps, available_steps)
        desired_length = max(desired_length, self.min_episode_length)

        max_start = max(available_steps - desired_length, 0)
        if self.random_start and max_start > 0:
            start_step = int(self.np_random.integers(0, max_start + 1))
        else:
            start_step = 0

        end_step = min(start_step + desired_length, available_steps)
        return start_step, end_step

    def get_performance_frame(self) -> pd.DataFrame:
        """Expose the portfolio and benchmark curves for evaluation helpers."""
        return pd.DataFrame(
            {
                "portfolio_value": self.portfolio_history,
                "benchmark_value": self.benchmark_history,
            },
            index=pd.DatetimeIndex(self.portfolio_dates),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode from either a random or deterministic point."""
        super().reset(seed=seed if seed is not None else self._default_seed)

        if self.fixed_ticker:
            self.current_ticker = self.fixed_ticker
        else:
            ticker_choices = list(self.data_by_ticker.keys())
            selection_index = int(self.np_random.integers(0, len(ticker_choices)))
            self.current_ticker = ticker_choices[selection_index]

        self.episode_frame = self.data_by_ticker[self.current_ticker]
        self.start_step, self.end_step = self._sample_episode_bounds()
        self.current_step = self.start_step

        first_price = float(self.episode_frame.iloc[self.current_step]["close_raw"])
        first_date = pd.Timestamp(self.episode_frame.index[self.current_step])
        self._reset_episode_state(first_price=first_price, first_date=first_date)

        info = {
            "ticker": self.current_ticker,
            "date": first_date.isoformat(),
            "portfolio_value": self.initial_cash,
            "position": self.position,
        }
        return self._get_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action and advance the episode by a single trading day."""
        assert self.episode_frame is not None

        current_row = self.episode_frame.iloc[self.current_step]
        current_date = pd.Timestamp(self.episode_frame.index[self.current_step])
        current_price = float(current_row["close_raw"])
        previous_portfolio_value = self._portfolio_value(current_price)
        previous_benchmark_value = self._benchmark_value(current_price)
        trade_penalty = 0.0
        invalid_action_penalty = 0.0

        action_name = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1

        if action == 1 and self.position == 0:
            investable_cash = self.cash * (1.0 - self.transaction_cost)
            self.shares_held = investable_cash / current_price
            self.entry_price = current_price
            self.cash = 0.0
            self.position = 1
            self.last_buy_step = self.current_step
            trade_penalty = self.transaction_cost
            self.trade_log.append(
                {
                    "date": current_date.isoformat(),
                    "ticker": self.current_ticker,
                    "side": "BUY",
                    "price": current_price,
                    "shares": self.shares_held,
                    "reason": "agent_action",
                }
            )
        elif action == 1 and self.position == 1:
            self.invalid_action_count += 1
            invalid_action_penalty = self.invalid_action_penalty
        elif action == 2 and self.position == 1:
            self._close_position(price=current_price, date=current_date, reason="agent_action")
            trade_penalty = self.transaction_cost
        elif action == 2 and self.position == 0:
            self.invalid_action_count += 1
            invalid_action_penalty = self.invalid_action_penalty

        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False

        next_row = self.episode_frame.iloc[self.current_step]
        next_date = pd.Timestamp(self.episode_frame.index[self.current_step])
        next_price = float(next_row["close_raw"])

        if terminated and self.position == 1:
            # Force liquidation on the final bar so every episode finishes flat.
            self._close_position(price=next_price, date=next_date, reason="episode_end")

        new_portfolio_value = self._portfolio_value(next_price)
        new_benchmark_value = self._benchmark_value(next_price)
        daily_return = (
            (new_portfolio_value - previous_portfolio_value) / previous_portfolio_value
            if previous_portfolio_value > 0
            else 0.0
        )
        benchmark_return = (
            (new_benchmark_value - previous_benchmark_value) / previous_benchmark_value
            if previous_benchmark_value > 0
            else 0.0
        )
        excess_return = daily_return - benchmark_return
        self.daily_returns.append(daily_return)

        if self.position == 0:
            self.flat_steps += 1
            self.flat_steps_in_row += 1
        else:
            self.position_steps += 1
            self.flat_steps_in_row = 0

        opportunity_cost_penalty = (
            max(benchmark_return, 0.0) * self.benchmark_opportunity_cost_weight
            if self.position == 0
            else 0.0
        )
        flat_position_penalty = (
            self.flat_position_penalty
            if self.position == 0 and self.flat_steps_in_row >= self.flat_penalty_after_steps
            else 0.0
        )
        sharpe_bonus = self._compute_sharpe_bonus()
        reward = float(
            daily_return
            + (self.benchmark_reward_weight * excess_return)
            - trade_penalty
            - invalid_action_penalty
            - opportunity_cost_penalty
            - flat_position_penalty
            + (self.sharpe_reward_weight * sharpe_bonus)
        )

        self.portfolio_dates.append(next_date)
        self.portfolio_history.append(float(new_portfolio_value))
        self.benchmark_history.append(float(new_benchmark_value))

        info = {
            "ticker": self.current_ticker,
            "date": next_date.isoformat(),
            "portfolio_value": float(new_portfolio_value),
            "benchmark_value": float(new_benchmark_value),
            "position": self.position,
            "daily_return": float(daily_return),
            "benchmark_return": float(benchmark_return),
            "excess_return": float(excess_return),
            "trade_count": len(self.trade_log),
            "action_counts": dict(self.action_counts),
            "invalid_action_count": self.invalid_action_count,
            "reward_components": {
                "daily_return": float(daily_return),
                "benchmark_return": float(benchmark_return),
                "excess_return": float(excess_return),
                "trade_penalty": float(trade_penalty),
                "invalid_action_penalty": float(invalid_action_penalty),
                "opportunity_cost_penalty": float(opportunity_cost_penalty),
                "flat_position_penalty": float(flat_position_penalty),
                "sharpe_bonus": float(sharpe_bonus),
            },
            "episode_diagnostics": self.get_episode_diagnostics() if terminated else None,
            "trade_log": self.trade_log if terminated else None,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> dict[str, Any]:
        """Return a compact state snapshot that is easy to print or log."""
        latest_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_cash
        return {
            "ticker": self.current_ticker,
            "portfolio_value": latest_value,
            "position": self.position,
            "trades": len(self.trade_log),
        }
