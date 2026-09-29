"""Central configuration for the standalone RL trading agent package."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

# Canonical universe. A leaf module, so this import has no side effects — the
# previous AST scrape of apex_dashboard.py returned {} on any parse failure,
# silently swapping in a drifting hand-maintained copy of the watchlists.
from apex_universe import INDIA_WATCHLIST as CANONICAL_INDIA_WATCHLIST
from apex_universe import US_WATCHLIST as CANONICAL_US_WATCHLIST


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Load both the repo-level .env and an optional package-local .env.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(PACKAGE_ROOT / ".env", override=False)


def _deduplicate(values: Sequence[str]) -> list[str]:
    """Preserve order while removing duplicates from a ticker list."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _resolve_storage_root() -> Path:
    """Pick a writable root for datasets, checkpoints, and trained artifacts."""
    configured = os.getenv("TRADING_AGENT_STORAGE_DIR", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    return PACKAGE_ROOT


def _resolve_default_tickers() -> list[str]:
    """
    Build the default training universe from environment, dashboard, or fallback lists.

    ``RL_TICKERS`` overrides everything. Otherwise ``RL_MARKET_UNIVERSE`` can
    select ``india``, ``us``, or ``all``.
    """
    env_tickers = _deduplicate(os.getenv("RL_TICKERS", "").split(","))
    if env_tickers:
        return env_tickers

    india_watchlist = list(CANONICAL_INDIA_WATCHLIST)
    us_watchlist = list(CANONICAL_US_WATCHLIST)
    market_universe = os.getenv("RL_MARKET_UNIVERSE", "all").strip().lower()

    if market_universe == "india":
        return list(india_watchlist)
    if market_universe == "us":
        return list(us_watchlist)
    return _deduplicate(india_watchlist + us_watchlist)


DEFAULT_ACTIVE_TICKERS = _resolve_default_tickers()


@dataclass
class Settings:
    """Project-wide settings shared by data, training, evaluation, and serving."""

    # Paths
    package_root: Path = PACKAGE_ROOT
    repo_root: Path = REPO_ROOT
    storage_root: Path = field(default_factory=_resolve_storage_root)
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    sentiment_cache_dir: Path = field(init=False)
    tensorboard_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    scaler_path: Path = field(init=False)
    feature_columns_path: Path = field(init=False)
    dataset_metadata_path: Path = field(init=False)
    training_summary_path: Path = field(init=False)
    evaluation_metrics_path: Path = field(init=False)
    validation_metrics_path: Path = field(init=False)
    chart_path: Path = field(init=False)
    latest_model_path: Path = field(init=False)
    best_model_path: Path = field(init=False)

    # Dataset settings
    lookback_years: int = 10
    interval: str = "1d"
    train_split: float = 0.80
    validation_split: float = 0.10
    test_split: float = 0.10
    minimum_rows_after_features: int = 252
    active_tickers: list[str] = field(default_factory=lambda: list(DEFAULT_ACTIVE_TICKERS))
    dashboard_india_watchlist: list[str] = field(
        default_factory=lambda: list(CANONICAL_INDIA_WATCHLIST)
    )
    dashboard_us_watchlist: list[str] = field(
        default_factory=lambda: list(CANONICAL_US_WATCHLIST)
    )

    # Feature settings
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_lower",
            "bb_mid",
            "bb_upper",
            "ema_20",
            "ema_50",
            "atr_14",
            "obv",
            "sentiment",
            # NOTE: jev_trend_strength is deliberately absent. The committed
            # model and scaler were fitted on 29 columns that do not include it,
            # and it is not in feature_columns.json. Listing it here meant a
            # missing columns file produced a 30-column contract, dropna emptied
            # every frame, and RL inference silently returned None forever.
            # The live contract is feature_columns.json; see
            # load_saved_feature_columns, which now fails closed.
        ]
    )

    # Finnhub / news settings
    finnhub_api_key: str = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", "").strip())
    enable_sentiment: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SENTIMENT", "true").strip().lower() != "false"
    )
    news_chunk_days: int = int(os.getenv("FINNHUB_CHUNK_DAYS", "90"))
    finnhub_sleep_seconds: float = float(os.getenv("FINNHUB_SLEEP_SECONDS", "0.4"))
    finnhub_timeout_seconds: int = int(os.getenv("FINNHUB_TIMEOUT_SECONDS", "30"))
    sentiment_forward_fill_limit: int = int(os.getenv("SENTIMENT_FORWARD_FILL_LIMIT", "5"))

    # Environment / reward settings
    initial_cash: float = float(os.getenv("RL_INITIAL_CASH", "10000"))
    transaction_cost: float = 0.001
    episode_length: int = int(os.getenv("RL_EPISODE_LENGTH", "252"))
    min_episode_length: int = int(os.getenv("RL_MIN_EPISODE_LENGTH", "64"))
    sharpe_window: int = 30
    sharpe_reward_weight: float = float(os.getenv("RL_SHARPE_REWARD_WEIGHT", "0.01"))
    benchmark_reward_weight: float = float(os.getenv("RL_BENCHMARK_REWARD_WEIGHT", "1.0"))
    benchmark_opportunity_cost_weight: float = float(
        os.getenv("RL_BENCHMARK_OPPORTUNITY_COST_WEIGHT", "0.35")
    )
    flat_position_penalty: float = float(os.getenv("RL_FLAT_POSITION_PENALTY", "0.0002"))
    flat_penalty_after_steps: int = int(os.getenv("RL_FLAT_PENALTY_AFTER_STEPS", "8"))
    invalid_action_penalty: float = float(os.getenv("RL_INVALID_ACTION_PENALTY", "0.0005"))
    random_seed: int = int(os.getenv("RL_RANDOM_SEED", "42"))

    # Institutional feature toggles
    enable_vwap: bool = field(default_factory=lambda: os.getenv("RL_ENABLE_VWAP", "true").lower() == "true")
    enable_volume_profile: bool = field(
        default_factory=lambda: os.getenv("RL_ENABLE_VOLUME_PROFILE", "true").lower() == "true"
    )
    enable_liquidity_sweeps: bool = field(
        default_factory=lambda: os.getenv("RL_ENABLE_LIQUIDITY_SWEEPS", "true").lower() == "true"
    )
    vwap_lookback: int = field(default_factory=lambda: int(os.getenv("RL_VWAP_LOOKBACK", "30")))
    vp_lookback: int = field(default_factory=lambda: int(os.getenv("RL_VP_LOOKBACK", "20")))
    swing_lookback: int = field(default_factory=lambda: int(os.getenv("RL_SWING_LOOKBACK", "20")))

    # PPO hyperparameters required by the project brief
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = int(os.getenv("RL_TOTAL_TIMESTEPS", "500000"))
    eval_freq: int = int(os.getenv("RL_EVAL_FREQ", "10000"))
    checkpoint_freq: int = int(os.getenv("RL_CHECKPOINT_FREQ", "25000"))
    policy_hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
    validation_sharpe_weight: float = float(os.getenv("RL_VALIDATION_SHARPE_WEIGHT", "1.0"))
    validation_excess_return_weight: float = float(os.getenv("RL_VALIDATION_EXCESS_RETURN_WEIGHT", "2.0"))
    validation_drawdown_weight: float = float(os.getenv("RL_VALIDATION_DRAWDOWN_WEIGHT", "0.75"))
    validation_trade_coverage_weight: float = float(os.getenv("RL_VALIDATION_TRADE_COVERAGE_WEIGHT", "0.25"))

    # Serving settings
    api_host: str = os.getenv("TRADING_AGENT_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("TRADING_AGENT_PORT", "8000"))

    def __post_init__(self) -> None:
        """Resolve storage-dependent paths after initialization."""
        self.storage_root = Path(self.storage_root).expanduser()
        if not self.storage_root.is_absolute():
            self.storage_root = (self.repo_root / self.storage_root).resolve()

        self.data_dir = self.storage_root / "data"
        self.model_dir = self.storage_root / "agent" / "model"
        self.sentiment_cache_dir = self.data_dir / "cache" / "sentiment"
        self.tensorboard_dir = self.model_dir / "tensorboard"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.scaler_path = self.model_dir / "scaler.joblib"
        self.feature_columns_path = self.model_dir / "feature_columns.json"
        self.dataset_metadata_path = self.model_dir / "dataset_metadata.json"
        self.training_summary_path = self.model_dir / "training_summary.json"
        self.evaluation_metrics_path = self.model_dir / "evaluation_metrics.json"
        self.validation_metrics_path = self.model_dir / "validation_metrics.json"
        self.chart_path = self.model_dir / "cumulative_return_comparison.png"
        self.latest_model_path = self.model_dir / "latest_model.zip"
        self.best_model_path = self.model_dir / "best_model.zip"

        if self.enable_vwap:
            self.feature_columns += ["vwap_dist", "vwap_above", "vwap_band_2_dist"]
        if self.enable_volume_profile:
            self.feature_columns += ["vp_poc_dist", "vp_in_value_area", "vp_above_poc"]
        if self.enable_liquidity_sweeps:
            self.feature_columns += [
                "ls_pdh_dist", "ls_pdl_dist",
                "ls_swing_high_dist", "ls_swing_low_dist",
                "ls_sweep", "ls_choch",
            ]

    def ensure_directories(self) -> None:
        """Create all writable directories used by the pipeline."""
        for directory in (
            self.data_dir,
            self.model_dir,
            self.sentiment_cache_dir,
            self.tensorboard_dir,
            self.checkpoint_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolved_tickers(self, explicit_tickers: Sequence[str] | None = None) -> list[str]:
        """Return the effective training universe for the current run."""
        if explicit_tickers:
            return _deduplicate(explicit_tickers)
        return list(self.active_tickers)

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings to JSON-friendly primitives for metadata files."""
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def save_runtime_snapshot(self) -> None:
        """Persist the resolved configuration used for the current run."""
        self.ensure_directories()
        snapshot_path = self.model_dir / "runtime_config.json"
        snapshot_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


settings = Settings()
settings.ensure_directories()
