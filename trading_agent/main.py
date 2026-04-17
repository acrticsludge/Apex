"""Command-line entry point for training, evaluation, and serving."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from trading_agent.agent.evaluate import run_evaluation
from trading_agent.agent.train import TrainingResult, run_training
from trading_agent.config import Settings
from trading_agent.integration.bot_bridge import serve


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI used for local runs and Google Colab notebooks."""
    parser = argparse.ArgumentParser(description="Standalone RL trading agent")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "serve", "full"],
        default="train",
        help="`full` runs train -> evaluate -> serve in one session.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional ticker override, for example: --tickers AAPL MSFT NVDA",
    )
    parser.add_argument(
        "--refresh-sentiment",
        action="store_true",
        help="Force a fresh Finnhub sentiment download instead of using the cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from `latest_model.zip` if one exists.",
    )
    parser.add_argument(
        "--storage-dir",
        help="Optional artifact root for datasets, checkpoints, and saved models.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Optional override for PPO total timesteps.",
    )
    return parser


def main() -> None:
    """Execute the requested workflow."""
    args = build_argument_parser().parse_args()
    settings_kwargs = {}
    if args.storage_dir:
        settings_kwargs["storage_root"] = Path(args.storage_dir)
    if args.timesteps is not None:
        settings_kwargs["total_timesteps"] = args.timesteps
    runtime_settings = Settings(**settings_kwargs)

    training_result: TrainingResult | None = None
    evaluation_summary: dict | None = None

    if args.mode in {"train", "full"}:
        training_result = run_training(
            tickers=args.tickers,
            current_settings=runtime_settings,
            refresh_sentiment=args.refresh_sentiment,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "status": "training_complete",
                    "model_path": str(training_result.model_path),
                    "validation_summary": training_result.validation_summary,
                },
                indent=2,
            )
        )

    if args.mode in {"evaluate", "full"}:
        evaluation_summary = run_evaluation(
            tickers=args.tickers,
            prepared_data=training_result.bundle if training_result else None,
            model_path=training_result.model_path if training_result else None,
            current_settings=runtime_settings,
            refresh_sentiment=args.refresh_sentiment,
        )
        print(json.dumps({"status": "evaluation_complete", "summary": evaluation_summary}, indent=2))

    if args.mode in {"serve", "full"}:
        serve(runtime_settings)


if __name__ == "__main__":
    main()
