"""Google Colab helper for resumable RL training on the default watchlists."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from trading_agent.agent.evaluate import run_evaluation
from trading_agent.agent.train import run_training
from trading_agent.config import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def in_colab() -> bool:
    """Detect whether the current Python process is running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # pragma: no cover

        return True
    except ImportError:
        return False


def drive_is_mounted(mount_path: str = "/content/drive") -> bool:
    """Detect whether Google Drive is already mounted and accessible."""
    return (Path(mount_path) / "MyDrive").exists()


def maybe_mount_drive(mount_path: str = "/content/drive") -> bool:
    """Mount Google Drive when running inside Colab."""
    if not in_colab():
        return False

    if drive_is_mounted(mount_path):
        return True

    # Child Python processes launched from a notebook do not necessarily have a
    # live IPython kernel, so calling drive.mount() there can fail even though
    # the parent notebook already mounted Drive.
    try:
        from IPython import get_ipython  # type: ignore  # pragma: no cover

        shell = get_ipython()
        if shell is None or getattr(shell, "kernel", None) is None:
            logging.getLogger(__name__).info(
                "Colab detected but no interactive kernel is available; skipping drive.mount()."
            )
            return False
    except Exception:
        logging.getLogger(__name__).info("Could not inspect IPython kernel; skipping drive.mount().")
        return False

    from google.colab import drive  # type: ignore  # pragma: no cover

    drive.mount(mount_path, force_remount=False)
    return drive_is_mounted(mount_path)


def default_storage_root(use_drive: bool, drive_subdir: str) -> Path:
    """Choose a persistent storage directory for Colab artifacts."""
    if use_drive and in_colab():
        if maybe_mount_drive():
            return Path("/content/drive/MyDrive") / drive_subdir
        logging.getLogger(__name__).warning(
            "Google Drive is unavailable in this process. Falling back to temporary /content storage."
        )
    return Path("/content") / drive_subdir if in_colab() else Path("trading_agent_colab_runtime")


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI wrapper for common Colab training flows."""
    parser = argparse.ArgumentParser(description="Colab launcher for the RL trading agent")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override.")
    parser.add_argument("--timesteps", type=int, default=int(os.getenv("RL_TOTAL_TIMESTEPS", "500000")), help="Total PPO timesteps.")
    parser.add_argument("--refresh-sentiment", action="store_true", help="Refresh Finnhub sentiment cache.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest saved checkpoint.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Train only and skip test evaluation.")
    parser.add_argument("--no-drive", action="store_true", help="Do not mount Google Drive.")
    parser.add_argument(
        "--drive-subdir",
        default="trading_agent_runtime",
        help="Folder name under MyDrive used for persistent artifacts.",
    )
    return parser


def main() -> None:
    """Run a Colab-friendly training and evaluation workflow."""
    args = build_argument_parser().parse_args()
    storage_root = default_storage_root(use_drive=not args.no_drive, drive_subdir=args.drive_subdir)
    runtime_settings = Settings(storage_root=storage_root, total_timesteps=args.timesteps)

    training_result = run_training(
        tickers=args.tickers,
        current_settings=runtime_settings,
        refresh_sentiment=args.refresh_sentiment,
        resume=args.resume,
    )

    payload: dict[str, object] = {
        "status": "training_complete",
        "storage_root": str(runtime_settings.storage_root),
        "tickers": args.tickers or runtime_settings.active_tickers,
        "model_path": str(training_result.model_path),
        "validation_summary": training_result.validation_summary,
    }

    if not args.skip_evaluate:
        payload["evaluation_summary"] = run_evaluation(
            tickers=args.tickers,
            prepared_data=training_result.bundle,
            model_path=training_result.model_path,
            current_settings=runtime_settings,
            refresh_sentiment=args.refresh_sentiment,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
