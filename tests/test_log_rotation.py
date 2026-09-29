"""Regression tests for bounded, rotated log files.

The price updater ticks every 5 s and each cycle logs per held position, roughly
17k lines/day into a plain FileHandler with no rotation on a Railway volume.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_file_handler_rotates(dash):
    handlers = [h for h in dash.apex_log.handlers if isinstance(h, logging.FileHandler)]
    assert handlers, "no file handler on apex_log"
    for h in handlers:
        assert isinstance(h, RotatingFileHandler), (
            f"{type(h).__name__} is unbounded — use RotatingFileHandler"
        )
        assert h.maxBytes > 0, "maxBytes must bound the file"
        assert h.backupCount >= 1, "need at least one rotated file"


def test_rotation_limits_are_sane(dash):
    handlers = [h for h in dash.apex_log.handlers if isinstance(h, RotatingFileHandler)]
    h = handlers[0]
    assert h.maxBytes >= 1_000_000, "log budget below 1 MB is too small for this tick rate"
    assert 1 <= h.backupCount <= 10


def test_ring_buffer_is_still_bounded(dash):
    """The in-memory buffer backs /api/logs; it must not grow without bound."""
    assert dash._log_buffer.maxlen is not None
    assert dash._log_buffer.maxlen <= 10_000
