"""Where RL artifacts live, and which of those locations actually survives a deploy.

The problem this exists to solve
--------------------------------
``trading_agent/agent/model/`` is git-tracked, because the trained model has to
be in the image for the app to start. That made it the default write target for
the online learner, which has two consequences:

* every online update dirties the working tree, so a trained artifact and a
  model that happened to drift at 3am are indistinguishable in ``git status``;
* on a platform with an ephemeral filesystem — Railway, most containers — a
  redeploy silently discards every learned update, and the process comes back up
  on whatever weights were committed weeks ago, with no indication that anything
  was lost.

The fix is to stop treating those as one path. An artifact has a *baseline*, the
immutable copy that ships in the image, and a *runtime* copy that updates land
in. Reads prefer the runtime copy and fall back to the baseline. Writes only
ever touch the runtime copy. The baseline is never written, so a fresh checkout
is always a known-good starting point.

Behaviour is unchanged until ``TRADING_AGENT_STORAGE_DIR`` points somewhere
writable and separate. With it unset, the runtime directory *is* the tracked
directory, so this module logs that online learning is not persistent rather
than pretending otherwise. That keeps local development and CI behaving exactly
as they do now, and makes the gap visible instead of leaving it to be
discovered on a deploy.

Why a volume rather than object storage
---------------------------------------
A Railway volume is the least novel failure mode available: the read path is
already a plain file open, and the alternative — fetching a model from object
storage on every process start — adds a network dependency to the one operation
that must succeed for the RL agent to run at all. That is a deliberate
trade of cost against a new way to fail at boot.

Atomicity
---------
Writes go to a temporary file in the destination directory and are then renamed
over the target. ``Path.rename`` is atomic within a filesystem, so a crash or a
kill mid-save leaves either the old model or the new one, never a half-written
zip. This matters more on a volume than it did in a checkout, because a volume
outlives the process that corrupted it.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# The immutable baseline that ships in the image. Resolved relative to this file
# so it is correct regardless of the working directory.
_BASELINE_DIR = Path(__file__).resolve().parent.parent / "agent" / "model"

# Artifacts the runtime layer owns. The model is the one that mutates in
# production; the scaler and feature contract are written by training and read
# by inference, and both are pinned by the feature-contract check.
_ARTIFACT_NAMES = (
    "best_model.zip",
    "scaler.joblib",
    "feature_columns.json",
)

_warned: set[str] = set()
_warn_lock = threading.Lock()


def baseline_dir() -> Path:
    """The git-tracked, read-only artifact directory."""
    return _BASELINE_DIR


def baseline_path(name: str) -> Path:
    return _BASELINE_DIR / name


def runtime_dir(model_dir: Path) -> Path:
    """Where updates are written, for a given configured model directory."""
    return Path(model_dir)


def is_persistent(model_dir: Path) -> bool:
    """True when the runtime directory is genuinely separate from the baseline.

    This is the whole question. If they are the same directory, every online
    update is a working-tree modification and a redeploy throws it away.
    """
    try:
        return Path(model_dir).resolve() != _BASELINE_DIR.resolve()
    except OSError:
        return False


def resolve(name: str, model_dir: Path) -> Path | None:
    """The path to read ``name`` from: runtime if present, else baseline.

    Returns ``None`` when neither exists, so callers can distinguish "not
    trained yet" from "trained and here".
    """
    runtime = runtime_dir(model_dir) / name
    if runtime.is_file():
        return runtime
    base = baseline_path(name)
    if base.is_file():
        return base
    return None


def runtime_path(name: str, model_dir: Path) -> Path:
    """The path to write ``name`` to. Always the runtime location."""
    return runtime_dir(model_dir) / name


def write_atomic(save_fn, name: str, model_dir: Path) -> Path:
    """Persist an artifact via ``save_fn(tmp_path)``, then rename it into place.

    ``save_fn`` is given a ``Path`` with a ``.tmp`` suffix. SB3's ``save`` and
    joblib's ``dump`` both accept a path and both append their own extension if
    one is missing, so the caller must be given a suffix that will not confuse
    them — the temporary name keeps ``.zip`` for models, which is what SB3
    expects to find.

    On failure the temporary file is removed and the existing artifact is left
    untouched, so a failed save degrades to "no update" rather than "no model".
    """
    target = runtime_path(name, model_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    suffix = target.suffix
    tmp = target.with_name(f"{target.stem}.tmp{suffix}")
    try:
        save_fn(tmp)
        if not tmp.is_file():
            raise RuntimeError(f"save produced no file at {tmp}")
        # Same-directory rename: atomic on POSIX and on Windows for this case,
        # because the destination is an existing file being replaced.
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def backup(name: str, model_dir: Path) -> Path | None:
    """Copy the current artifact aside as ``<stem>_pre_online<suffix>``.

    Keeps the single most recent prior version, which is what recovering from a
    bad online update actually needs. Older copies are not retained: this used
    to accumulate one per update, and on a persistent volume that grows without
    bound.
    """
    current = resolve(name, model_dir)
    if current is None:
        return None
    target = runtime_path(name, model_dir)
    keep = target.with_name(f"{target.stem}_pre_online{target.suffix}")
    keep.parent.mkdir(parents=True, exist_ok=True)
    try:
        if keep.exists():
            keep.unlink()
        shutil.copy2(current, keep)
        return keep
    except OSError as exc:
        # A failed backup must not block the update; losing the old copy is
        # preferable to losing the new one.
        logger.warning("Could not write pre-update backup for %s: %s", name, exc)
        return None


def seed_runtime(model_dir: Path) -> list[str]:
    """Copy baseline artifacts into an empty runtime directory.

    Called once when a persistent volume is first used, so the mounted volume
    starts from the committed model rather than from nothing. Existing files are
    never overwritten: the volume is the source of truth once it has content,
    and re-seeding would silently undo a deploy's worth of learning.

    Returns the names copied, so a caller can log what happened.
    """
    copied: list[str] = []
    rt = runtime_dir(model_dir)
    for name in _ARTIFACT_NAMES:
        src = baseline_path(name)
        dst = rt / name
        if not src.is_file() or dst.exists():
            continue
        try:
            rt.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(name)
        except OSError as exc:
            logger.warning("Could not seed %s from baseline: %s", name, exc)
    if copied:
        logger.info("Seeded runtime artifacts from baseline: %s", ", ".join(copied))
    return copied


def warn_if_not_persistent(model_dir: Path) -> bool:
    """Log once per process that online updates will not survive a redeploy.

    Returns True when the location *is* persistent, so a caller can assert on it
    in a startup log line rather than leaving the state implicit.
    """
    ok = is_persistent(model_dir)
    if not ok:
        with _warn_lock:
            first = str(model_dir) not in _warned
            _warned.add(str(model_dir))
        if first:
            logger.warning(
                "Online RL updates are writing to %s, which is the git-tracked "
                "artifact directory. They will be discarded on redeploy. Set "
                "TRADING_AGENT_STORAGE_DIR to a mounted volume to make them "
                "persistent.",
                model_dir,
            )
    return ok
