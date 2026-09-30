"""Tests for the artifact store's baseline/runtime contract.

The distinction under test: an artifact has an immutable baseline that ships in
the image, and a mutable runtime copy that updates land in. Reads prefer the
runtime copy and fall back to the baseline. Writes only ever touch the runtime
copy. The baseline is never written.

The property that matters most in production is the negative one — a save must
never land on a tracked file. These tests assert that directly rather than
trusting the implementation to be careful.
"""
import os
import zipfile
from pathlib import Path

import pytest

from trading_agent.integration import artifact_store as store

BASELINE = store.baseline_dir()


@pytest.fixture(autouse=True)
def _reset_warnings():
    store._warned.clear()
    yield
    store._warned.clear()


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _saving(payload: bytes):
    """A save_fn in the shape SB3 and joblib use: given a path, write to it."""
    def _save(path):
        Path(path).write_bytes(payload)
    return _save


# ── The baseline is real and read-only from this layer's perspective ─────────

def test_the_baseline_model_ships_in_the_repo():
    assert store.baseline_path("best_model.zip").is_file(), (
        "the image must contain a model to start from, or a fresh deploy has "
        "nothing to load"
    )


def test_the_baseline_is_a_readable_zip():
    with zipfile.ZipFile(store.baseline_path("best_model.zip")) as zf:
        assert zf.namelist(), "the baseline model zip is empty"


def test_the_baseline_lives_inside_the_package_not_the_repo_root():
    assert BASELINE.name == "model"
    assert (BASELINE / "best_model.zip").is_file()


# ── resolve(): runtime wins, baseline is the fallback ───────────────────────

def test_resolve_falls_back_to_the_baseline_when_the_runtime_is_empty(tmp_path):
    found = store.resolve("best_model.zip", tmp_path)
    assert found == store.baseline_path("best_model.zip")
    assert found.is_file(), "a volume that has never been written still starts from the model in the image"


def test_resolve_prefers_the_runtime_copy_when_one_exists(tmp_path):
    _touch(tmp_path / "best_model.zip", b"newer")
    found = store.resolve("best_model.zip", tmp_path)
    assert found == tmp_path / "best_model.zip"
    assert found.read_bytes() == b"newer"


def test_resolve_returns_none_when_neither_exists(tmp_path):
    # A name that exists in neither location. "best_model.zip" cannot be used
    # here: the baseline ships in the image, so it is always present, which is
    # exactly the fallback the test above is about.
    assert store.resolve("no_such_artifact.zip", tmp_path) is None, (
        "absent must be distinguishable from empty, so callers can report "
        "'not trained yet' rather than a confusing path"
    )


# ── The property that matters: writes never touch the baseline ──────────────

def test_a_save_does_not_modify_the_tracked_baseline(tmp_path, monkeypatch):
    """The regression this module exists to prevent.

    Saves a model through the store and asserts the git-tracked file is byte
    for byte unchanged afterwards.
    """
    baseline = store.baseline_path("best_model.zip")
    before = baseline.read_bytes()
    baseline_mtime = baseline.stat().st_mtime

    written = store.write_atomic(_saving(b"learned weights"), "best_model.zip", tmp_path)

    assert written == tmp_path / "best_model.zip"
    assert baseline.read_bytes() == before, "the tracked model was modified"
    assert baseline.stat().st_mtime == baseline_mtime, "the tracked model was touched"


def test_a_save_into_a_separate_runtime_leaves_the_worktree_clean(tmp_path):
    """The symptom users report, asserted without ever touching the real model.

    A save into a runtime directory must not make the tracked artifact dirty.
    Verified by comparing bytes and mtime rather than by writing to the tracked
    path: an earlier version of this test did the real thing and left a corrupted
    model behind, which is exactly the failure this module exists to prevent.
    """
    import subprocess

    repo = BASELINE.parents[3]
    if not (Path(repo) / ".git").exists():
        pytest.skip("not a git checkout")

    def _dirty() -> bool:
        r = subprocess.run(
            ["git", "status", "--porcelain", "--", "trading_agent/agent/model"],
            cwd=str(repo), capture_output=True, text=True,
        )
        return bool(r.stdout.strip())

    assert not _dirty(), "the model directory is already dirty before this test"
    store.write_atomic(_saving(b"learned"), "best_model.zip", tmp_path)
    assert not _dirty(), "saving into a runtime volume dirtied the tracked model"


# ── Atomicity ───────────────────────────────────────────────────────────────

def test_a_failed_save_leaves_the_existing_artifact_intact(tmp_path):
    """The important half of atomicity: a crash mid-save must not destroy the
    model that is already there. Degrading to 'no update' is recoverable;
    degrading to 'no model' is not."""
    existing = _touch(tmp_path / "best_model.zip", b"the good model")

    def _explode(path):
        Path(path).write_bytes(b"half written")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        store.write_atomic(_explode, "best_model.zip", tmp_path)

    assert existing.read_bytes() == b"the good model", "a failed save destroyed the model"
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert not leftovers, f"the failed save left temporary files behind: {leftovers}"


def test_a_save_that_produces_nothing_is_an_error_not_a_silent_success(tmp_path):
    def _writes_nothing(path):
        return None

    with pytest.raises(RuntimeError, match="no file"):
        store.write_atomic(_writes_nothing, "best_model.zip", tmp_path)
    assert not (tmp_path / "best_model.zip").exists(), "a no-op save must not count as a model"


def test_a_save_keeps_the_zip_extension(tmp_path):
    """SB3's save() inspects the suffix, so the temporary name has to keep it.
    A temp name of 'best_model.tmp' would make it write best_model.zip.zip or
    fail outright."""
    seen = {}

    def _records(path):
        seen["path"] = Path(path)
        Path(path).write_bytes(b"z")

    store.write_atomic(_records, "best_model.zip", tmp_path)
    assert seen["path"].suffix == ".zip"
    assert ".tmp" in seen["path"].name


def test_a_save_creates_missing_parents(tmp_path):
    # model_dir is the directory, so pass the directory the artifact lives in.
    model_dir = tmp_path / "deep" / "nested"
    written = store.write_atomic(_saving(b"x"), "best_model.zip", model_dir)
    assert written == model_dir / "best_model.zip"
    assert written.is_file()


# ── backup() ─────────────────────────────────────────────────────────────────

def test_backup_keeps_the_current_artifact(tmp_path):
    _touch(tmp_path / "best_model.zip", b"v2")
    keep = store.backup("best_model.zip", tmp_path)
    assert keep is not None
    assert keep.read_bytes() == b"v2"
    assert "pre_online" in keep.name


def test_backup_rotates_rather_than_accumulating(tmp_path):
    """The old code kept every prior version. On a volume that grows forever."""
    for i in range(4):
        _touch(tmp_path / "best_model.zip", f"v{i}".encode())
        store.backup("best_model.zip", tmp_path)
    backups = [p for p in tmp_path.iterdir() if "pre_online" in p.name]
    assert len(backups) == 1, f"expected exactly one retained version, got {backups}"
    assert backups[0].read_bytes() == b"v3"


def test_backup_returns_none_when_there_is_nothing_to_back_up(tmp_path):
    # A name absent from both locations; best_model.zip always resolves to the
    # baseline, so it would be backed up rather than reported as missing.
    assert store.backup("no_such_artifact.zip", tmp_path) is None


def test_backup_falls_back_to_the_baseline_when_the_runtime_is_empty(tmp_path):
    """With a fresh volume the current model is the baseline one, so that is
    what the backup must capture — otherwise the pre-update copy would be
    useless for recovering a bad first update."""
    keep = store.backup("best_model.zip", tmp_path)
    assert keep is not None
    assert keep.read_bytes() == store.baseline_path("best_model.zip").read_bytes()


# ── seed_runtime() ──────────────────────────────────────────────────────────

def test_seeding_fills_an_empty_volume_from_the_baseline(tmp_path):
    copied = store.seed_runtime(tmp_path)
    assert "best_model.zip" in copied
    assert (tmp_path / "best_model.zip").is_file()
    assert (tmp_path / "best_model.zip").read_bytes() == store.baseline_path("best_model.zip").read_bytes()


def test_seeding_never_overwrites_what_the_volume_already_has(tmp_path):
    """Re-seeding would silently undo a deploy's worth of learning."""
    _touch(tmp_path / "best_model.zip", b"learned on the volume")
    copied = store.seed_runtime(tmp_path)
    assert "best_model.zip" not in copied
    assert (tmp_path / "best_model.zip").read_bytes() == b"learned on the volume"


def test_seeding_an_already_seeded_volume_is_a_no_op(tmp_path):
    store.seed_runtime(tmp_path)
    first = (tmp_path / "best_model.zip").read_bytes()
    store.seed_runtime(tmp_path)
    assert (tmp_path / "best_model.zip").read_bytes() == first


# ── The persistence question, and the warning ───────────────────────────────

def test_a_separate_runtime_directory_is_persistent(tmp_path):
    assert store.is_persistent(tmp_path) is True


def test_the_tracked_directory_is_not_persistent():
    assert store.is_persistent(BASELINE) is False, (
        "this is the condition that makes online updates vanish on redeploy"
    )


def test_warn_reports_non_persistence_for_the_tracked_directory(caplog):
    with caplog.at_level("WARNING", logger="trading_agent.integration.artifact_store"):
        assert store.warn_if_not_persistent(BASELINE) is False
    assert any("discarded on redeploy" in r.getMessage() for r in caplog.records)


def test_warn_is_silent_for_a_persistent_location(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="trading_agent.integration.artifact_store"):
        assert store.warn_if_not_persistent(tmp_path) is True
    assert not [r for r in caplog.records if "discarded on redeploy" in r.getMessage()]


def test_the_warning_is_logged_once_not_once_per_call(caplog):
    """A per-cycle warning would flood the log during a halt storm."""
    with caplog.at_level("WARNING", logger="trading_agent.integration.artifact_store"):
        for _ in range(5):
            store.warn_if_not_persistent(BASELINE)
    n = len([r for r in caplog.records if "discarded on redeploy" in r.getMessage()])
    assert n == 1, f"expected one warning, got {n}"


def test_the_warning_names_the_variable_to_set(caplog):
    """An operator hitting this needs to be told the fix, not just the symptom."""
    with caplog.at_level("WARNING", logger="trading_agent.integration.artifact_store"):
        store.warn_if_not_persistent(BASELINE)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "TRADING_AGENT_STORAGE_DIR" in msg
