"""End-to-end checks for model persistence across a simulated redeploy.

test_artifact_store.py covers the store in isolation. These tests drive the real
call sites and assert the property that motivated the change: a model written to
a volume is still there after the process restarts, and the tracked baseline in
the image is never the thing being written.

A deploy is simulated by pointing settings at a fresh directory and re-running
the load, which is what a redeploy does — new process, same mounted disk,
baseline read from the image.
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

import trading_agent.config as cfgmod
import trading_agent.integration.online_learner as ol
import trading_agent.integration.rl_signal as rl
from trading_agent.integration import artifact_store as store

BASELINE = store.baseline_dir()
REPO = BASELINE.parents[3]
OBS_DIM, ACTIONS = 12, 3


@pytest.fixture
def ppo():
    """A small but genuine PPO, so the save path is SB3's real one."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium import spaces
    import gymnasium as gym

    def _env():
        class _E(gym.Env):
            observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
            action_space = spaces.Discrete(ACTIONS)

            def reset(self, seed=None, options=None):
                return np.zeros(OBS_DIM, np.float32), {}

            def step(self, action):
                return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

        return _E()

    return PPO("MlpPolicy", DummyVecEnv([_env]), n_steps=16, batch_size=8, verbose=0)


@pytest.fixture
def volume(tmp_path):
    """A simulated Railway volume: a directory that outlives the process."""
    return tmp_path / "data"


@pytest.fixture
def clean_learner():
    ol._buffer.clear()
    ol._pending.clear()
    ol._retrain_log.clear()
    ol._new_count = 0
    ol._is_training = False
    ol._total_updates = 0
    yield
    from trading_agent.integration.model_registry import reset_for_tests
    reset_for_tests()
    ol._buffer.clear()
    ol._pending.clear()
    ol._new_count = 0


def _seed(n=16):
    for i in range(n):
        ol._buffer.append({
            "obs": np.random.randn(OBS_DIM).astype(np.float32),
            "action": i % ACTIONS,
            "reward": float(np.random.randn()),
            "regime": ["bullish", "bearish", "choppy", "crisis"][i % 4],
        })


def _model_dir(volume: Path) -> Path:
    """Mirror how Settings composes model_dir from a storage root."""
    return volume / "agent" / "model"


def _worktree_dirty() -> bool:
    r = subprocess.run(
        ["git", "status", "--porcelain", "--", "trading_agent/agent/model"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


# ── The real save path targets the volume, never the repo ──────────────────

def test_a_real_update_lands_in_the_volume_not_the_repo(
    ppo, volume, clean_learner, monkeypatch
):
    """The headline assertion, driven through online_learner._run_update.

    Uses the genuine PPO save, so this covers SB3's own write behaviour rather
    than a stand-in for it.
    """
    model_dir = _model_dir(volume)
    monkeypatch.setattr(rl, "_model", ppo, raising=False)
    monkeypatch.setattr(cfgmod.settings, "model_dir", model_dir, raising=False)
    monkeypatch.setattr(cfgmod.settings, "best_model_path",
                        model_dir / "best_model.zip", raising=False)
    from trading_agent.integration.model_registry import bind
    bind(ppo)

    baseline_before = store.baseline_path("best_model.zip").read_bytes()
    assert not _worktree_dirty(), "the model directory was already dirty"

    _seed()
    ol._run_update()

    written = model_dir / "best_model.zip"
    assert written.is_file(), f"nothing reached the volume; found {list(model_dir.glob('*'))}"
    assert store.baseline_path("best_model.zip").read_bytes() == baseline_before, (
        "the online update modified the tracked model"
    )
    assert not _worktree_dirty(), "the online update dirtied the worktree"


def test_a_real_update_is_readable_as_a_model(ppo, volume, clean_learner, monkeypatch):
    """The saved artifact is a loadable model, not just bytes at the right path."""
    from stable_baselines3 import PPO

    model_dir = _model_dir(volume)
    monkeypatch.setattr(rl, "_model", ppo, raising=False)
    monkeypatch.setattr(cfgmod.settings, "model_dir", model_dir, raising=False)
    monkeypatch.setattr(cfgmod.settings, "best_model_path",
                        model_dir / "best_model.zip", raising=False)
    from trading_agent.integration.model_registry import bind
    bind(ppo)

    _seed()
    ol._run_update()

    reloaded = PPO.load(str(model_dir / "best_model.zip"))
    assert reloaded.observation_space.shape[0] == OBS_DIM
    assert reloaded.action_space.n == ACTIONS


def test_an_update_backs_up_the_previous_model_once(
    ppo, volume, clean_learner, monkeypatch
):
    """One retained version, not one per update — the old code accumulated."""
    model_dir = _model_dir(volume)
    monkeypatch.setattr(rl, "_model", ppo, raising=False)
    monkeypatch.setattr(cfgmod.settings, "model_dir", model_dir, raising=False)
    monkeypatch.setattr(cfgmod.settings, "best_model_path",
                        model_dir / "best_model.zip", raising=False)
    from trading_agent.integration.model_registry import bind
    bind(ppo)

    for _ in range(3):
        _seed()
        ol._run_update()

    backups = [p for p in model_dir.iterdir() if "pre_online" in p.name]
    assert len(backups) == 1, f"expected one retained backup, got {backups}"


def test_a_failed_save_does_not_destroy_the_deployed_model(
    ppo, volume, clean_learner, monkeypatch
):
    """A crash mid-update must leave a working model behind. On a volume that
    outlives the process, a corrupted artifact would break every future boot."""
    model_dir = _model_dir(volume)
    monkeypatch.setattr(rl, "_model", ppo, raising=False)
    monkeypatch.setattr(cfgmod.settings, "model_dir", model_dir, raising=False)
    monkeypatch.setattr(cfgmod.settings, "best_model_path",
                        model_dir / "best_model.zip", raising=False)
    from trading_agent.integration.model_registry import bind
    bind(ppo)

    _seed()
    ol._run_update()
    good = (model_dir / "best_model.zip").read_bytes()

    original = ppo.save

    def _half_written(path):
        Path(path).write_bytes(b"corrupt")
        raise RuntimeError("disk full during save")

    monkeypatch.setattr(ppo, "save", _half_written)
    _seed()
    ol._run_update()   # _run_update swallows the failure by design

    assert (model_dir / "best_model.zip").read_bytes() == good, (
        "a failed save replaced the working model"
    )
    assert not [p for p in model_dir.iterdir() if ".tmp" in p.name], (
        "a failed save left a temporary file behind"
    )
    monkeypatch.setattr(ppo, "save", original)


# ── The deploy survives a restart ───────────────────────────────────────────

def test_a_model_written_to_the_volume_is_found_after_a_restart(volume):
    model_dir = _model_dir(volume)
    store.write_atomic(lambda p: Path(p).write_bytes(b"the updated weights"),
                       "best_model.zip", model_dir)

    # "Restart": a fresh lookup, exactly as a new process would do.
    found = store.resolve("best_model.zip", _model_dir(volume))
    assert found == model_dir / "best_model.zip"
    assert found.read_bytes() == b"the updated weights"


def test_a_fresh_volume_starts_from_the_image_model(volume):
    """Nothing on the volume means the committed model is used. Otherwise a new
    deploy would have no model at all."""
    model_dir = _model_dir(volume)
    copied = store.seed_runtime(model_dir)
    assert "best_model.zip" in copied
    found = store.resolve("best_model.zip", _model_dir(volume))
    assert found == model_dir / "best_model.zip"
    assert found.read_bytes() == store.baseline_path("best_model.zip").read_bytes()


def test_repeated_restarts_never_regress_to_the_baseline(volume):
    model_dir = _model_dir(volume)
    store.seed_runtime(model_dir)
    store.write_atomic(lambda p: Path(p).write_bytes(b"generation 1"),
                       "best_model.zip", model_dir)
    for _ in range(3):
        found = store.resolve("best_model.zip", _model_dir(volume))
        assert found.read_bytes() == b"generation 1", "a restart lost the learned model"
    assert store.is_persistent(model_dir) is True


def test_a_volume_is_seeded_once_and_then_left_alone(volume):
    """Re-seeding on every boot would silently undo a deploy's worth of
    learning."""
    model_dir = _model_dir(volume)
    store.seed_runtime(model_dir)
    store.write_atomic(lambda p: Path(p).write_bytes(b"learned"),
                       "best_model.zip", model_dir)
    assert store.seed_runtime(model_dir) == []
    assert (model_dir / "best_model.zip").read_bytes() == b"learned"


def test_without_a_volume_behaviour_is_unchanged_and_flagged(caplog):
    """Deploys that have not mounted a volume keep today's behaviour, and say so
    rather than failing quietly.

    The store only warns once per process per directory, so the guard is reset
    here. Without that this assertion depends on test ordering, and fails in a
    full run where another test already consumed the warning.
    """
    store._warned.clear()
    with caplog.at_level("WARNING", logger="trading_agent.integration.artifact_store"):
        persistent = store.warn_if_not_persistent(BASELINE)
    assert persistent is False
    assert "TRADING_AGENT_STORAGE_DIR" in caplog.text
    store._warned.clear()


# ── Mismatched artifact pairs stay detectable ───────────────────────────────

def test_a_volume_holding_only_a_model_resolves_the_contract_from_the_baseline(volume):
    """The risk of resolving per artifact is a model from one place and a
    feature contract from another.

    A volume holding only the model is reachable — a partial write, a manual
    copy — and in that state the two resolve to different directories. The
    feature-count check in rl_signal is what rejects the pair; this test pins
    that the split is visible rather than silently wrong.
    """
    model_dir = _model_dir(volume)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_model.zip").write_bytes(b"runtime model")

    model = store.resolve("best_model.zip", model_dir)
    contract = store.resolve("feature_columns.json", model_dir)
    assert model.parent != contract.parent, (
        "if these ever come from the same directory the partial-write case is "
        "no longer reachable and this test should be revisited"
    )
