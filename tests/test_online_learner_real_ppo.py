"""The online-learner fix, against a real PPO policy and real torch.

tests/test_online_learner_runtime.py proves the same properties against a
torch-shaped fake, which works everywhere. This file runs the real thing, so
the deepcopy, the optimizer rebuild and the weight swap are exercised on the
actual nn.Module machinery the bot will use in production.
"""
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PPO = pytest.importorskip("stable_baselines3").PPO

import trading_agent.integration.online_learner as ol  # noqa: E402
import trading_agent.integration.rl_signal as rl  # noqa: E402
from trading_agent.integration.model_registry import get_registry, reset_for_tests  # noqa: E402


OBS_DIM = 12
ACTIONS = 3


@pytest.fixture
def real_ppo():
    """A small but genuine PPO with an attached optimiser, plus a real scaler."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium import spaces

    def _env():
        import gymnasium as gym

        class _E(gym.Env):
            observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
            action_space = spaces.Discrete(ACTIONS)

            def reset(self, seed=None, options=None):
                return np.zeros(OBS_DIM, np.float32), {}

            def step(self, action):
                return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

        return _E()

    model = PPO("MlpPolicy", DummyVecEnv([_env]), n_steps=16, batch_size=8, verbose=0)
    before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    return model, before


@pytest.fixture(autouse=True)
def clean_learner():
    ol._buffer.clear()
    ol._pending.clear()
    ol._retrain_log.clear()
    ol._new_count = 0
    ol._is_training = False
    ol._total_updates = 0
    yield
    reset_for_tests()
    ol._buffer.clear()
    ol._pending.clear()
    ol._new_count = 0
    ol._is_training = False


def _seed(n=16):
    for i in range(n):
        ol._buffer.append({
            "obs": np.random.randn(OBS_DIM).astype(np.float32),
            "action": i % ACTIONS,
            "reward": float(np.random.randn()),
            "regime": ["bullish", "bearish", "choppy", "crisis"][i % 4],
        })


def _live_weights(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def test_the_model_is_actually_trained_by_the_update(real_ppo, monkeypatch, tmp_path, accepting_updates):
    """Sanity: the harness exercises real gradients, not a no-op."""
    model, _ = real_ppo
    monkeypatch.setattr(rl, "_model", model, raising=False)
    monkeypatch.setattr(ol, "get_rl_signal", None, raising=False)
    import trading_agent.config as cfgmod
    monkeypatch.setattr(cfgmod.settings, "best_model_path", tmp_path / "m.zip")
    monkeypatch.setattr(cfgmod.settings, "model_dir", tmp_path, raising=False)

    from trading_agent.integration.model_registry import bind
    bind(model)
    before = _live_weights(model)
    _seed()
    ol._run_update()
    after = _live_weights(model)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "the update did not change a single weight — nothing was trained"


def test_gradient_step_does_not_touch_the_live_policy(real_ppo, monkeypatch, tmp_path, accepting_updates):
    """The whole point: the live policy is only ever written by publish()."""
    model, _ = real_ppo
    monkeypatch.setattr(rl, "_model", model, raising=False)
    import trading_agent.config as cfgmod
    monkeypatch.setattr(cfgmod.settings, "best_model_path", tmp_path / "m.zip")
    monkeypatch.setattr(cfgmod.settings, "model_dir", tmp_path, raising=False)
    from trading_agent.integration.model_registry import bind
    reg = bind(model)

    live_optimizer = model.policy.optimizer
    before = _live_weights(model)
    _seed()
    ol._run_update()

    assert reg.generation == 1, "no publish happened"
    after = _live_weights(model)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "publish did not move any weights — the update trained nothing"
    # The live optimiser's own state must be untouched: it was never stepped.
    assert live_optimizer is model.policy.optimizer, "the live optimiser was replaced"


def test_clone_owns_its_parameters(real_ppo):
    """deepcopy of an nn.Module shares nothing: the fix depends on that."""
    model, _ = real_ppo
    from trading_agent.integration.model_registry import ModelRegistry

    reg = ModelRegistry(model)
    try:
        clone = reg.training_copy()
        live = {id(p) for p in model.policy.parameters()}
        cloned = {id(p) for p in clone.parameters()}
        assert not (live & cloned), "clone shares parameter tensors with the live policy"

        # And the rebuilt optimiser must point at the clone, not the original.
        owned = {id(p) for group in clone.optimizer.param_groups for p in group["params"]}
        assert not (owned & live), "the clone's optimiser still references live parameters"
    finally:
        reg.shutdown()


def test_concurrent_inference_never_sees_a_partial_swap(real_ppo):
    """A reader must observe one whole generation, never a mix of two.

    The model is primed to a uniform state *before* the reader starts, so every
    coherent read sees a tensor whose elements are all the same generation
    marker. A torn read shows a tensor with mixed values, which is exactly what
    the pre-fix code permitted.
    """
    model, _ = real_ppo
    from trading_agent.integration.model_registry import ModelRegistry

    reg = ModelRegistry(model)
    with torch.no_grad():                    # prime: uniform generation 0
        for p in model.policy.parameters():
            p.fill_(0.0)

    mixed = []
    seen = set()
    stop = threading.Event()

    def infer():
        while not stop.is_set():
            with reg.read_lock():
                for p in model.policy.parameters():
                    v = p.detach()
                    lo, hi = float(v.min()), float(v.max())
                    if abs(hi - lo) > 1e-9:     # a half-written tensor
                        mixed.append((lo, hi))
                    else:
                        seen.add(lo)

    t = threading.Thread(target=infer, daemon=True)
    t.start()
    try:
        for i in range(1, 61):
            clone = reg.training_copy()
            with torch.no_grad():
                for p in clone.parameters():
                    p.fill_(float(i))
            reg.publish(clone)                 # the real swap path
    finally:
        stop.set()
        t.join(timeout=5)
        reg.shutdown()

    assert not mixed, (
        f"inference observed a half-swapped weight tensor {len(mixed)} time(s): "
        f"{mixed[:3]}"
    )
    assert len(seen) > 1, (
        f"the reader only ever saw {seen}; the test never exercised a swap"
    )


def test_failed_update_leaves_the_live_policy_byte_identical(real_ppo, monkeypatch, tmp_path):
    model, _ = real_ppo
    monkeypatch.setattr(rl, "_model", model, raising=False)
    import trading_agent.config as cfgmod
    monkeypatch.setattr(cfgmod.settings, "best_model_path", tmp_path / "m.zip")
    monkeypatch.setattr(cfgmod.settings, "model_dir", tmp_path, raising=False)
    from trading_agent.integration.model_registry import bind
    reg = bind(model)

    before = _live_weights(model)
    gen = reg.generation

    def boom(_m, _r):
        raise RuntimeError("clone failed")
    monkeypatch.setattr(ol, "_make_training_model", boom)

    _seed()
    ol._run_update()

    after = _live_weights(model)
    for k in before:
        assert torch.equal(before[k], after[k]), f"{k} was corrupted by a failed update"
    assert reg.generation == gen
    assert ol._is_training is False
    assert any(e["event"] == "failed" for e in ol.get_retrain_log())


def test_committed_artifacts_load_under_the_pinned_stack():
    """The torch 2.8 -> 2.14 jump across a pickled PPO artifact is only safe if
    the artifact still loads and still infers."""
    from trading_agent.config import settings
    import joblib

    model = PPO.load(str(settings.best_model_path))
    scaler = joblib.load(str(settings.scaler_path))
    n = model.observation_space.shape[0]
    assert scaler.n_features_in_ == n, (
        f"scaler expects {scaler.n_features_in_}, model takes {n}"
    )
    obs = np.zeros((1, n), dtype=np.float32)
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        probs = model.policy.get_distribution(tensor).distribution.probs
    assert torch.isfinite(probs).all(), "inference on the committed model is not finite"
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
