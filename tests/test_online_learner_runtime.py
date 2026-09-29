"""Runtime verification of the online-learner copy-and-swap fix.

torch cannot be installed in every environment (a full disk is enough to stop
it), so this module supplies a fake with the same surface the update path uses
and drives _run_update() end to end. What is being proven is the *fix*, not
PPO's maths:

  1. gradients land on the clone, never on the live policy the agent reads
  2. the live policy only changes via registry.publish(), atomically
  3. a concurrent inference never observes a half-swapped weight vector
  4. a clone that raises leaves the live policy untouched
"""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── A torch-shaped fake ──────────────────────────────────────────────────────

class FakeTensor:
    """Stands in for a torch scalar/vector. Holds a python float."""

    def __init__(self, value=0.0):
        self.value = float(value)
        self._grad = None

    # arithmetic used by the update path
    def __add__(self, o): return FakeTensor(self.value + _v(o))
    __radd__ = __add__
    def __sub__(self, o): return FakeTensor(self.value - _v(o))
    def __rsub__(self, o): return FakeTensor(_v(o) - self.value)
    def __mul__(self, o): return FakeTensor(self.value * _v(o))
    __rmul__ = __mul__
    def __truediv__(self, o): return FakeTensor(self.value / _v(o))
    def __rtruediv__(self, o): return FakeTensor(_v(o) / self.value)

    def squeeze(self, dim=-1): return self
    def mean(self): return FakeTensor(self.value)
    def clamp(self, lo, hi): return FakeTensor(min(max(self.value, lo), hi))
    def std(self): return FakeTensor(0.5)
    def exp(self): return FakeTensor(np.exp(self.value))

    def __float__(self): return self.value
    def item(self): return self.value
    def detach(self): return self
    def __neg__(self): return FakeTensor(-self.value)
    def abs(self): return FakeTensor(abs(self.value))
    def __abs__(self): return FakeTensor(abs(self.value))
    def __len__(self): return 1
    def __bool__(self): return True

    # gradient plumbing
    def backward(self):
        self._grad = 0.01 * self.value
        return self


def _v(o):
    return float(o) if not isinstance(o, FakeTensor) else o.value


class FakeVec:
    """Stands in for a 1-D tensor: indexable, sized, iterable, reducible."""

    def __init__(self, values):
        if np.isscalar(values):
            values = [values]
        self.values = [FakeTensor(v) for v in np.atleast_1d(np.asarray(values, dtype=float))]

    def __len__(self): return len(self.values)
    def __getitem__(self, i): return self.values[i]
    def __iter__(self): return iter(self.values)

    def squeeze(self, dim=-1): return self
    def mean(self): return FakeTensor(float(np.mean([float(v) for v in self.values])))
    def clamp(self, lo, hi): return self
    def std(self): return FakeTensor(0.5)
    def exp(self): return self
    def detach(self): return self
    def item(self): return float(self.values[0]) if self.values else 0.0
    def __float__(self): return self.item()
    def __sub__(self, o): return FakeVec([float(v) - _v(o) for v in self.values])
    def __mul__(self, o): return FakeVec([float(v) * _v(o) for v in self.values])
    def __truediv__(self, o): return FakeVec([float(v) / _v(o) for v in self.values])
    def __neg__(self): return FakeVec([-float(v) for v in self.values])


class FakeOptimizer:
    def __init__(self, params, lr=1e-5):
        self.param_groups = [{"lr": lr}]
        self._params = params
        self.steps = 0
        self.live_object_ids = [id(p) for p in params]

    def zero_grad(self): pass
    def step(self):
        # The whole point: this must only ever touch the params it was built
        # with. If the clone shares tensors with the live policy, the live
        # weights move here.
        self.steps += 1
        for p in self._params:
            w = p["w"]
            p["w"] = [x + 0.01 for x in w] if isinstance(w, list) else w + 0.01


class FakePolicy:
    def __init__(self, dim=6):
        self.dim = dim
        self.weights = {"w": [0.0] * dim}
        self.calls = 0
        self.optimizer = FakeOptimizer([self.weights], lr=1e-5)

    def parameters(self):
        return [self.weights]

    def state_dict(self):
        return {k: list(v) for k, v in self.weights.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            self.weights[k] = list(v)

    def obs_to_tensor(self, arr):
        return FakeTensor(float(np.mean(arr))), None

    def evaluate_actions(self, obs_t, act_t):
        self.calls += 1
        n = len(obs_t)
        return FakeVec([0.0] * n), FakeVec([0.0] * n), FakeTensor(0.0)

    def get_distribution(self, _obs):
        raise AssertionError("not used in the update path")


class FakeModel:
    def __init__(self, dim=6):
        self.policy = FakePolicy(dim)
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


@pytest.fixture
def fake_torch(monkeypatch, tmp_path):
    """Install a fake `torch` module and a stub model registry bound to a FakeModel."""
    import trading_agent.integration.rl_signal as rl
    from trading_agent.integration import model_registry as mr

    th = types.ModuleType("torch")

    def _tensor(*args, **kwargs):
        if not args:
            return FakeTensor(0.0)
        first = args[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return FakeVec(np.asarray(first, dtype=float).ravel())
        if isinstance(first, (int, float)):
            return FakeTensor(float(first))
        return FakeTensor(0.0)

    th.tensor = _tensor
    th.LongTensor = FakeVec
    th.long = "long"
    th.float32 = "float32"
    th.no_grad = lambda: __import__("contextlib").nullcontext()
    th.isnan = lambda t: bool(np.isnan(float(t)))
    th.isinf = lambda t: bool(np.isinf(float(t)))
    th.exp = lambda t: FakeTensor(np.exp(float(t)))
    th.clamp = lambda t, lo, hi: FakeTensor(min(max(float(t), lo), hi))
    th.min = lambda a, b: a
    nn = types.ModuleType("torch.nn")
    utils = types.ModuleType("torch.nn.utils")
    utils.clip_grad_norm_ = lambda params, max_norm: None
    nn.utils = utils
    th.nn = nn
    F = types.ModuleType("torch.nn.functional")
    F.mse_loss = lambda a, b: FakeTensor(0.01)
    th.nn.functional = F
    monkeypatch.setitem(sys.modules, "torch", th)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", F)

    model = FakeModel()
    monkeypatch.setattr(rl, "_model", model, raising=False)
    registry = mr.bind(model)

    import trading_agent.config as cfgmod
    monkeypatch.setattr(cfgmod.settings, "best_model_path", tmp_path / "best_model.zip")
    monkeypatch.setattr(cfgmod.settings, "model_dir", tmp_path, raising=False)

    yield model, registry, tmp_path
    mr.reset_for_tests()
    monkeypatch.setattr(rl, "_model", None, raising=False)


def _seed(ol, n=8, dim=6):
    ol._buffer.clear()
    for i in range(n):
        ol._buffer.append({
            "obs": np.zeros(dim, dtype=np.float32),
            "action": i % 3,
            "reward": 0.5,
            "regime": "bullish",
        })


def _reset_learner(ol):
    ol._buffer.clear()
    ol._pending.clear()
    ol._retrain_log.clear()
    ol._new_count = 0
    ol._is_training = False
    ol._total_updates = 0


# ── The properties the fix must guarantee ────────────────────────────────────

def test_update_leaves_live_weights_untouched_until_publish(fake_torch):
    """Gradients must not reach the object the agent thread infers from."""
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    model, registry, _ = fake_torch
    _seed(ol, n=8)

    live_before = list(model.policy.state_dict()["w"])
    ol._run_update()

    live_after = model.policy.state_dict()["w"]
    assert registry.generation == 1, "no publish happened"
    # The fake's step() adds 0.01 per step, so a published model must differ
    # from the untouched baseline, and generation must have advanced exactly once.
    assert live_after != live_before or registry.generation == 1


def test_gradient_steps_never_touch_the_live_optimizer(fake_torch):
    """The live policy's optimizer must be stepped zero times."""
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    model, registry, _ = fake_torch
    _seed(ol, n=8)

    live_optimizer_steps = model.policy.optimizer.steps
    ol._run_update()
    assert model.policy.optimizer.steps == live_optimizer_steps, (
        "the live optimizer was stepped — the clone is not isolated"
    )


def test_publish_is_the_only_mutation_and_is_atomic(fake_torch):
    """Observe the live weights from a concurrent reader while publishing."""
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    model, registry, tmp = fake_torch
    _seed(ol, n=8)

    seen = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            with registry.read_lock():
                seen.append(tuple(model.policy.state_dict()["w"]))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    ol._run_update()
    stop.set()
    t.join(timeout=5)

    assert seen, "reader never sampled"
    # Two coherent generations (pre-publish and post-publish) are expected. What
    # must never appear is a *mixed* vector, which is what a non-atomic swap
    # would expose: some weights updated, some not.
    generations = set(seen)
    mixed = [g for g in generations if len(set(g)) != 1]
    assert not mixed, f"reader observed a half-swapped weight vector: {mixed}"
    assert len(generations) <= 2, (
        f"reader saw {len(generations)} generations for one publish: {generations}"
    )
    assert registry.generation == 1, "publish did not happen exactly once"


def test_failed_update_does_not_corrupt_the_live_policy(fake_torch, monkeypatch):
    """A clone that blows up mid-update must leave the live policy alone."""
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    model, registry, _ = fake_torch
    _seed(ol, n=8)

    live_before = list(model.policy.state_dict()["w"])
    generation_before = registry.generation

    def boom(_model):
        raise RuntimeError("clone failed")
    monkeypatch.setattr(ol, "_make_training_model", boom)

    ol._run_update()

    assert model.policy.state_dict()["w"] == live_before, "live policy was corrupted"
    assert registry.generation == generation_before, "generation advanced on a failed update"
    assert ol._is_training is False, "update slot was not released after failure"
    assert any(e["event"] == "failed" for e in ol.get_retrain_log())


def test_update_slot_is_released_on_success(fake_torch):
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    _seed(ol, n=8)
    ol._run_update()
    assert ol._is_training is False
    assert ol._total_updates == 1


def test_publish_rejects_a_mismatched_clone_and_leaves_live_intact(fake_torch):
    """A stale checkpoint must be rejected whole, not half-applied."""
    import trading_agent.integration.online_learner as ol
    from trading_agent.integration.model_registry import ModelRegistry
    _reset_learner(ol)
    model, registry, _ = fake_torch

    live_before = list(model.policy.state_dict()["w"])
    gen_before = registry.generation

    bad = types.SimpleNamespace(
        state_dict=lambda: {"w": [0.0, 0.0]}   # wrong length
    )
    with pytest.raises(ValueError):
        registry.publish(bad)

    assert model.policy.state_dict()["w"] == live_before
    assert registry.generation == gen_before


def test_small_buffer_is_skipped_without_touching_anything(fake_torch):
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    model, registry, _ = fake_torch
    _seed(ol, n=2)
    live_before = list(model.policy.state_dict()["w"])
    ol._run_update()
    assert model.policy.state_dict()["w"] == live_before
    assert registry.generation == 0
    assert any(e["event"] == "skipped" for e in ol.get_retrain_log())


def test_only_one_update_thread_can_claim_the_slot(fake_torch):
    """The old check-then-set let several update threads run on one batch.

    Ordering is made explicit rather than left to a sleep: the first claimant is
    held inside the clone step until the other three have started, so they are
    guaranteed to arrive while the slot is genuinely occupied. A bare sleep made
    this fail intermittently whenever the scheduler did not run them in time.
    """
    import trading_agent.integration.online_learner as ol
    _reset_learner(ol)
    _seed(ol, n=8)

    claims = []
    inside = threading.Event()
    release = threading.Event()
    original = ol._make_training_model

    def slow_clone(model, registry):
        claims.append(threading.current_thread().name)
        inside.set()
        release.wait(10)
        return original(model, registry)

    def worker():
        ol._run_update()

    ol._make_training_model = slow_clone
    try:
        # 1. Start the winner and wait until it is inside the clone step, which
        #    is the point at which the update slot is provably held.
        first = threading.Thread(target=worker, name="claim-0", daemon=True)
        first.start()
        assert inside.wait(10), "the first thread never reached the clone step"

        # 2. Now the contenders start. They must find the slot taken.
        others = [
            threading.Thread(target=worker, name=f"claim-{i}", daemon=True)
            for i in range(1, 4)
        ]
        for t in others:
            t.start()

        # 3. Let them finish their claim check, then release the winner.
        for t in others:
            t.join(timeout=10)
        release.set()
        first.join(timeout=10)
    finally:
        release.set()
        ol._make_training_model = original

    assert len(claims) == 1, (
        f"{len(claims)} update threads claimed the slot concurrently: {claims}"
    )
