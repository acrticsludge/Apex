"""Model registry: atomically swap fine-tuned weights into the live policy.

The online learner ran optimizer.step() directly on the object the agent thread
was inferring from. Under gunicorn's 8 request threads plus the agent thread, a
concurrent forward pass could read a half-updated weight tensor.

The update thread now trains a private copy and publishes the result under a
single write lock; inference takes the read lock for the length of a forward
pass. Torch is not required for the registry itself, so these tests use fakes.
"""
import threading
import time

import pytest


class FakePolicy:
    """Minimal stand-in for a torch policy: named float weights."""

    def __init__(self, weights=None):
        self.weights = list(weights or [0.0] * 8)

    def state_dict(self):
        return {"weights": list(self.weights)}

    def load_state_dict(self, sd):
        self.weights = list(sd["weights"])

    def read(self):
        return tuple(self.weights)


class FakeModel:
    def __init__(self, weights=None):
        self.policy = FakePolicy(weights)


@pytest.fixture
def registry():
    from trading_agent.integration.model_registry import ModelRegistry

    reg = ModelRegistry(FakeModel([0.0] * 8))
    yield reg
    reg.shutdown()


def test_inference_sees_a_consistent_weight_snapshot(registry):
    """A forward pass must not observe a partially updated weight vector.

    All mutation goes through publish(); a writer that reached past the registry
    would not be covered, which is exactly why the swap is centralised here.
    """
    torn = []
    stop = threading.Event()

    def infer():
        while not stop.is_set():
            with registry.read_lock():
                w = registry.model.policy.read()
            if len(set(w)) != 1:      # all-equal means a coherent snapshot
                torn.append(w)

    t = threading.Thread(target=infer, daemon=True)
    t.start()
    for i in range(300):
        registry.publish(FakePolicy([float(i)] * 8))
    stop.set()
    t.join(timeout=5)

    assert not torn, f"inference observed a torn weight vector: {torn[:3]}"


def test_publish_waits_for_an_in_flight_reader(registry):
    """A swap must not land while a forward pass is mid-flight."""
    order = []
    reading = threading.Event()
    may_finish = threading.Event()

    def slow_reader():
        with registry.read_lock():
            reading.set()
            may_finish.wait(5)
            order.append("reader-done")

    r = threading.Thread(target=slow_reader, daemon=True)
    r.start()
    assert reading.wait(5)

    def writer():
        registry.publish(FakePolicy([4.0] * 8))
        order.append("publish-done")

    w = threading.Thread(target=writer, daemon=True)
    w.start()
    time.sleep(0.1)
    order.append("while-reading")
    may_finish.set()
    r.join(timeout=5)
    w.join(timeout=5)

    assert order.index("reader-done") < order.index("publish-done"), (
        f"publish ran concurrently with a forward pass: {order}"
    )
    assert registry.model.policy.read() == (4.0,) * 8


def test_publish_does_not_swap_the_policy_object_itself(registry):
    """Inference may hold a reference to the policy; replacing the object would
    strand it. Only the weights inside may change."""
    before = registry.model.policy
    copy = registry.training_copy()
    copy.weights = [3.0] * 8
    registry.publish(copy)
    assert registry.model.policy is before, "policy object was replaced, not its weights"


def test_training_copy_does_not_alias_the_live_policy(registry):
    copy = registry.training_copy()
    copy.weights = [9.0] * 8
    assert registry.model.policy.read() == (0.0,) * 8, (
        "the training copy shares storage with the live policy"
    )


def test_publish_swaps_weights_under_the_write_lock(registry):
    copy = registry.training_copy()
    copy.weights = [7.0] * 8
    registry.publish(copy)
    assert registry.model.policy.read() == (7.0,) * 8


def test_read_lock_excludes_a_concurrent_reader(registry):
    """The registry uses one re-entrant lock, not a reader-writer lock: inference
    and weight swaps are serialised. Swaps are rare, so exclusivity costs
    nothing and removes a whole class of races."""
    order = []
    inside = threading.Event()
    may_leave = threading.Event()

    def holder():
        with registry.read_lock():
            inside.set()
            may_leave.wait(5)
            order.append("first-done")

    def contender():
        with registry.read_lock():
            order.append("second-done")

    a = threading.Thread(target=holder, daemon=True)
    a.start()
    assert inside.wait(5)
    b = threading.Thread(target=contender, daemon=True)
    b.start()
    time.sleep(0.1)
    may_leave.set()
    a.join(timeout=5)
    b.join(timeout=5)

    assert order == ["first-done", "second-done"], f"lock did not serialise: {order}"


def test_generation_counter_advances_only_on_publish(registry):
    before = registry.generation
    copy = registry.training_copy()
    copy.weights = [1.0] * 8
    assert registry.generation == before, "generation moved before a publish"
    registry.publish(copy)
    assert registry.generation == before + 1


def test_publish_ignores_a_mismatched_state(registry):
    """A shape mismatch must not corrupt the live policy."""
    with pytest.raises(Exception):
        registry.publish(FakePolicy([0.0] * 3))
    assert registry.model.policy.read() == (0.0,) * 8, "live policy was damaged"


def test_registry_refuses_updates_after_shutdown(registry):
    registry.shutdown()
    with pytest.raises(RuntimeError):
        registry.publish(FakePolicy([0.0] * 8))
