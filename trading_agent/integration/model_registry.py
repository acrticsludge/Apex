"""Model registry for the RL policy.

The online learner used to call ``optimizer.step()`` on the *live* policy object
while the agent thread ran forward passes against it. Nothing serialised those
two paths, so a concurrent inference could read a weight tensor mid-update.

Ownership model:

* inference  -> ``read_lock()``  (shared; many readers at once)
* training   -> ``training_copy()``  (a private deep copy; no live state touched)
* publishing -> ``publish()``  (exclusive; copies weights into the live policy)

The policy *object* is never replaced, only the weights inside it, so a reader
that already holds a reference to the policy keeps seeing a valid object.

Deliberately free of any torch import so it can be unit-tested with plain
objects, and so importing it cannot drag the training stack into the serving
process.
"""

from __future__ import annotations

import copy
import logging
import threading

logger = logging.getLogger(__name__)


def _shape_of(value):
    """Shape of a weight: torch/numpy expose .shape; sequences expose len."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


class ModelRegistry:
    """Serialises reads of the live policy against weight swaps."""

    def __init__(self, model=None):
        self._model = model
        self._lock = threading.RLock()
        self._closed = False
        self._generation = 0

    # ── Access ────────────────────────────────────────────────────────────────
    @property
    def model(self):
        return self._model

    @property
    def generation(self) -> int:
        """Increments once per published update; useful for cache invalidation."""
        with self._lock:
            return self._generation

    # ── Read path ─────────────────────────────────────────────────────────────
    def read_lock(self) -> threading.RLock:
        """Hold for the duration of a forward pass. Re-entrant."""
        return self._lock

    # ── Write path ────────────────────────────────────────────────────────────
    def training_copy(self):
        """A private deep copy of the policy to run gradients against.

        Taking a copy means a slow or failing update cannot leave the live
        policy half-mutated.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model registered")
            if self._closed:
                raise RuntimeError("Registry is shut down")
            return copy.deepcopy(self._model.policy)

    def publish(self, trained_policy) -> None:
        """Copy trained weights into the live policy, atomically for readers."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Registry is shut down")
            if self._model is None:
                raise RuntimeError("No model registered")
            incoming = trained_policy.state_dict()
            # Validate before mutating: a checkpoint whose feature count or
            # layer sizes disagree with the live policy must be rejected whole,
            # not half-applied. This is the mismatch that silently kills RL
            # inference when a stale artifact is loaded.
            self._assert_compatible(incoming)
            # Replaces the weight tensors under the exclusive lock. Readers
            # either see every old value or every new one.
            self._model.policy.load_state_dict(incoming)
            self._generation += 1
            logger.info("Published online-RL weights (generation %d)", self._generation)

    def _assert_compatible(self, incoming: dict) -> None:
        current = self._model.policy.state_dict()
        missing = set(current) - set(incoming)
        extra = set(incoming) - set(current)
        if missing or extra:
            raise ValueError(
                f"Weight key mismatch: missing={sorted(missing)} unexpected={sorted(extra)}"
            )
        for key, want in current.items():
            got = incoming[key]
            if _shape_of(want) != _shape_of(got):
                raise ValueError(
                    f"Weight shape mismatch for {key!r}: "
                    f"live {_shape_of(want)} vs trained {_shape_of(got)}"
                )

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True


# Process-wide registry. Bound to the model once rl_signal has loaded it.
_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry | None:
    """The registry, or None when no model has been registered yet."""
    return _registry


def bind(model) -> ModelRegistry:
    """Register the live model. Idempotent: rebinding the same object is a no-op."""
    global _registry
    with _registry_lock:
        if _registry is not None and _registry.model is model:
            return _registry
        if _registry is not None:
            _registry.shutdown()
        _registry = ModelRegistry(model)
        logger.info("Bound model registry (generation 0)")
        return _registry


def reset_for_tests() -> None:
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.shutdown()
        _registry = None
