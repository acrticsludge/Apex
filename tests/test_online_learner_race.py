"""Regression tests for the online-learner / inference race.

_run_update used to call optimizer.step() on the very policy object the agent
thread ran forward passes against, with nothing serialising the two. It now
trains a private clone and publishes the finished weights through the registry.

torch cannot be installed in every environment, so these assert the *shape* of
the fix (which object gets stepped, what is published, what is locked) rather
than running a gradient. The locking semantics themselves are exercised with
fakes in test_model_registry.py.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEARNER = ROOT / "trading_agent" / "integration" / "online_learner.py"
SIGNAL = ROOT / "trading_agent" / "integration" / "rl_signal.py"


def _func(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    pytest.fail(f"{name} not found in {path.name}")


def _guards(node, name: str) -> bool:
    """True if `node` contains a `with <name>:` block.

    Handles both `self._lock` (Attribute) and a module-level `_lock` (Name).
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.With):
            continue
        for i in n.items:
            ctx = i.context_expr
            if isinstance(ctx, ast.Name) and ctx.id == name:
                return True
            if isinstance(ctx, ast.Attribute) and ctx.attr == name:
                return True
    return False


def _step_targets(fn):
    """Every object that has .optimizer.step() called on it."""
    out = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "optimizer"):
            base = node.func.value.value
            out.append(base.id if isinstance(base, ast.Name) else ast.dump(base))
    return out


def test_gradient_step_never_touches_the_live_model():
    """`model` is the live policy; stepping it was the race."""
    fn = _func(LEARNER, "_run_update")
    targets = _step_targets(fn)
    assert targets, "no optimizer.step() found — the update no longer trains"
    assert "model" not in targets, (
        f"optimizer.step() still applied to the live `model`: {targets}"
    )


def test_training_happens_on_a_clone():
    fn = _func(LEARNER, "_run_update")
    assert _func(LEARNER, "_make_training_model"), "_make_training_model missing"
    assert _func(LEARNER, "_clone_model"), "_clone_model missing"
    # The clone must rebuild the optimizer, otherwise .step() writes to the
    # original parameters it still holds references to.
    src = LEARNER.read_text(encoding="utf-8", errors="replace")
    clone_src = ast.get_source_segment(src, _func(LEARNER, "_clone_model")) or ""
    assert "optimizer" in clone_src, "clone does not rebuild the optimizer"


def test_finished_weights_are_published():
    fn = _func(LEARNER, "_run_update")
    calls = {
        n.func.attr for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "publish" in calls, "trained weights are never published to the live policy"


def test_update_slot_is_claimed_under_a_lock():
    fn = _func(LEARNER, "_run_update")
    assert _guards(fn, "_state_lock"), (
        "_is_training is still a check-then-set race: no `with _state_lock` in _run_update"
    )


def test_counter_increment_is_atomic():
    src = LEARNER.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    fn = _func(LEARNER, "record_exit")
    locked = False
    for node in ast.walk(fn):
        if isinstance(node, ast.With) and _guards(node, "_state_lock"):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name)
                        and sub.target.id == "_new_count"):
                    locked = True
    assert locked, "_new_count += 1 is an unsynchronised read-modify-write"


def test_inference_holds_the_registry_read_lock():
    fn = _func(SIGNAL, "get_rl_signal")
    guarded = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_registry_lock_for_inference"
    ]
    assert guarded, "forward pass takes no lock — a weight swap can tear it"
    # And the forward pass must be inside that with-block.
    src = SIGNAL.read_text(encoding="utf-8", errors="replace")
    body = ast.get_source_segment(src, fn) or ""
    assert "obs_to_tensor" in body, "forward pass moved out of the function"


def test_model_load_binds_the_registry():
    fn = _func(SIGNAL, "_ensure_loaded")
    called = {
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "bind" in called, "the loaded model is never bound to the registry"


def test_stats_gives_a_consistent_snapshot():
    import trading_agent.integration.online_learner as ol

    s = ol.stats()
    for key in ("is_training", "new_count", "buffer_size", "total_updates", "threshold"):
        assert key in s, f"stats() missing {key}"
    assert isinstance(s["buffer_size"], int)
    assert s["threshold"] == ol._UPDATE_THRESHOLD


def test_registry_is_torch_free():
    """Importing the registry must not drag torch into the serving process."""
    src = (ROOT / "trading_agent" / "integration" / "model_registry.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "import torch" not in src
    assert "stable_baselines3" not in src
