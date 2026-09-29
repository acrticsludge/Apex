# ADR-002: Train the online learner on a clone, publish through a registry

- Status: Accepted
- Date: 2026-09-29

## Context

`online_learner._run_update()` ran a PPO fine-tuning loop directly on the live
policy object:

```python
model = _rl._model                       # the object rl_signal infers from
...
model.policy.optimizer.step()            # mutates its parameters in place
```

Meanwhile `rl_signal.get_rl_signal()` ran forward passes against that same
object with no synchronisation. Under gunicorn's `--workers 1 --threads 8`
plus the agent thread, a concurrent inference could read a weight tensor while
`optimizer.step()` was midway through writing it. PyTorch's kernels mutate
storage in place, so this yields torn reads and NaN activations — not a
crash, but silently wrong trading decisions.

The original code held `_update_lock` around the gradient block, which looked
like protection. It was not: the lock serialised *updates*, not updates
against *reads*.

Two smaller races sat alongside it:
- `_is_training` was a check-then-set, so several update threads could start on
  one batch.
- `_new_count += 1` was an unsynchronised read-modify-write shared with the
  `/api/retrain/log` request thread.

## Decision

**Never mutate a model that a reader may be holding. Train a private copy and
swap the weights atomically.**

1. `ModelRegistry` (`trading_agent/integration/model_registry.py`) owns the
   live model behind a re-entrant lock. It exposes `read_lock()` for inference,
   `training_copy()` for the update thread, and `publish()` for the swap.
2. `_run_update()` calls `training_copy()`, which deep-copies the policy **and
   rebuilds its optimiser** against the clone's own parameters. Skipping the
   rebuild leaves the optimiser holding references to the original parameters,
   so `.step()` writes back into the live model.
3. `publish()` copies the finished weights into the live policy's existing
   tensors under the exclusive lock. The policy *object* is never replaced, so a
   reader holding a reference keeps seeing a valid object.
4. `publish()` validates the incoming `state_dict` — key set and per-tensor
   shape — **before** mutating. A stale checkpoint is rejected whole rather
   than half-applied.
5. `_is_training`, `_new_count` and `_total_updates` sit behind `_state_lock`;
   `stats()` returns a consistent snapshot for the dashboard.
6. `rl_signal` takes the registry read lock around the whole forward pass.

The registry deliberately imports no torch, so it is unit-testable with plain
objects and importing it cannot drag the training stack into the serving
process.

## Alternatives considered

**Hold the existing `_update_lock` for the read path too.** Rejected: it
serialises inference behind the full multi-epoch gradient block, adding up to
seconds of latency to every symbol's signal during an update. The registry
exists so the swap is short and the training is long.

**Stop the agent during updates.** Rejected: an update every 16 closed trades
would mean repeated trading halts.

**Snapshot weights, train, then assign `model.policy = new_policy`.** Rejected:
replacing the object strands any reader that already dereferenced it, and a
reader mid-forward-pass would use the old policy for that symbol and the new one
for the next — inconsistent within a single cycle.

**Move training out of process.** Not rejected on merit, just not yet: it adds
a job queue and a serialisation format to solve a problem the in-process
registry already solves. Worth revisiting if update duration grows.

## Consequences

- Inference latency is unaffected; only the weight swap is serialised, and that
  is a few milliseconds.
- Memory roughly doubles transiently during an update (the clone). The policy
  is ~148k parameters, so this is negligible.
- `_clone_model` is the subtle part. If the optimiser is not rebuilt, the fix
  silently does nothing. `test_online_learner_real_ppo.py` asserts the clone's
  optimiser parameters share no identity with the live ones, and
  `test_gradient_step_does_not_touch_the_live_policy` asserts the live
  optimiser is never stepped.
- `publish()` now rejects a shape-mismatched checkpoint instead of writing it.
  That is also what stops a stale artifact from half-loading.
