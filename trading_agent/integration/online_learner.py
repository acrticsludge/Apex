"""Fully-automatic online PPO fine-tuning from real closed-trade experience.

After every _UPDATE_THRESHOLD closed trades the module fires a background thread
that runs a small PPO gradient update directly on the loaded model weights, then
saves the updated model to disk.  No retraining loop or environment needed.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
_BUFFER_MAXLEN    = 64     # rolling window kept in memory
_UPDATE_THRESHOLD = 16     # fire update after this many new closed trades
_FINE_TUNE_LR     = 1e-5   # 30× slower than original 3e-4 — prevents forgetting
_CLIP_EPS         = 0.10   # tighter clip than training 0.2 for stability
_N_EPOCHS         = 3      # gradient passes per update batch

# ── Validation gate ──────────────────────────────────────────────────────────
# An update used to be published whenever the training loss fell, which only
# proves the policy fitted the batch harder. It is now held back unless the
# candidate beats the incumbent on experiences neither of them trained on.
#
# The holdout is the most recent tail of the buffer, never a random sample: over
# a time series a random split trains on the future and validates on the past,
# which looks like progress and is not.
_VAL_FRACTION     = 0.20   # share of the batch reserved for scoring
_MIN_TRAIN        = 4      # below this the batch cannot support both sides
_MIN_VALIDATION   = 2      # minimum scoring slice for the comparison to mean anything
_MIN_BATCH_FOR_VALIDATION = 12   # smallest total batch worth splitting

# How much better the candidate must be, in percent, before it is published.
# Zero means "reject anything that is not an improvement". Raise it to stop
# noise-level gains from random-walking the policy away from a good starting
# point; the measured values are logged either way so it can be tuned from data.
_MIN_VALIDATION_IMPROVEMENT_PCT = 0.0

# ── State ─────────────────────────────────────────────────────────────────────
_buffer:      deque[dict] = deque(maxlen=_BUFFER_MAXLEN)
_pending:     dict[str, dict] = {}          # symbol → trade-open context
_retrain_log: deque[dict]     = deque(maxlen=500)
_update_lock  = threading.Lock()
# Guards the plain-int counters below. They are read by the Flask request thread
# (/api/retrain/log) and written by the agent thread, so the increments need to
# be atomic read-modify-writes.
_state_lock    = threading.Lock()
_new_count    = 0      # experiences accumulated since last update trigger
_is_training  = False
_total_updates = 0     # cumulative applied updates
_total_rejected = 0    # cumulative updates held back by the validation gate


def _split_train_validation(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a batch into a training portion and a held-out scoring tail.

    Walks forward: the validation slice is the *most recent* experiences, and
    training only ever sees what came before them. Returns ``([], [])`` when the
    batch is too small to support both sides, which the caller treats as "do not
    update" rather than "update without validating".

    Skipped batches are not consumed — ``_buffer`` keeps them, so the experiences
    are picked up by the next update instead of being lost.
    """
    if len(batch) < _MIN_BATCH_FOR_VALIDATION:
        return [], []

    n_val = int(len(batch) * _VAL_FRACTION)
    # Guarantee both sides clear their own minimum, whatever the fraction gives.
    n_val = max(n_val, _MIN_VALIDATION)
    n_val = min(n_val, len(batch) - _MIN_TRAIN)
    if n_val < _MIN_VALIDATION or (len(batch) - n_val) < _MIN_TRAIN:
        return [], []
    return batch[: len(batch) - n_val], batch[len(batch) - n_val:]


def _validation_metric(model, val_batch: list[dict]) -> float:
    """Score a policy on held-out experiences. Lower is better.

    Uses the same PPO surrogate objective the update optimises, so the incumbent
    and the candidate are compared on one scale. Only the reported loss and
    entropy terms are used — the advantage is normalised, not scaled by the
    value estimate — which keeps the number comparable between two policies
    without either of them moving it by changing its own critic.
    """
    import torch as th

    if not val_batch:
        return float("nan")

    obs = th.tensor(np.stack([e["obs"] for e in val_batch]).astype(np.float32))
    act = th.tensor([e["action"] for e in val_batch], dtype=th.long)
    ret = th.tensor([e["reward"] for e in val_batch], dtype=th.float32)

    was_training = model.policy.training
    model.policy.eval()
    try:
        with th.no_grad():
            _vals, log_probs, entropy = model.policy.evaluate_actions(obs, act)
            # -log p(action) as a cross-entropy proxy, plus a small entropy
            # bonus. Deterministic, data-only, and identical for both policies.
            score = float(((-log_probs).mean() - 0.01 * entropy.mean()).item())
    finally:
        if was_training:
            model.policy.train()
    return score


def _make_training_model(model, registry):
    """A private, independently trainable clone of the live model.

    Uses the registry's deep copy so the update thread's gradients never touch
    the object the agent thread infers from. Falls back to the live model only
    when no registry is bound, which keeps the pre-registry path working.
    """
    if registry is None:
        return model
    return _clone_model(model)


def _clone_model(model):
    import copy as _copy

    clone = _copy.deepcopy(model)
    # The optimizer holds references to the original parameters, so it must be
    # rebuilt against the clone's own parameters or .step() would still write to
    # the live model's tensors.
    try:
        policy = clone.policy
        policy.optimizer = type(policy.optimizer)(
            policy.parameters(), lr=_FINE_TUNE_LR
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not build a trainable clone: {exc}") from exc
    return clone


def stats() -> dict:
    """Consistent snapshot of the counters for the dashboard."""
    with _state_lock:
        return {
            "is_training": _is_training,
            "new_count": _new_count,
            "buffer_size": len(_buffer),
            "total_updates": _total_updates,
            "rejected_updates": _total_rejected,
            "threshold": _UPDATE_THRESHOLD,
        }


def _rt_log(event: str, **kwargs) -> None:
    _retrain_log.append({
        "ts":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **kwargs,
    })


def get_retrain_log() -> list[dict]:
    return list(_retrain_log)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called from apex_dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

def record_entry(symbol: str, obs: np.ndarray, action: int,
                 entry_price: float, atr: float, regime: str = "bullish") -> None:
    """Store the observation + action when an RL-driven trade opens."""
    _pending[symbol] = {
        "obs":         obs.copy(),
        "action":      action,
        "entry_price": entry_price,
        "atr":         float(atr),
        "regime":      regime,  # Task 32: store regime at entry
    }
    logger.debug("RL entry recorded: %s  action=%d  price=%.2f  regime=%s", symbol, action, entry_price, regime)


def record_exit(symbol: str, exit_price: float, pnl: float,
                atr_at_entry: float, side: str = "long", portfolio_stress: float = 0.5) -> None:
    """Called when a trade closes.  Computes ATR-normalised reward, buffers
    the experience, and triggers a gradient update when the threshold is met.
    Task 32: Weights reward by inverse portfolio_stress (less learning when stressed)."""
    global _new_count

    pending = _pending.pop(symbol, None)
    if pending is None:
        return  # not an RL-driven trade

    obs    = pending["obs"]
    action = pending["action"]
    entry  = pending["entry_price"]
    atr    = atr_at_entry if atr_at_entry > 0 else pending.get("atr", 0.0)
    regime = pending.get("regime", "bullish")  # Task 32: get regime at entry

    # ATR-normalised reward keeps the scale consistent across volatile/calm stocks
    if atr > 0:
        raw = (exit_price - entry) / atr if side == "long" else (entry - exit_price) / atr
    else:
        raw = pnl / max(entry * 0.015, 1e-9)

    # Task 32: Weight reward by inverse portfolio_stress (less learning when stressed)
    stress_weight = max(0.2, 1.0 - portfolio_stress)  # 0.2 to 1.0
    reward = float(max(-3.0, min(3.0, raw * stress_weight)))

    _buffer.append({"obs": obs, "action": action, "reward": reward, "regime": regime})  # Task 32: store regime
    with _state_lock:
        _new_count += 1
        due = _new_count >= _UPDATE_THRESHOLD and not _is_training
        if due:
            _new_count = 0
    logger.debug("RL exit: %s  reward=%.3f  stress_weight=%.2f  buf=%d  new_count=%d",
                 symbol, reward, stress_weight, len(_buffer), _new_count)

    if due:
        threading.Thread(target=_run_update, daemon=True, name="rl-online-update").start()


# ─────────────────────────────────────────────────────────────────────────────
# Internal — PPO gradient update
# ─────────────────────────────────────────────────────────────────────────────

def _run_update() -> None:
    global _is_training, _total_updates, _total_rejected

    # Claim the update slot under the lock. This used to be a check-then-set
    # race, so several update threads could run at once on one batch.
    with _state_lock:
        if _is_training:
            return
        _is_training = True

    try:
        from trading_agent.integration.model_registry import get_registry
        import trading_agent.integration.rl_signal as _rl  # type: ignore

        model = _rl._model
        registry = get_registry()
        if model is None:
            _rt_log("skipped", msg="Model not loaded — skipping online update")
            return

        batch = list(_buffer)
        if len(batch) < 4:
            _rt_log("skipped", msg=f"Buffer too small ({len(batch)} < 4) — skipping")
            return

        # Hold out the most recent experiences before anything trains, so the
        # candidate and the incumbent can be scored on data neither has seen.
        train_batch, val_batch = _split_train_validation(batch)
        if not val_batch:
            _rt_log(
                "skipped",
                msg=(f"Batch of {len(batch)} is too small to validate "
                     f"(need {_MIN_BATCH_FOR_VALIDATION}) — skipping; the "
                     f"experiences stay buffered for the next update"),
            )
            return

        # Task 32: Stratify by regime for updates
        regimes = ["bullish", "bearish", "choppy", "crisis"]
        regime_batches = {r: [e for e in train_batch if e.get("regime", "bullish") == r] for r in regimes}

        n = len(train_batch)
        _rt_log("started", trades=len(batch), train_size=n, val_size=len(val_batch),
                msg=(f"Triggered by {len(batch)} closed trades — "
                     f"{n} to train, {len(val_batch)} held out for validation…"))
        logger.info(
            "Online RL update: batch_size=%d, train=%d, val=%d, regime_dist=%s",
            len(batch), n, len(val_batch), {r: len(v) for r, v in regime_batches.items()},
        )

        import torch as th
        import torch.nn.functional as F

        # Train a private copy. The live policy is only touched by publish(),
        # which copies the finished weights in under the registry's exclusive
        # lock — so a forward pass in another thread can never read a tensor
        # that optimizer.step() is midway through writing.
        work = _make_training_model(model, registry)

        with _update_lock:
            # Score the incumbent on the holdout *before* training, so the
            # comparison is like-for-like: same data, same objective, two
            # different weight sets.
            val_before = _validation_metric(model, val_batch)

            # Process each regime separately for stratified updates
            for regime in regimes:
                regime_batch = regime_batches[regime]
                if len(regime_batch) < 2:  # Need at least 2 for gradient
                    continue

                obs_t = th.tensor(
                    np.stack([e["obs"] for e in regime_batch]).astype(np.float32)
                )
                act_t = th.tensor([e["action"] for e in regime_batch], dtype=th.long)
                ret_t = th.tensor([e["reward"] for e in regime_batch], dtype=th.float32)

                # Baseline loss before touching weights
                with th.no_grad():
                    vals_old, lp_old, _ = work.policy.evaluate_actions(obs_t, act_t)
                    vals_old = vals_old.squeeze(-1)
                    adv_pre  = ret_t - vals_old
                    loss_before = float((-(lp_old * adv_pre)).mean().abs())

                loss_after = loss_before

                for _epoch in range(_N_EPOCHS):
                    vals, log_probs, entropy = work.policy.evaluate_actions(obs_t, act_t)
                    vals = vals.squeeze(-1)

                    adv = ret_t - vals.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    ratio   = th.exp(log_probs - lp_old.detach())
                    pg_loss = -th.min(
                        ratio * adv,
                        th.clamp(ratio, 1 - _CLIP_EPS, 1 + _CLIP_EPS) * adv,
                    ).mean()
                    v_loss  = F.mse_loss(vals, ret_t)
                    e_loss  = -entropy.mean()
                    loss    = pg_loss + 0.5 * v_loss + 0.01 * e_loss

                    if th.isnan(loss) or th.isinf(loss):
                        raise ValueError(f"Non-finite loss ({float(loss):.4f}) at epoch {_epoch}")

                    for pg in work.policy.optimizer.param_groups:
                        pg["lr"] = _FINE_TUNE_LR

                    work.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(work.policy.parameters(), 0.5)
                    work.policy.optimizer.step()
                    loss_after = float(loss.item())

            improvement = round(
                (loss_before - loss_after) / (abs(loss_before) + 1e-9) * 100, 1
            )

            # Now score the candidate on the same held-out slice.
            val_after = _validation_metric(work, val_batch)

        # The decision. Every branch fails closed: a comparison that could not be
        # made, or was not an improvement, leaves the live policy alone. NaN and
        # inf are excluded explicitly because every NaN comparison is False
        # (which would silently reject everything) and inf beats any finite value
        # (which would silently accept everything).
        finite = np.isfinite(val_before) and np.isfinite(val_after) and val_before > 0
        val_improvement = (
            round((val_before - val_after) / (abs(val_before) + 1e-9) * 100, 1)
            if finite else None
        )
        accept = (
            val_improvement is not None
            and val_improvement > _MIN_VALIDATION_IMPROVEMENT_PCT
        )

        if not accept:
            with _state_lock:
                _total_rejected += 1
            reason = (
                "non-finite or non-positive validation score"
                if not finite else
                f"held out: {val_before:.4f} → {val_after:.4f} "
                f"({val_improvement:+.1f}%, needed > "
                f"{_MIN_VALIDATION_IMPROVEMENT_PCT:+.1f}%)"
            )
            _rt_log("rejected", trades=len(batch),
                    training_improvement_pct=improvement,
                    validation_before=val_before, validation_after=val_after,
                    validation_improvement_pct=val_improvement, applied=False,
                    msg=f"✗ HELD BACK — {reason}")
            logger.info(
                "Online RL update rejected: validation %.4f → %.4f (%+.1f%%); "
                "training loss said %+.1f%%, which did not carry over",
                val_before, val_after, val_improvement or 0.0, improvement,
            )
            return

        # Only now may the live policy and the persisted artifact change.
        with _update_lock:
            # Back up the current weights, then persist and publish.
            #
            # The save goes through artifact_store rather than straight to
            # settings.best_model_path. That path is git-tracked so the model
            # ships in the image, which made it the default write target and
            # meant every online update dirtied the working tree and was
            # discarded on redeploy. The store writes to the runtime directory
            # only, atomically, so a crash mid-save leaves the previous model
            # intact instead of a half-written zip.
            from trading_agent.config import settings  # type: ignore
            from trading_agent.integration import artifact_store

            artifact_store.backup("best_model.zip", settings.model_dir)
            artifact_store.write_atomic(
                work.save, "best_model.zip", settings.model_dir
            )
            if registry is not None:
                registry.publish(work.policy)

        with _state_lock:
            _total_updates += 1
        _rt_log(
            "completed",
            trades=len(batch),
            loss_before=round(loss_before, 4),
            loss_after=round(loss_after, 4),
            improvement_pct=improvement,
            validation_before=val_before,
            validation_after=val_after,
            validation_improvement_pct=val_improvement,
            applied=True,
            msg=(
                f"loss {loss_before:.3f} → {loss_after:.3f}  "
                f"({improvement:+.1f}%)  |  held out: {val_before:.4f} → "
                f"{val_after:.4f} ({val_improvement:+.1f}%)  ✓ APPLIED"
            ),
        )
        logger.info(
            "Online update complete: training %+.1f%%, validation %+.1f%%",
            improvement, val_improvement,
        )

    except Exception as exc:
        _rt_log("failed", error=str(exc), applied=False,
                msg=f"✗ FAILED — {exc}")
        logger.error("Online RL update failed: %s", exc, exc_info=True)

    finally:
        with _state_lock:
            _is_training = False
