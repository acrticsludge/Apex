"""Fully-automatic online PPO fine-tuning from real closed-trade experience.

After every _UPDATE_THRESHOLD closed trades the module fires a background thread
that runs a small PPO gradient update directly on the loaded model weights, then
saves the updated model to disk.  No retraining loop or environment needed.
"""
from __future__ import annotations

import logging
import shutil
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

# ── State ─────────────────────────────────────────────────────────────────────
_buffer:      deque[dict] = deque(maxlen=_BUFFER_MAXLEN)
_pending:     dict[str, dict] = {}          # symbol → trade-open context
_retrain_log: deque[dict]     = deque(maxlen=500)
_update_lock  = threading.Lock()
_new_count    = 0      # experiences accumulated since last update trigger
_is_training  = False
_total_updates = 0     # cumulative completed updates


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
    _new_count += 1
    logger.debug("RL exit: %s  reward=%.3f  stress_weight=%.2f  buf=%d  new_count=%d",
                 symbol, reward, stress_weight, len(_buffer), _new_count)

    if _new_count >= _UPDATE_THRESHOLD and not _is_training:
        _new_count = 0
        threading.Thread(target=_run_update, daemon=True, name="rl-online-update").start()


# ─────────────────────────────────────────────────────────────────────────────
# Internal — PPO gradient update
# ─────────────────────────────────────────────────────────────────────────────

def _run_update() -> None:
    global _is_training, _total_updates

    if _is_training:
        return
    _is_training = True

    try:
        import trading_agent.integration.rl_signal as _rl  # type: ignore

        model = _rl._model
        if model is None:
            _rt_log("skipped", msg="Model not loaded — skipping online update")
            return

        batch = list(_buffer)
        if len(batch) < 4:
            _rt_log("skipped", msg=f"Buffer too small ({len(batch)} < 4) — skipping")
            return

        # Task 32: Stratify by regime for updates
        regimes = ["bullish", "bearish", "choppy", "crisis"]
        regime_batches = {r: [e for e in batch if e.get("regime", "bullish") == r] for r in regimes}
        
        n = len(batch)
        _rt_log("started", trades=n,
                msg=f"Triggered by {n} closed trades — computing gradients…")
        logger.info("Online RL update: batch_size=%d, regime_dist=%s", n, 
                    {r: len(v) for r, v in regime_batches.items()})

        import torch as th
        import torch.nn.functional as F

        with _update_lock:
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
                    vals_old, lp_old, _ = model.policy.evaluate_actions(obs_t, act_t)
                    vals_old = vals_old.squeeze(-1)
                    adv_pre  = ret_t - vals_old
                    loss_before = float((-(lp_old * adv_pre)).mean().abs())

                loss_after = loss_before

                for _epoch in range(_N_EPOCHS):
                    vals, log_probs, entropy = model.policy.evaluate_actions(obs_t, act_t)
                    vals = vals.squeeze(-1)

                    adv = ret_t - vals.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    ratio   = th.exp(log_probs - lp_old.detach())
                    pg_loss = -th.min(
                        ratio * adv,
                        th.clamp(ratio, 1 - _CLIP_EPS, 1 + _CL_EPS) * adv,
                    ).mean()
                    v_loss  = F.mse_loss(vals, ret_t)
                    e_loss  = -entropy.mean()
                    loss    = pg_loss + 0.5 * v_loss + 0.01 * e_loss

                    if th.isnan(loss) or th.isinf(loss):
                        raise ValueError(f"Non-finite loss ({float(loss):.4f}) at epoch {_epoch}")

                    for pg in model.policy.optimizer.param_groups:
                        pg["lr"] = _FINE_TUNE_LR

                    model.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    model.policy.optimizer.step()
                    loss_after = float(loss.item())

            improvement = round(
                (loss_before - loss_after) / (abs(loss_before) + 1e-9) * 100, 1
            )

            # Backup previous weights then save updated model
            from trading_agent.config import settings  # type: ignore
            backup = settings.best_model_path.parent / "best_model_pre_online.zip"
            if settings.best_model_path.exists():
                shutil.copy2(str(settings.best_model_path), str(backup))
            model.save(str(settings.best_model_path))

        _total_updates += 1
        _rt_log(
            "completed",
            trades=n,
            loss_before=round(loss_before, 4),
            loss_after=round(loss_after, 4),
            improvement_pct=improvement,
            applied=True,
            msg=(
                f"loss {loss_before:.3f} → {loss_after:.3f}  "
                f"({improvement:+.1f}%)  ✓ APPLIED"
            ),
        )
        logger.info("Online update complete: improvement=%.1f%%", improvement)

    except Exception as exc:
        _rt_log("failed", error=str(exc), applied=False,
                msg=f"✗ FAILED — {exc}")
        logger.error("Online RL update failed: %s", exc, exc_info=True)

    finally:
        _is_training = False
