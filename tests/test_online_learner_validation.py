"""Tests for the online learner's validation gate.

The defect this exists to close: `_run_update` computed `improvement` from the
before/after *training* loss and then published unconditionally. Lower training
loss means the policy fitted that batch harder, not that it got better. There
was no held-out data anywhere in the module — `validation_metrics.json` and
`evaluation_metrics.json` are both absent from the repo.

That was survivable while updates were discarded on every redeploy. It stopped
being survivable once the artifact store made them persist: a bad update now
survives instead of evaporating.

The gate holds back an update unless the trained policy beats the live policy on
experiences neither of them trained on. Two details make the comparison mean
something:

- The held-out slice is the *most recent* part of the buffer, never a random
  sample. Random validation over a time series leaks the future into training.
- Both policies are scored on the identical slice with the identical objective,
  so the number is like-for-like rather than two different metrics.

Every test asserts on the observable outcome — whether the live weights moved,
whether the artifact was rewritten — not on internal state, because that is what
a stuck gate or a leaky one would look like in production.

`_validation_metric` is called twice per update: once for the incumbent, once for
the candidate. `_scored_as` supplies those two numbers in order, which is how a
test states "the candidate validated worse" without reaching into the module.
"""
from pathlib import Path

import numpy as np
import pytest

import trading_agent.integration.online_learner as ol


REGIMES = ["bullish", "bearish", "choppy", "crisis"]


@pytest.fixture(autouse=True)
def clean():
    ol._buffer.clear()
    ol._pending.clear()
    ol._retrain_log.clear()
    ol._new_count = 0
    ol._is_training = False
    ol._total_updates = 0
    ol._total_rejected = 0
    yield
    from trading_agent.integration.model_registry import reset_for_tests
    reset_for_tests()
    ol._buffer.clear()
    ol._pending.clear()
    ol._new_count = 0
    ol._is_training = False
    ol._total_rejected = 0


def _seed(n=40, dim=12, actions=3, regimes=True):
    """Buffer the learner with genuine 2-D observations and valid actions."""
    for i in range(n):
        ol._buffer.append({
            "obs": np.random.randn(dim).astype(np.float32),
            "action": i % actions,
            "reward": 1.0 + float(np.random.randn()) * 0.01,
            "regime": REGIMES[i % 4] if regimes else "bullish",
        })


def _live(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def _changed(before, after):
    return [k for k in before
            if not np.array_equal(before[k].cpu().numpy(), after[k].cpu().numpy())]


def _scored_as(incumbent, candidate, record=None):
    """A validation metric returning `incumbent` then `candidate`.

    `record`, if given, collects the slice each call was handed so a test can
    assert the two policies were judged on the same data.
    """
    seq = [incumbent, candidate]

    def _metric(model, val_batch, *a, **k):
        if record is not None:
            record.append([id(e) for e in val_batch])
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return _metric


@pytest.fixture
def real(real_ppo, tmp_path, monkeypatch):
    """A real PPO wired the way the app wires it, writing into a temp dir."""
    import trading_agent.config as cfgmod
    import trading_agent.integration.rl_signal as rl
    from trading_agent.integration.model_registry import bind

    model, _ = real_ppo
    model_dir = tmp_path / "agent" / "model"
    monkeypatch.setattr(rl, "_model", model, raising=False)
    monkeypatch.setattr(cfgmod.settings, "model_dir", model_dir, raising=False)
    monkeypatch.setattr(cfgmod.settings, "best_model_path",
                        model_dir / "best_model.zip", raising=False)
    bind(model)
    return model


# ── The gate blocks a regression, and allows an improvement ─────────────────

def test_a_validation_regression_does_not_reach_the_live_policy(real, monkeypatch):
    """The headline case: the candidate scores worse on held-out data, so the
    live policy must be left exactly as it was."""
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    ol._run_update()
    assert not _changed(before, _live(real)), (
        "a policy that validated worse was published to the live model"
    )


def test_a_validation_improvement_is_published(real, monkeypatch):
    """The control for the test above: a candidate that validates better is
    taken, so the gate is not simply refusing everything."""
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0))
    ol._run_update()
    assert _changed(before, _live(real)), (
        "a policy that validated better was not published — the gate is too strict"
    )


def test_a_rejected_update_does_not_rewrite_the_artifact(real, monkeypatch):
    """Persistence makes a bad write expensive, so the disk must be untouched."""
    from trading_agent.integration import artifact_store as store
    model_dir = tmp_path_of(real)
    # A known-good artifact already on disk.
    store.write_atomic(lambda p: Path(p).write_bytes(b"the good model"),
                       "best_model.zip", model_dir)
    good = (model_dir / "best_model.zip").read_bytes()

    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    ol._run_update()

    assert (model_dir / "best_model.zip").read_bytes() == good, (
        "a rejected update still overwrote the persisted model"
    )


def test_an_accepted_update_replaces_the_artifact(real, monkeypatch):
    """The other direction: an approved update must actually reach the disk, or
    the gate would be pointless."""
    from trading_agent.integration import artifact_store as store
    model_dir = tmp_path_of(real)
    store.write_atomic(lambda p: Path(p).write_bytes(b"the old model"),
                       "best_model.zip", model_dir)

    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0))
    ol._run_update()

    assert (model_dir / "best_model.zip").read_bytes() != b"the old model", (
        "an approved update never reached the persisted model"
    )


def tmp_path_of(real):
    """The runtime model dir the `real` fixture pointed settings at."""
    import trading_agent.config as cfgmod
    return cfgmod.settings.model_dir


def test_an_equal_score_is_rejected_by_default(real, monkeypatch):
    """No measurable improvement is not a reason to change a working model."""
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.5, 1.5))
    ol._run_update()
    assert not _changed(before, _live(real)), (
        "an unchanged validation score was treated as an improvement"
    )


def test_a_configured_margin_requires_more_than_a_token_gain(real, monkeypatch):
    """The margin exists so noise-level 'improvements' cannot random-walk the
    policy away from a good starting point."""
    monkeypatch.setattr(ol, "_MIN_VALIDATION_IMPROVEMENT_PCT", 5.0)
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(100.0, 99.0))  # +1%
    ol._run_update()
    assert not _changed(before, _live(real)), "a 1% gain was published against a 5% margin"


def test_a_gain_above_the_margin_is_published(real, monkeypatch):
    monkeypatch.setattr(ol, "_MIN_VALIDATION_IMPROVEMENT_PCT", 5.0)
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(100.0, 80.0))  # +20%
    ol._run_update()
    assert _changed(before, _live(real)), "a 20% gain was rejected against a 5% margin"


# ── Undecidable comparisons fail closed ─────────────────────────────────────

def test_a_non_finite_candidate_score_rejects_the_update(real, monkeypatch):
    """Every NaN comparison is False, which would silently reject everything,
    and inf beats any finite value, which would silently accept anything."""
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, float("nan")))
    ol._run_update()
    assert not _changed(before, _live(real)), "a NaN validation score was acted on"


def test_a_non_finite_incumbent_score_rejects_the_update(real, monkeypatch):
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(float("inf"), 1.0))
    ol._run_update()
    assert not _changed(before, _live(real)), "an infinite baseline was treated as valid"


def test_a_non_positive_incumbent_score_rejects_the_update(real, monkeypatch):
    """Improvement is expressed relative to the baseline, so a zero or negative
    one has no meaningful percentage."""
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(0.0, 0.5))
    ol._run_update()
    assert not _changed(before, _live(real)), "a zero baseline produced a decision"


# ── Too little data to validate means no update ────────────────────────────

def test_too_few_experiences_to_split_are_skipped(real):
    before = _live(real)
    _seed(4)
    ol._run_update()
    assert not _changed(before, _live(real)), "trained without enough data to validate"


def test_the_skip_is_logged_and_explains_itself(real):
    _seed(6)
    ol._run_update()
    events = [e for e in ol.get_retrain_log() if e["event"] == "skipped"]
    assert events, "skipping for insufficient data was silent"
    msg = " ".join(e.get("msg", "") for e in events).lower()
    assert "validat" in msg, f"the skip did not mention validation: {msg!r}"


def test_skipped_batches_are_kept_for_the_next_update(real):
    """_new_count resets when the thread fires, so a buffer that was also
    cleared would lose those experiences outright."""
    _seed(6)
    ol._run_update()
    assert len(ol._buffer) == 6, "a skipped update discarded buffered experience"


# ── The holdout must be held out, and must be the recent part ──────────────

def _ordered_batch(n=20):
    return [{"obs": np.zeros(4, np.float32), "action": 0, "reward": float(i),
             "regime": "bullish"} for i in range(n)]


def test_the_validation_slice_is_not_trained_on(real, monkeypatch):
    """Leakage check: the gate is meaningless if the candidate trained on the
    data it is scored against."""
    trained_on, scored_on = [], []
    real_split = ol._split_train_validation

    def _spy(batch):
        train, val = real_split(batch)
        trained_on.extend(id(e) for e in train)
        scored_on.extend(id(e) for e in val)
        return train, val

    monkeypatch.setattr(ol, "_split_train_validation", _spy)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0))
    ol._run_update()
    assert scored_on, "no validation slice was produced"
    assert not set(trained_on) & set(scored_on), (
        "the validation slice was included in the training slice"
    )


def test_the_validation_slice_is_the_most_recent_experiences():
    """Walk-forward, not a random split. A random split over a time series
    trains on the future and validates on the past."""
    train, val = ol._split_train_validation(_ordered_batch(20))
    assert val, "no validation slice"
    rewards = [e["reward"] for e in val]
    assert rewards == sorted(rewards), f"validation is not the recent tail: {rewards}"
    assert rewards[-1] == 19.0, "the newest experience is not in validation"
    assert all(e["reward"] < 19.0 for e in train), "training contains the newest experience"


def test_the_split_leaves_both_sides_usable():
    train, val = ol._split_train_validation(_ordered_batch(20))
    assert len(train) >= 4, f"training slice too small: {len(train)}"
    assert len(val) >= 2, f"validation slice too small: {len(val)}"


def test_a_batch_too_small_to_split_yields_nothing():
    train, val = ol._split_train_validation(_ordered_batch(8))
    assert train == [] and val == [], (
        "a batch too small to validate must not silently train without a holdout"
    )


# ── Both policies are scored identically, and the incumbent is the live one ─

def test_the_incumbent_and_candidate_are_scored_on_the_same_slice(real, monkeypatch):
    """A comparison across different data is not a comparison."""
    record = []
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0, record=record))
    ol._run_update()
    assert len(record) == 2, f"expected two scores (incumbent, candidate), got {len(record)}"
    assert record[0] == record[1], "the two policies were scored on different data"


def test_the_incumbent_is_the_live_policy_not_the_trained_copy(real, monkeypatch):
    """The baseline must be what is in production, otherwise the gate is
    comparing a policy against itself."""
    live_before = _live(real)
    first_seen = {}

    def _metric(model, val_batch, *a, **k):
        if not first_seen:
            first_seen.update(
                (key, v.detach().clone())
                for key, v in model.policy.state_dict().items()
            )
        return 1.0

    monkeypatch.setattr(ol, "_validation_metric", _metric)
    _seed(40)
    ol._run_update()
    assert first_seen, "the validation metric was never called"
    assert not _changed(live_before, first_seen), (
        "the first score came from an already-trained copy, not the live policy"
    )


def test_the_candidate_is_the_trained_copy(real, monkeypatch):
    """The mirror of the test above: the second score must be the trained model,
    or the gate would be comparing the live policy with itself."""
    live_before = _live(real)
    second_seen = {}
    seq = [1.0, 1.0]

    def _metric(model, val_batch, *a, **k):
        if len(seq) == 1:
            second_seen.update(
                (key, v.detach().clone())
                for key, v in model.policy.state_dict().items()
            )
        return seq.pop(0)

    monkeypatch.setattr(ol, "_validation_metric", _metric)
    _seed(40)
    ol._run_update()
    assert second_seen, "the candidate was never scored"
    assert _changed(live_before, second_seen), (
        "the candidate was not the trained policy — the gate compared a model "
        "with itself"
    )


# ── The decision must be visible ────────────────────────────────────────────

def test_a_rejection_is_logged_with_the_numbers(real, monkeypatch):
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    ol._run_update()
    rejected = [e for e in ol.get_retrain_log() if e["event"] == "rejected"]
    assert rejected, "a rejected update left no record"
    e = rejected[-1]
    assert e["validation_before"] == pytest.approx(1.0)
    assert e["validation_after"] == pytest.approx(2.0)
    assert e["applied"] is False
    assert "HELD BACK" in e["msg"]


def test_an_acceptance_is_logged_with_the_numbers(real, monkeypatch):
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0))
    ol._run_update()
    applied = [e for e in ol.get_retrain_log() if e["event"] == "completed"]
    assert applied, "an applied update left no record"
    assert applied[-1]["validation_before"] == pytest.approx(2.0)
    assert applied[-1]["validation_after"] == pytest.approx(1.0)
    assert applied[-1]["applied"] is True


def test_a_rejection_is_counted(real, monkeypatch):
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    ol._run_update()
    assert ol.stats()["rejected_updates"] == 1, "a rejection was not counted"


def test_rejections_accumulate_across_updates(real, monkeypatch):
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    _seed(40)
    ol._run_update()
    _seed(40)
    ol._run_update()
    assert ol.stats()["rejected_updates"] == 2
    assert ol.stats()["total_updates"] == 0


def test_an_applied_update_is_counted_and_not_marked_rejected(real, monkeypatch):
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(2.0, 1.0))
    ol._run_update()
    assert ol.stats()["total_updates"] == 1
    assert ol.stats()["rejected_updates"] == 0


def test_a_rejected_update_still_trains_before_deciding(real, monkeypatch):
    """The candidate must be trained for there to be anything to validate. A
    gate evaluated on an untrained copy would always reject."""
    record = []
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0, record=record))
    ol._run_update()
    assert len(record) == 2, "the candidate was not trained before being scored"


def test_a_rejection_leaves_the_live_policy_object_in_place(real, monkeypatch):
    """Gradients run on a private clone, so a rejected update must leave no
    weight change and no stray publish."""
    from trading_agent.integration.model_registry import get_registry
    before = _live(real)
    _seed(40)
    monkeypatch.setattr(ol, "_validation_metric", _scored_as(1.0, 2.0))
    ol._run_update()
    assert not _changed(before, _live(real))
    assert get_registry().model is real
