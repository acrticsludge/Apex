"""Shared pytest setup.

Imports of apex_dashboard must be side-effect free, so the autostart guard
(APEX_SKIP_AUTOSTART) is set before anything imports the module. The real
.env is deliberately NOT loaded into these tests.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["APEX_SKIP_AUTOSTART"] = "1"
os.environ.setdefault("APEX_USER", "test-user")
os.environ.setdefault("APEX_PASS", "test-pass-not-a-real-secret")
os.environ.setdefault("APEX_SECRET", "test-secret-key-for-suite-only")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "")
os.environ.setdefault("APEX_ENV", "pytest")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def dash():
    """The imported apex_dashboard module."""
    import apex_dashboard

    return apex_dashboard


@pytest.fixture
def client(dash):
    """A logged-in Flask test client."""
    dash.app.config.update(TESTING=True)
    with dash.app.test_client() as c:
        with c.session_transaction() as sess:
            sess["logged_in"] = True
        yield c


@pytest.fixture
def anon_client(dash):
    """A Flask test client with no session."""
    dash.app.config.update(TESTING=True)
    with dash.app.test_client() as c:
        yield c


@pytest.fixture
def real_ppo():
    """A small but genuine PPO with an attached optimiser.

    Tests that need to assert something about real gradient behaviour cannot use
    a fake policy: a fake cannot show that weights actually moved, that a
    candidate really validated worse, or that an optimizer was rebuilt against a
    clone's own parameters. The cost is a few seconds per test, paid only by the
    files that ask for it.

    Returns (model, weights_before) so a caller can assert on what changed.
    """
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium import spaces
    import gymnasium as gym

    dim, actions = 12, 3

    def _env():
        class _E(gym.Env):
            observation_space = spaces.Box(-1, 1, (dim,), np.float32)
            action_space = spaces.Discrete(actions)

            def reset(self, seed=None, options=None):
                return np.zeros(dim, np.float32), {}

            def step(self, action):
                return np.zeros(dim, np.float32), 0.0, False, False, {}

        return _E()

    model = PPO("MlpPolicy", DummyVecEnv([_env]), n_steps=16, batch_size=8, verbose=0)
    before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    return model, before


@pytest.fixture
def accepting_updates(monkeypatch):
    """Make the online learner's validation gate accept every candidate.

    Tests about gradient mechanics, persistence and publish atomicity are not
    tests of the gate. Left alone they would mostly assert "nothing changed",
    because the gate correctly refuses a candidate that did not validate better
    on held-out data — which says nothing about whether the training path works.

    Taking this fixture states that intent explicitly. Tests that *are* about the
    gate deliberately do not take it.
    """
    import trading_agent.integration.online_learner as ol

    seq = [1.0, 0.5]  # incumbent, then a strictly better candidate

    def _metric(model, val_batch, *a, **k):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(ol, "_validation_metric", _metric)
    return ol
