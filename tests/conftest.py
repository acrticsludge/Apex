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
