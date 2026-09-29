"""Supply-chain and packaging guards.

requirements.txt used to carry 17 of 19 entries as a bare `>=`, so every deploy
re-resolved the tree and a yanked or compromised release arrived with no review.
These tests fail if a pin is loosened again.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
DEV_REQ = ROOT / "requirements-dev.txt"
CI = ROOT / ".github" / "workflows" / "ci.yml"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
DOCKERFILE = ROOT / "Dockerfile"
RAILWAYIGNORE = ROOT / ".railwayignore"
GITIGNORE = ROOT / ".gitignore"


def _requirements():
    entries = []
    for raw in REQ.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        entries.append(line)
    return entries


def test_requirements_are_not_empty():
    assert _requirements(), "requirements.txt parsed to nothing"


@pytest.mark.parametrize("entry", _requirements())
def test_every_requirement_is_upper_bounded(entry):
    """No bare `>=`: an unbounded range means a breaking major can land silently."""
    assert ">=" in entry or "==" in entry, f"{entry} has no version at all"
    if ">=" in entry:
        assert re.search(r">=\s*[\d.]+\s*,\s*<\s*[\d.]+", entry), (
            f"{entry} uses an open-ended range; use `>=x.y,<z` or an exact pin"
        )
    assert not entry.endswith(">"), f"{entry} has no upper bound"


def test_exact_pins_carry_no_range_operator():
    for entry in _requirements():
        if "==" in entry:
            assert ">=" not in entry, f"{entry} mixes an exact pin with a range"


def test_requirements_are_sorted_by_section_and_alphabetically():
    """Stable ordering keeps the Dependabot diffs readable."""
    sections, current = [], None
    for raw in REQ.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# ──"):
            current = raw
            sections.append((current, []))
        elif sections and raw.split("#", 1)[0].strip():
            sections[-1][1].append(raw.split("#", 1)[0].strip().lower())
    for name, entries in sections:
        assert entries == sorted(entries), f"section {name} is not alphabetical: {entries}"


def test_dev_requirements_extend_runtime():
    text = DEV_REQ.read_text(encoding="utf-8")
    assert "-r requirements.txt" in text
    assert "pytest" in text


def test_ci_runs_the_suite():
    text = CI.read_text(encoding="utf-8")
    assert "pytest" in text, "CI never runs the tests"
    assert "APEX_SKIP_AUTOSTART" in text, "CI imports the app with side effects enabled"
    assert "pip-audit" in text, "CI has no dependency vulnerability check"
    assert "docker build" in text, "CI never builds the deploy image"


def test_ci_asserts_jev_is_live_in_production_import_order():
    """The circular-import defect made every JEV gate a silent no-op in
    production while the suite stayed green, because the tests imported
    apex_jev first. CI must import the dashboard first, like gunicorn does."""
    text = CI.read_text(encoding="utf-8")
    assert "_JEV_AVAILABLE" in text, "CI does not assert the JEV subsystem loaded"
    assert "import apex_dashboard" in text


def test_dependabot_covers_pip_and_actions():
    text = DEPENDABOT.read_text(encoding="utf-8")
    assert "package-ecosystem: pip" in text
    assert "package-ecosystem: github-actions" in text


def test_deploy_ignores_secrets_and_runtime_state():
    for path, name in ((RAILWAYIGNORE, ".railwayignore"), (GITIGNORE, ".gitignore")):
        text = path.read_text(encoding="utf-8")
        assert ".env" in text, f"{name} does not exclude .env"
        assert "__pycache__" in text, f"{name} does not exclude __pycache__"
        # Either an exact name or a covering glob is fine; both are acceptable
        # ways to keep the log out of the repo and out of the image.
        assert "apex.log" in text or "*.log" in text, (
            f"{name} does not exclude the log"
        )
        assert "apex_dual_state.json" in text, f"{name} does not exclude the state file"


def test_dockerfile_copies_the_template_and_static_assets():
    """The SPA moved out of the Python module; the image must still ship it."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY . ." in text
    for required in ("templates/", "static/"):
        assert (ROOT / required).is_dir(), f"{required} is missing from the repo"


def test_pandas_ta_pin_is_preserved():
    """The trained feature columns were built against 0.4.71b0; silently
    upgrading it changes every indicator the model relies on."""
    assert "pandas-ta==0.4.71b0" in REQ.read_text(encoding="utf-8")
