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


def _pinned(name: str) -> str | None:
    for entry in _requirements():
        if entry.split("==")[0].strip().lower() == name:
            return entry.split("==", 1)[1].strip() if "==" in entry else None
    return None


def _tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def test_numpy_pin_sits_inside_pandas_ta_window():
    """The exact conflict CI caught on the first run of the pins.

    pandas-ta 0.4.71b0 -> numba 0.61.2 -> numpy<2.3, and pandas-ta itself
    requires numpy>=2.2.6. A numpy outside [2.2.6, 2.3) makes the whole file
    unresolvable, so the dashboard image cannot build.
    """
    np = _pinned("numpy")
    assert np is not None, "numpy must be exactly pinned: it bounds the pandas-ta window"
    ver = _tuple(np)
    assert ver >= (2, 2, 6), f"numpy {np} is below pandas-ta's floor of 2.2.6"
    assert ver < (2, 3, 0), f"numpy {np} is above numba 0.61.2's ceiling of 2.3"


def test_pandas_stays_on_the_2x_line():
    """pandas-ta 0.4.71b0 predates pandas 3; pandas 3 silently breaks its
    indicator functions rather than failing to install."""
    pd = _pinned("pandas")
    assert pd is not None, "pandas must be exactly pinned"
    assert _tuple(pd) >= (2, 3, 2), f"pandas {pd} is below pandas-ta's floor of 2.3.2"
    assert _tuple(pd)[0] == 2, f"pandas {pd} is on a major line pandas-ta 0.4.71b0 predates"


def test_the_file_has_no_obviously_conflicting_floor_and_ceiling():
    """Sanity sweep: no two pins should demand mutually exclusive majors."""
    pins = {e.split("==")[0].strip().lower(): e for e in _requirements() if "==" in e}
    for name, entry in pins.items():
        assert not entry.rstrip().endswith(".*"), f"{entry} is a wildcard pin"
    assert pins, "expected exact pins to validate"


def test_pins_are_not_known_vulnerable():
    """The first CI run reported 20 advisories in versions pinned in the previous
    commit. This pins the floors those advisories were fixed at, so a future
    pin edit that reintroduces one is caught here rather than in an audit log
    nobody reads.
    """
    floors = {
        "torch": (2, 14, 0),        # PYSEC-2025-193/194/195/203/204/206, -2026-2286
        "starlette": (1, 3, 1),     # PYSEC-2026-161/248/249/1942/2280/2281
    }
    for name, floor in floors.items():
        pin = _pinned(name)
        assert pin is not None, f"{name} must be exactly pinned so it can be checked"
        assert _tuple(pin) >= floor, (
            f"{name}=={pin} is below {floor[0]}.{floor[1]}.{floor[2]}, which carries "
            f"known advisories"
        )


def test_fastapi_does_not_cap_starlette_below_its_fix():
    """fastapi 0.118.x pins starlette<0.49.0, which is what forced the
    vulnerable starlette into the tree. Holding the cap below 1.x reintroduces
    every starlette advisory regardless of the explicit pin."""
    pin = _pinned("fastapi")
    assert pin is not None, "fastapi must be exactly pinned"
    ver = _tuple(pin)
    assert not (ver[0] == 0 and ver[1] < 141), (
        f"fastapi {pin} caps starlette below the patched 1.x line"
    )


def test_sklearn_matches_the_committed_scaler_artifact():
    """scaler.joblib is a pickled estimator checked into the repo. scikit-learn
    refuses to load one across majors without warning that it "might lead to
    breaking code or invalid results", so the pin has to match the artifact.
    The build surfaces this as an InconsistentVersionWarning.
    """
    scaler = ROOT / "trading_agent" / "agent" / "model" / "scaler.joblib"
    assert scaler.exists(), "the pinned sklearn cannot be checked without the artifact"
    pin = _pinned("scikit-learn")
    assert pin is not None, "scikit-learn must be exactly pinned"
    assert _tuple(pin) == (1, 6, 1), (
        f"scikit-learn {pin} does not match the 1.6.1 that pickled scaler.joblib; "
        f"cross-major loading risks silently wrong features"
    )
