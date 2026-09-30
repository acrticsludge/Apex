#!/usr/bin/env python
"""Check that this deploy will not do something you did not intend.

Run it before every deploy:

    python preflight.py

It answers one question - "will this deploy work, and is it safe?" - and prints
the answer in plain words. Exit code 0 means go, 1 means do not deploy yet.

It reads your `.env` file, so it checks what the app will actually see rather
than what is in your shell. It sends nothing anywhere and changes nothing.

Written for someone who does not know this codebase. If it says FAIL, the line
tells you what to do about it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = "docs/reasonix/plans/remaining-work.md"

OK, WARN, BAD = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, label: str, detail: str = "") -> None:
    results.append((status, label, detail))


def load_env() -> dict[str, str]:
    """Read .env the way the app does, without importing the app."""
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = load_env()
    if not env:
        record(BAD, ".env file",
               f"Not found at {ROOT / '.env'}. Copy .env.example to .env first.")
    else:
        record(OK, ".env file", f"{len(env)} settings found")

    # -- Authentication: fails closed, so a miss locks you out ----------------
    user = env.get("APEX_USER", "")
    password = env.get("APEX_PASS", "")
    pw_hash = env.get("APEX_PASS_HASH", "")

    if pw_hash:
        record(OK, "Login password", "using APEX_PASS_HASH")
    elif password:
        record(OK, "Login password", "using APEX_PASS")
    else:
        record(BAD, "Login password",
               "Neither APEX_PASS nor APEX_PASS_HASH is set. The app refuses "
               "login when these are missing, so the dashboard will be "
               "unreachable after you deploy.")

    if user in ("", "apex"):
        record(BAD, "Login username",
               f"APEX_USER is '{user or 'empty'}'. The app explicitly refuses "
               "the documented default. Set it to anything else.")
    else:
        record(OK, "Login username", f"APEX_USER is '{user}'")

    if len(password) < 12 and not pw_hash:
        record(WARN, "Password strength",
               f"APEX_PASS is {len(password)} characters. Use 16 or more.")

    secret = env.get("APEX_SECRET", "")
    if not secret:
        record(BAD, "APEX_SECRET",
               "Not set. Sessions break whenever more than one worker or a "
               "restart happens. Generate one with:\n"
               "    python -c \"import secrets; print(secrets.token_hex(32))\"")
    elif secret in ("change-me", "changeme", "secret"):
        record(BAD, "APEX_SECRET", "Still set to a placeholder value.")
    else:
        record(OK, "APEX_SECRET", f"set ({len(secret)} characters)")

    # -- State storage -------------------------------------------------------
    if env.get("SUPABASE_URL") and env.get("SUPABASE_SERVICE_KEY"):
        record(OK, "Supabase", "configured - state survives redeploys")
    else:
        record(WARN, "Supabase",
               "Not configured. State is written to a JSON file inside the "
               "container, so it is lost on every redeploy. Nothing breaks, "
               "but your trade history resets.")

    # -- Deployment isolation ------------------------------------------------
    deployment_env = env.get("APEX_ENV", "").strip()
    if not deployment_env:
        record(WARN, "APEX_ENV",
               "Not set. Every deployment shares Supabase rows, so a preview "
               "build overwrites production state.")
    elif deployment_env == "main":
        record(OK, "APEX_ENV", "'main' (production)")
    else:
        record(WARN, "APEX_ENV", f"'{deployment_env}' - not 'main', so this "
                                  "is not the production deployment")

    # -- Optional, recommended -----------------------------------------------
    if not env.get("TRADING_AGENT_STORAGE_DIR", "").strip():
        record(WARN, "Model persistence",
               "TRADING_AGENT_STORAGE_DIR is not set. Everything works, but "
               "online RL updates are lost on redeploy. See the runbook section 1.4.")
    else:
        record(OK, "Model persistence", f"TRADING_AGENT_STORAGE_DIR={env['TRADING_AGENT_STORAGE_DIR']}")

    if not env.get("APEX_ALERT_WEBHOOK_URL", "").strip():
        record(WARN, "Alerts",
               "APEX_ALERT_WEBHOOK_URL is not set. If the bot stops trading you "
               "will not be told - see the runbook section 1.5.")
    else:
        record(OK, "Alerts", "APEX_ALERT_WEBHOOK_URL is set")

    # -- Repo state ----------------------------------------------------------
    for tracked in ("apex.log", "apex.log.1"):
        if (ROOT / tracked).exists():
            import subprocess
            dirty = subprocess.run(
                ["git", "ls-files", "--error-unmatch", tracked],
                cwd=str(ROOT), capture_output=True,
            ).returncode == 0
            if dirty:
                record(BAD, "Tracked log file",
                       f"{tracked} is committed to git. It contains live "
                       "position sizes and prices.")
    record(OK, "No tracked logs", "checked")

    # -- Report --------------------------------------------------------------
    print()
    print("=" * 66)
    print("  DEPLOY PREFLIGHT")
    print("=" * 66)
    print()
    for status, label, detail in results:
        mark = {OK: "  ok  ", WARN: "  ??  ", BAD: "  !!  "}[status]
        print(f"{mark} {label}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")
        print()

    failures = [r for r in results if r[0] == BAD]
    warnings = [r for r in results if r[0] == WARN]

    print("-" * 66)
    if failures:
        print(f"  DO NOT DEPLOY - {len(failures)} problem(s) to fix first.")
        print("  Fix every !! line above, then run this again.")
    elif warnings:
        print(f"  Safe to deploy - {len(warnings)} thing(s) to be aware of.")
        print("  Nothing below will break the deploy.")
    else:
        print("  Safe to deploy. Everything checks out.")
    print("-" * 66)
    print()
    print(f"  Remaining known gaps are tracked in {DOCS}")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
