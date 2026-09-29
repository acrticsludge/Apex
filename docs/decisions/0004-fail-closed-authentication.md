# ADR-004: Fail-closed dashboard authentication

- Status: Accepted
- Date: 2026-09-29
- Related: `docs/operations/deployment-runbook.md` (this decision changes the
  deploy procedure)

## Context

Apex's dashboard is the only control surface for an automated trading agent. A
compromise is not a data breach, it is an adversary who can start and stop
trading, liquidate positions, rewrite the accounting ledger, or change risk
limits.

The original gate was a single shared password:

```python
_AUTH_USER = os.environ.get("APEX_USER", "apex")
_AUTH_PASS = os.environ.get("APEX_PASS", "admin")
```

If `APEX_PASS` was unset in the deployment environment, the dashboard was
reachable with `apex` / `admin` — the values documented in `.env.example`, and
therefore the values a "copy the example" deployment would use. There was no
lockout, no rate limit, no timing-safe comparison, and the comparison was a
plain `==`. The log already contained two failed login attempts from a scanner.

Separately, `POST /api/config` cast and stored client values with no bounds
check, and `POST /api/edit/position` accepted a negative `qty` — which made
`execute_sell` compute negative proceeds, subtract cash, and inflate
`realised_pnl` by an arbitrary amount.

## Decision

**An unset password must not produce a guessable password.**

1. **No guessable fallback.** If `APEX_PASS` is unset, the app mints a random
   `secrets.token_urlsafe(32)` at boot. If `APEX_USER`/`APEX_PASS` are both the
   documented defaults, login is refused with 503 and an ERROR is logged — this
   is checked at request time, so it also catches an explicit `APEX_PASS=admin`.
2. **Throttle.** Five failures per IP, five-minute lockout, tracked in a
   `_login_failures` dict under its own lock. A successful login clears it.
3. **Constant-time compare** via `hmac.compare_digest` on both fields.
4. **Optional hashed password.** `APEX_PASS_HASH` accepts
   `<salt_hex>$<scrypt_digest_hex>`, so the plaintext need not sit in the
   environment. Takes precedence over `APEX_PASS`.
5. **Validate every client-writable value.** `apex_config.CONFIG_BOUNDS` gives
   each setting a legal domain; `clean_numeric` does the same for ledger edits.
   Out-of-range, non-finite and unknown keys are rejected with 400.
6. **The risk-bypass toggle requires confirmation.** `settings_enabled=false`
   bypasses the confidence gate, position cap, ADX and index filters and the
   ATR stop clamps. It now needs
   `{"settings_enabled": false, "confirm_disable_settings": true}`, and the UI
   asks first.
7. **Hardening around it.** `SESSION_COOKIE_SECURE` + `HttpOnly`, four security
   headers including a CSP, and a 256 KB request body cap.

## Alternatives considered

**Keep a default and log a warning.** Rejected: a warning in a log file is not
a control. For a system that places orders, the safe failure is the one that
stops.

**Fail startup hard when `APEX_PASS` is unset.** Rejected for now: it would
prevent the read-only dashboard from starting, so an operator with a mis-set
environment has no way to see the problem. The 503-on-login path surfaces it
and keeps the UI reachable. Revisit once the dashboard is the only way to
observe state.

**Put the dashboard behind a VPN or IP allowlist.** Complementary, not a
substitute. Worth adding if the Railway deployment is ever internet-facing;
not sufficient on its own because the source IP is not fixed.

**Add CSRF tokens.** Deferred, not rejected. Every mutating route is POST-only
and `SameSite=Lax` blocks the cookie on cross-site POST, so it is not currently
exploitable. It becomes a real gap the moment any GET route mutates.
`test_route_inventory.py` enforces that no mutating route answers GET.

## Consequences

- **This changes the deploy procedure.** Deploying without `APEX_PASS` set now
  locks the dashboard rather than opening it with `admin`. The runbook has the
  pre-deploy checklist.
- Operators cannot disable the risk gates with one click any more. That is the
  intent; the confirmation is deliberate friction in front of a control that
  removes the control.
- `APEX_PASS_HASH` uses `hashlib.scrypt` from the standard library — no new
  dependency. A helper snippet is in `.env.example`.
- The session secret falls back to a per-process random value. Safe at
  `--workers 1`; it becomes a session-breakage trap if anyone scales workers,
  which is worth a test if that changes.
