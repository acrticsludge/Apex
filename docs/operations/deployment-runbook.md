# Deployment Runbook

Apex deploys to Railway from the branch that is merged into the deploy
environment. The Docker image runs gunicorn against `apex_dashboard:app`.

**Read the fail-closed note before your first deploy.** It will lock you out
if you skip the secrets step.

---

## 1. Pre-deploy checklist

### 1.1 Environment variables — REQUIRED

The app **fails closed**. If `APEX_PASS` is unset, a random password is
minted at boot and nobody can log in. If `APEX_USER`/`APEX_PASS` are the
documented `apex`/`admin`, login returns 503. This is deliberate
([ADR-004](../decisions/0004-fail-closed-authentication.md)); it is also the
single most likely way to lock yourself out.

Set these as Railway **service variables**, not in a committed file:

| Variable | Value | If missing |
|---|---|---|
| `APEX_SECRET` | `python -c "import secrets; print(secrets.token_hex(32))"` | random per process; sessions break if workers scale |
| `APEX_USER` | anything but `apex` | — |
| `APEX_PASS` | a strong password | **dashboard unreachable** |
| `APEX_PASS_HASH` | preferred over `APEX_PASS`, see below | — |
| `SUPABASE_URL` | project URL | falls back to `apex_dual_state.json` (ephemeral on Railway) |
| `SUPABASE_SERVICE_KEY` | service-role key | same |
| `APEX_ENV` | `main` for prod, distinct per preview | preview stomps prod state |
| `TRADING_AGENT_STORAGE_DIR` | `/data` with a volume mounted there | online RL updates are discarded on redeploy (see 1.4) |

To use a hash instead of a plaintext password:

```python
import secrets, hashlib
salt = secrets.token_bytes(16)
print(f"{salt.hex()}${hashlib.scrypt(b'YOUR-PASSWORD', salt=salt, n=2**14, r=8, p=1, dklen=32).hex()}")
```

Paste the output as `APEX_PASS_HASH`. It takes precedence over `APEX_PASS`.

**Verify before deploying.** If you have not confirmed these are set and are not
the defaults, do not deploy. The review that produced these changes flagged
this three times without ever being able to confirm it.

### 1.2 Secrets are not in the image

`.env` is gitignored and listed in `.railwayignore`. CI asserts the built image
contains no `.env`, `apex.log` or `apex_dual_state.json`. If that job ever goes
red, treat it as a credential exposure, not a lint failure.

### 1.3 The branch is green

CI must be green on the exact commit you are deploying. It runs: the test
suite, a production-import-order assertion for the JEV subsystem, `pip-audit`,
and a Docker build.

---

### 1.4 Persisting online RL updates — OPTIONAL, recommended

Without this, the app works exactly as before and the online learner still runs.
What changes is that **every weight update it makes is thrown away on the next
deploy**, because the container filesystem is ephemeral. The process restarts on
whatever weights are committed in the image.

The app says so once at boot, so you can tell which state you are in:

```
Online RL updates are writing to /app/trading_agent/agent/model, which is the
git-tracked artifact directory. They will be discarded on redeploy. Set
TRADING_AGENT_STORAGE_DIR to a mounted volume to make them persistent.
```

**To fix it:** add a Railway volume mounted at `/data`, then set
`TRADING_AGENT_STORAGE_DIR=/data`. Nothing else changes.

| | No volume | Volume at `/data` |
|---|---|---|
| App starts | ✅ uses the committed model | ✅ seeds the volume from the committed model, then uses it |
| Online updates | ✅ applied in memory, ❌ lost on redeploy | ✅ applied and persisted |
| Worktree on the host | ✅ stays clean | ✅ stays clean |
| Cost | — | volume's monthly cost |

How it works
([`artifact_store.py`](../../trading_agent/integration/artifact_store.py)):
an artifact has an immutable **baseline** — the model in the image, committed, so
a fresh deploy has something to load — and a mutable **runtime** copy that
updates land in. Reads prefer the runtime copy and fall back to the baseline.
Writes only ever touch the runtime copy, and go through a temporary file and a
rename, so a crash mid-save leaves the previous model intact rather than a
half-written zip. A freshly mounted volume is seeded from the baseline on first
use and is never re-seeded, so a redeploy cannot undo a deploy's worth of
learning.

Nothing is written outside `/data`, and no credential is involved. Rolling back
by redeploying an older image is unaffected: the volume keeps the newer model,
and the loader prefers it, so a rollback does **not** restore the older weights.
To roll the model back, delete the artifact on the volume or redeploy with
`TRADING_AGENT_STORAGE_DIR` unset.

**If you skip this**, the only consequence is that the model does not learn
across deploys. It is not a correctness or security problem, which is why it is
optional rather than a blocker.


### 1.5 Getting told when it stops — RECOMMENDED

Without this, an unattended bot can fail silently and you will not know until
you log in.

Set one variable:

| Variable | Value | If missing |
|---|---|---|
| `APEX_ALERT_WEBHOOK_URL` | a webhook URL | no alerts; a notice is logged once at boot |
| `APEX_ALERT_TIMEOUT` | seconds, default 5, capped at 30 | — |

Slack, Discord, ntfy and PagerDuty all accept a plain JSON POST, so a normal
incoming-webhook URL is enough. The payload is:

```json
{"event": "agent_cycle_failures", "level": "error",
 "message": "Agent has failed 12 consecutive cycle(s) — trading may be stalled",
 "occurrences": 12, "env": "main", "at": "2026-09-30T12:00:00+00:00"}
```

You are alerted for three conditions:

| Event | Meaning |
|---|---|
| `agent_cycle_failures` | Cycles are failing repeatedly. Trading is stalled. |
| `agent_recovered` | It came back. Worth knowing after a stall. |
| `agent_stopped` | The loop exited — either you stopped it, or the thread died. **The thread-dying case is why this exists**: a background thread that stops writes nothing anywhere, so the process keeps serving while no trading happens. |

A repeated event alerts at 1, 3, 10, 30 and 100 occurrences, then goes quiet.
That is deliberate — hundreds of identical notifications teach you to ignore the
channel. Recovery resets the counter, so the next incident's first failure is
never the silent one.

**For the case alerting cannot see**, `/healthz` reports whether the agent is
actually cycling. It needs no third party and no login:

```
GET /healthz   ->  200 when healthy, 503 when not
```

Point a free monitor at it (UptimeRobot, healthchecks.io, Better Stack — all
have free tiers) and email yourself. It is unauthenticated because a monitor has
no session, and it returns only liveness booleans and timings — no positions, no
ledger, no keys, and deliberately not the alerting error string, which would
carry the webhook provider's hostname.

Nothing here changes trading behaviour. With the variable unset the app behaves
exactly as before.

## 2. Deploying

1. Merge to the deploy branch. Railway rebuilds from the Dockerfile.
2. Watch the build. It resolves 22 exact pins; the RL stack is the slow part.
3. After the health check, open the dashboard and log in.
4. Confirm the JEV subsystem reports active:
   - `GET /api/jev/status` returns `"enabled": true`
   - the log contains `JEV subsystem loaded — risk gates active`
   - it does **not** contain `JEV subsystem unavailable`
5. If JEV failed to load, the risk layer is off. Do not trade. Check
   `/api/jev/status` for `import_error`.

---

## 3. Canary the first cycle — IMPORTANT

**The JEV risk gates switch from inert to live on the first cycle after this
deploy.** Before this change they were silently disabled, so the bot was
running with no regime overlay at all. This is the intended behaviour, but it
is a real behaviour change to a live system.

Watch one full cycle before leaving it unattended:

| Watch | Where | Expect |
|---|---|---|
| Regime calls | Decision Log tab, `JEV` category | one entry per open market per cycle |
| Gate application | `apex.log`, `[JEV] ... gates applied:` | only when a gate actually changes a value |
| Risk scaling | log line `→ risk_per_trade=` | base value; the gated value is logged separately |
| Peak portfolio | Summary cards | rises with the day, not instantly |
| Position count | Summary cards | respects `*_max_positions` |
| Circuit breaker | `apex.log`, `JEV circuit breaker` | should not trip; three consecutive failures open it |

If the regime overlay starts cutting size aggressively, check
`trend_strength` and `portfolio_stress` in the Decision Log before disabling
anything. The per-feature kill switches are `JEV_RISK_ENABLED`,
`JEV_TREND_ENABLED`, `JEV_ACTION_ENABLED`, `JEV_REGIME_ENABLED`,
`JEV_NEWS_ENABLED`, and `JEV_ENABLED`.

---

## 4. Rollback

Railway redeploys a previous successful build. There is no data migration in
this change, so rolling back the code does not require undoing state.

**If you roll back, the pre-fix build has the auth defect** — it falls back to
`apex`/`admin` when `APEX_PASS` is unset. Keep `APEX_PASS` set even on a
rollback.

To disable a failing feature without redeploying:

| Symptom | Action |
|---|---|
| JEV gating too aggressive | set `JEV_ENABLED=false`, restart |
| One JEV feature misbehaving | set the specific `JEV_*_ENABLED=false` |
| Bot unstable in-cycle | `POST /api/agent/pause` or `/api/agent/stop` |
| Trading halted for the day | expected: daily-loss or drawdown gate. Reset with `POST /api/reset/{market}` |

---

## 5. Observability worth knowing

- **Rotating log.** `apex.log`, 5 MB × 3. Old rotations are `.1`/`.2`/`.3`.
- **State persistence.** Supabase `apex_state`, one row per `APEX_ENV`. Look up
  `_STATE_ID` = `{APEX_ENV}-singleton`, `_CONFIG_ID` = `{APEX_ENV}-config`.
- **Agent health.** `GET /api/state` → `agent.status`. Values: `idle`,
  `running`, `paused`, `error`, `stopped`.
- **A stalled agent logs** `Agent has failed N consecutive cycles — trading may
  be stalled`. This is the alert to watch. A cycle failure does not kill the
  thread; it is retried after a 30-second backoff.
- **RL disabled indicators.** `No known vulnerabilities` aside, the RL path
  disables itself loudly if the feature contract or the model artifacts do not
  agree. Look for `Refusing to load the RL model` in the log.

---

## 6. Known gaps — read before trusting the security posture

These are real and not yet fixed. Tracked in
`docs/reasonix/plans/remaining-work.md`.

1. **RLS is disabled** on `apex_state` and `apex_rl_decisions`. The app uses a
   service-role key, which bypasses RLS regardless, so there is no
   database-layer isolation behind the application. A leaked service key is
   full read/write on the trading ledger.
2. **No CSRF tokens.** Mitigated by POST-only mutating routes and
   `SameSite=Lax`, and enforced by a test. Becomes exploitable the moment a
   mutating route answers GET.
3. **No branch protection.** CI reports; it does not block.
4. **The dashboard is internet-facing on a public Railway hostname.** IP
   allowlisting or a tunnel would be a meaningful addition.
