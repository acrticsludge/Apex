# ADR-003: Snapshot state under the lock, persist outside it

- Status: Accepted
- Date: 2026-09-29

## Context

Every call to `save_state()` ran inside `with _lock:`. `save_state` performs a
synchronous Supabase upsert with no timeout and no retry cap, falling back to a
JSON file write on failure.

Gunicorn runs `--workers 1 --threads 8` with a 120-second timeout. A single slow
Supabase round-trip therefore held the single state lock and stalled:

- all eight request threads, so the dashboard stopped responding, and
- the agent thread, mid-`apply_cycle`.

Worse, the stall could exceed the 120-second gunicorn timeout while
`apply_cycle` was part-way through mutating positions, killing a live trading
cycle. Eleven call sites were affected, including every ledger edit route and
the end-of-cycle save.

There was a second, quieter problem: a single slow query was indistinguishable
from a hung bot, and nothing surfaced which one was happening.

## Decision

**Hold the lock only for the mutation. Do the I/O outside it.**

```python
with _lock:
    _state[market]["positions"][sym] = {...}   # mutation only
    _snap = snapshot_state()                   # deep copy, still under the lock
persist_state(_snap)                           # blocking I/O, lock released
```

- `snapshot_state()` deep-copies via the same `json.loads(json.dumps(...))`
  round-trip `save_state` already used, so the writer can never observe a
  half-applied mutation.
- `persist_state()` is a thin, explicitly-named wrapper that documents the
  "lock must be released" contract at the call site.
- `tests/test_lock_io_separation.py` walks the AST of every `with _lock:` block
  and fails if a blocking persist call reappears inside one. It also uses a
  Supabase stub that records `lock.locked()` at the moment `execute()` is
  called, so the property is proven rather than assumed.

## Alternatives considered

**Move persistence to a background writer thread with a queue.** Rejected for
now: it changes durability semantics. A queued write can be lost on process
exit, and ordering between the queue and a later mutation needs care. That is a
real design, but it is a behaviour change that should be a deliberate decision
rather than a refactor. Revisit if persist latency ever needs to be hidden from
the HTTP response too.

**Shard the lock per market.** Rejected: it invites lock-ordering bugs between
the two markets, and buys little — the contention is with the I/O, not between
markets.

**Add a short timeout to the Supabase client and leave it under the lock.**
Rejected: it bounds the damage but keeps the architecture wrong. The lock should
protect state, not a network call.

**Replace Supabase.** Not on the table. The cost is the I/O, not the vendor.

## Consequences

- A slow persist now delays only the request that triggered it. The agent cycle
  and the other seven threads proceed.
- The state handed to the writer is a copy, so it costs one serialisation per
  persist. At this size that is sub-millisecond; the round-trip was orders of
  magnitude more expensive.
- The session-rotation path previously performed two `save_state()` calls in one
  lock acquisition; it now takes one snapshot and writes once.
- Startup `load_state()` / `load_cfg()` also moved outside the lock — nothing
  else is running at that point, so taking it bought nothing.
