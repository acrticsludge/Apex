# ADR-005: Pin every dependency exactly, constrained by pandas-ta

- Status: Accepted
- Date: 2026-09-29

## Context

`requirements.txt` had 17 of 19 entries as a bare `>=`. Every Railway rebuild
resolved the tree fresh, so a yanked or compromised transitive release could
land in production with no review gate. There was no lockfile, no Dependabot,
and no CI.

Pinning was then introduced, and immediately exposed that pinning correctly is
not the same as pinning arbitrarily.

## The constraint chain

`trading_agent/data/indicator_engine.py` builds the RL feature set with
`pandas-ta`, pinned at `0.4.71b0` because that is the version the trained
feature columns were computed with. That release is unmaintained, has no 1.x,
and drags in a hard chain:

```
pandas-ta 0.4.71b0
  ├── numba == 0.61.2        (exact)
  │     └── numpy <2.3, >=1.24
  └── numpy >=2.2.6
                    ▲
                    └── the only valid window is [2.2.6, 2.3)
```

The first pin attempt used `numpy==2.4.6` — the version in the author's local
virtualenv, which had never been reinstalled. All three CI jobs failed with
`ResolutionImpossible`.

Two further constraints only surfaced by running things:

- **pandas must stay on 2.x.** `pandas-ta 0.4.71b0` predates pandas 3, and the
  2.x line is the one its `>=2.3.2` floor was written against.
- **scikit-learn must match the committed artifact.** `scaler.joblib` was
  pickled by 1.6.1. Loading it under 1.7.2 works but emits
  `InconsistentVersionWarning` and sklearn's own words: it "might lead to
  breaking code or invalid results". For a scaler that means silently wrong
  features feeding the policy, not a crash. Pinned to 1.6.1.

A separate audit pass found 20 known vulnerabilities in versions pinned in the
previous commit: eight in `torch 2.8.0`, and twelve in `starlette 0.48.0` —
the latter pulled in transitively because `fastapi 0.118.3` caps
`starlette<0.49.0`.

## Decision

**Every entry is an exact pin. Constraints discovered at build time are
recorded as tests, not as comments.**

1. `requirements.txt` pins all 22 entries exactly, with the serving and RL
   stacks in labelled sections.
2. The numpy window, the pandas major line, and the scikit-learn match to the
   artifact are each asserted by a test in `tests/test_supply_chain.py`. A
   well-meaning dependency bump that reintroduces any of them fails the suite
   rather than the build.
3. `requirements-dev.txt` carries test tooling, so the deploy image never
   installs it.
4. Dependabot is configured for pip and GitHub Actions, grouped so a routine
   patch wave arrives as one reviewable diff. The `torch-stack` and
   `data-stack` groups exist because those are the ones with real constraint
   edges.
5. CI prints the resolved versions on every run, so a future conflict is
   diagnosable from the log without reproducing it.
6. `pip-audit` runs in CI and **fails the job** on any fixable advisory. It
   passes only `PYSEC-2026-139`, which has no fixed release on any torch
   version. The previous `|| echo warning` swallowed the exit code entirely,
   which is how twenty advisories produced a green tick.

## Alternatives considered

**Keep ranges and add a lockfile (`uv.lock` / `pip-compile`).** Attractive, and
probably the right long-term answer: a lockfile is hash-verified and resolves
the transitive graph properly. Not adopted here because it adds a tool the
repository does not otherwise use and the constraint edges are not expressible
in `requirements.txt` at all. Worth revisiting as a standalone change.

**Drop `pandas-ta` and reimplement the indicators.** The only way to escape the
numpy pin. Rejected: it would change every indicator value the model was
trained on, which is a retraining project, not a dependency bump.

**Use `torch<2.9` to dodge the advisories.** The audit says 2.9.0+ for most, so
that is backwards.

## Consequences

- Dependency upgrades are now a deliberate act. That is the point, but it means
  the Dependabot grouping matters: a small group that is noisy gets ignored.
- Python is now pinned to 3.12 in three places, because `pandas-ta` requires
  it. `setup.py` declares `python_requires=">=3.12"` and a test asserts the
  Docker image and CI agree. Without this, installing on 3.11 fails with a numpy
  resolution error pointing at entirely the wrong cause.
- The serving stack can be reinstalled on any machine. The RL stack needs ~4 GB
  for torch.
- A dependency bump that breaks the numpy window now fails in ~15 seconds
  rather than at deploy time.
