# ADR-C3: Global batch as a pure function of (seed, step)

Status: accepted (2026-10-22)

## Decision
The sampler draws the global batch of step `s` from a generator seeded with `(seed, s)`;
each rank/micro-batch takes a fixed slice. No data-loader state is checkpointed.

## Why
* Equivalence tests across world sizes need identical global batches.
* Resume needs only the step counter: no iterator state to serialize or reshard.

## Trade-off
Sampling is with replacement across steps (random windows), not epoch-exact. For a
systems lab this is acceptable; an epoch-based pipeline would checkpoint a permutation and
offset instead.
