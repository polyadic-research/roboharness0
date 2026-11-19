# Harness 2.0 selected200 results

This repository includes the existing **Harness 2.0 / Xiaomi selected200 r002**
panel: **141 successes, 59 failures, 70.5%**. The source is unchanged original
1.2.3.5 at `c8911c64f24393e8665bd9ecc34f819173d9c3ba`.

| Split | Successes | Accepted scenes | Success rate |
| --- | ---: | ---: | ---: |
| Atomic-Seen | 62 | 72 | 86.111111% |
| Composite-Seen | 45 | 64 | 70.3125% |
| Composite-Unseen | 34 | 64 | 53.125% |

The [result report](2.0-selected200/REPORT.md) describes the exact four selected
batches, original attempts, excluded followup03, ten 600-second planning
overrides, fixed priors and archive-qualification limits. The five canonical
files in that directory retain their original bytes and provenance paths.

From the source root, recompute the published aggregate without credentials,
weights or a simulator:

```bash
python3 tools/verify_results.py
```

This read-only wrapper calls the selected200 verifier, which checks fixed
SHA-256 digests before recomputing the accepted outcomes. It does not re-open
raw archives or execute a native trajectory. These are historical author-run
results, not an official completed 2,500-rollout evaluation or a claim of full
workflow/canary/archive qualification. No other source version's results are
included or pooled into this denominator.
