# Harness governance

`rpent/harness/lifecycle.py` is the authority for externally visible states, events, legal transitions and allowed operations. The [generated lifecycle reference](lifecycle.md) and [diagram](lifecycle.mmd) are derived review artifacts. Internal mechanisms remain owned by `PlanStateMachine`, `OutcomeGraph`, `VerificationCoordinator`, `RecoveryTracker` and `StageRuntime`.

This guide describes unchanged Harness 2.0 / Xiaomi, originally internal version 1.2.3.5. `SOURCE_PROVENANCE.json` records its scientific source and `publication/source-manifest.json` records the public file boundary. Historical development diaries and other source versions are outside this distribution.

## Execution and recovery boundaries

`start_instruction` is a one-shot pre-lifecycle bootstrap. State remains unstarted (`None`) until successful plan registration; failed bootstrap leaves it unstarted, and a later start is rejected after lifecycle entry.

Normal execution requests recovery from `action_required` or `verification_required` through `request_exception`. `RecoveryTracker` selects the recovery scope and budget; the LLM cannot select them directly.

At an `observation_required` stage boundary, only `submit_observation` may
atomically generate and advance the next stage; `request_exception` is not
available there. `apply_recovery` is the agent-facing recovery mutation. Compatibility implementation details do not define additional public operations.

## Evidence and completion

- Original outcomes remain immutable and cannot be silently completed.
- Advisory, stale, uncertain or conflicting evidence cannot authorize success.
- Satisfied outcomes awaiting authoritative final verification remain distinct from uncertain outcomes.
- Waiting for predicate verification cannot silently execute another physical action.
- Remaining-stage repartition covers every unresolved outcome exactly once.
- `finish(success)` requires `contract_success=true`; exhausted recovery fails closed.

Policy capability tables and guides are [frozen runtime reference inputs](../runtime-reference-data.md), not new result claims. The [selected200 results](../../results/README.md) retain their original selection and parameter overrides. Original native-success feedback, success-dependent verification and RGB-D grounding remain enabled and disclosed for organizer review; this publication does not claim an official protocol exemption.

## Making a behavior change

Follow the [change checklist](change-checklist.md): change the authority module first, add legal/illegal transition and invariant tests, update consumers and regenerate documentation. Run the complete CPU suite, Ruff on changed Python and `git diff --check`. A deployed smoke is required for behavior changes, and paired tasks/seeds are required before claiming a success-rate improvement.

Publication curation preserves the bytes of retained runtime/model-input files and records excluded source paths explicitly. It does not authorize new robot or GPU runs.
