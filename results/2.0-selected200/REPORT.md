# Harness 2.0: unchanged code and historical selected200 results

This package prepares the owner's selected historical results for submission review.
It preserves the runtime protocol, policy checkpoint, model-bound reference inputs,
selected attempts and original outcomes. It is not an official full-benchmark score
or a claim that the unchanged system satisfies every current Agentic requirement.

The current display name **Harness 2.0 / Xiaomi** identifies original Harness
**1.2.3.5**, source commit `c8911c64f24393e8665bd9ecc34f819173d9c3ba`, source digest
`751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc`.
Its internal source/package version remains unchanged. The superseded
`1836faeaa909edf328d6f445f546a40775064264` timing-only derivative is historical.

## Recomputed results

The r002 index retains 200 unique scenes: four episodes for each of the 50 tasks.
Atomic, composite-seen and composite-unseen correspond to the preserved A, C and U
scene categories. All 200 selected records have accepted native scores; attempts
and repeats do not add scenes to the denominator.

| Split | Accepted scenes | Successes | Failures | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Atomic-Seen | 72 | 62 | 10 | 86.111111% |
| Composite-Seen | 64 | 45 | 19 | 70.3125% |
| Composite-Unseen | 64 | 34 | 30 | 53.125% |
| Overall | 200 | 141 | 59 | 70.5% |

| Original selected batch | Scenes | Successes | Failures |
| --- | ---: | ---: | ---: |
| fresh50 | 50 | 34 | 16 |
| followup01 | 50 | 36 | 14 |
| followup02 | 50 | 34 | 16 |
| followup04 | 50 | 37 | 13 |

These are author-run historical selected200 rates, not the official 50-task,
50-rollout-per-task evaluation. They must not be described as 2,500 completed
rollouts or replaced by pure-policy paired250 results. Followup03 remains excluded
only from this selection; its original results and evidence remain historical.

The original selection includes ten user-authorized 600-second nested-planning
timeout overrides: U11-04, U12-02, C10-03, U04-03, U10-03, U11-03, U13-03,
U08-04, U12-04 and U13-04. All accepted supplemental outcomes were adopted,
including failures; an unscored attempt did not erase a prior accepted result.
This is the original mixed-protocol supplemental summary and is not a uniform
180-second run. Original attempts, failures, source records and budgets remain
unchanged.

## Planner and retained knowledge disclosures

The original recorded job configuration selected `gpt-6-astra`,
`reasoning_effort=xhigh`, `max_tokens=0`, an API request timeout of 330 seconds,
and the `codexgateway` transport. Zero `max_tokens` is the workflow's configured
no-output-token-cap setting, not a claim about provider-internal limits.
The configured provider path is the OpenAI Codex SDK through CodexGateway;
the [planner evidence](../../evaluations/robocasa365/2.0/planner-evidence.json)
records the historical source/receipt basis for that configured identity.
The retained fresh50 audit leaves independent upstream provider/model attestation
unset. Configuration evidence is distinct from such independent attestation.
The public [planner configuration](../../evaluations/robocasa365/2.0/planner.json)
contains no credential or reviewer test key. Reasoning effort is configured on
the gateway/server; the original client does not transmit that setting.

The unchanged Xiaomi capability guidance and capability JSON are model-bound fixed
priors. Their own provenance records measurements on 65 atomic tasks with 15
episodes each and five composite tasks with five episodes each, dated
2026-10-12–2026-10-13. This calibration is separate from selected200; the presence
of fixed priors must be disclosed when discussing memory or generalization.
This package does not certify the absence of all cross-episode state. In
particular, `stall-timing-original975.json` from the later timing-only derivative
is not an input of the current historical 2.0 source.

Before success, tool responses can expose `environment.success=false` to the planner.
Native success terminates the physical action stream. Its terminal status can
then be returned for Harness verification and model `finish` bookkeeping; this
post-terminal notification does not change earlier actions or the native
first-success score. The retained RGB-D grounding paths remain documented as
part of the observation scope. These explanations do not modify scientific source
or original result records.

## Evidence and offline verification

The five files alongside this report are byte-exact copies of the canonical
historical identity package: `README.zh-CN.md`, `version-identity.json`,
`verification.json`, and the original r002 summary and selected-attempt index.
Historical machine paths and the original README's deployment references are
provenance, not current installation links. The original README is frozen evidence;
use this report for maintained navigation.

Run from the source root:

```bash
python3 tools/verify_selected200.py
```

The verifier checks fixed SHA-256 digests before recomputing accepted outcomes,
split denominators, original batches and timeout-override disclosures. It rejects
changed source identities and replaced result records. Its split rates retain
full numerical precision for preparation tools.

This offline check does not read back raw archives or claim a new native replay.
The original summary explicitly says all200 full-workflow/canary/archive
qualification is incomplete; its recorded verified full-archive scope is 150
cases. Those historical statements remain intact. No policy predictions, model
requests, environment constructors, new episodes, retries or budget changes are
performed by preparing or verifying this package.

## Historical manifests and evidence archive

The [historical selection manifest](../../evaluations/robocasa365/2.0/historical-selection.json)
points to the exact original records and batch/split counts. The
[parameter manifest](../../evaluations/robocasa365/2.0/historical-parameters.json)
binds each of the ten overrides to its original selected attempt. They describe
existing evidence and explicitly do not support reconstructing saved states or
policy RNG from scene IDs or integer seeds.

To create a new portable archive of the small evidence only:

```bash
python3 tools/run_robocasa_submission.py \
  --archive-historical-results /path/to/new/harness-2.0-selected200.tar.gz
```

The operation verifies fixed record hashes and manifest consistency, preserves
the five canonical files byte-for-byte, includes public planner identity/configuration
evidence and offline verifier code, and emits
archive/member SHA-256 digests. It refuses to overwrite an existing archive.
Raw logs, images, physics states, weights and private environment values remain
external. The separate future reviewer workload is defined in
`evaluations/robocasa365/2.0/evaluation-profile.json`; it is not selected200 replay
or a replacement result.
