# Unchanged 2.0 / Xiaomi submission packet

The owner requested the existing **2.0 / Xiaomi code and selected200 r002 results**, with no protocol changes or new evaluation. The scientific source is `c8911c64f24393e8665bd9ecc34f819173d9c3ba`, digest `751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc`. Its original internal/package version remains **1.2.3.5**. The superseded `1836faea` timing-only derivative is historical. This packet has not been externally submitted or officially verified. The source publication is a curated RoboCasa/Xiaomi subset; the original scientific digest identifies the full historical source, not a claim that unrelated integrations are still bundled.

The [official Agentic rules](https://github.com/robocasa-benchmark/leaderboard#agentic-submissions) govern eligibility. [Xiaomi's evaluation directory](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/tree/main/eval_robocasa365) guides the policy/simulator split. See [candidate.json](candidate.json), the [result report](../../../results/2.0-selected200/REPORT.md), [submission draft](RoboHarness-Zero_2026_11_15.json) and [PR draft](PR_DRAFT.md). The [project metadata](../../../README.md) lists the team, contact and paper authors. [BibTeX citation](../../../CITATION.bib). The project source is licensed under [Apache-2.0](../../../LICENSE), with upstream attribution in [NOTICE](../../../NOTICE) and [third-party notices](../../../THIRD_PARTY_NOTICES.md). Native success stops physical execution before the terminal status is used for Harness/model `finish` bookkeeping. RGB-D and fixed-prior usage remain documented as part of the observation and method scope.

## Existing selected200 results

The panel has **141 successes / 59 failures, 70.5%**: Atomic-Seen **62/72 (86.111111%)**, Composite-Seen **45/64 (70.3125%)**, Composite-Unseen **34/64 (53.125%)**. It contains four selected episodes per each of 50 tasks, from fresh50, followup01, followup02 and followup04. Original attempts, exclusion of followup03 and ten 600-second nested-planning overrides are preserved. This is a mixed-parameter historical author-run panel, **not an official 2,500-rollout evaluation**. All200 workflow/archive closure is not claimed. Results from other source versions or pure-policy runs are not substituted.

Recompute the fixed records offline with `python3 tools/verify_selected200.py`.

## Inspect the source root

The repository root contains the curated RoboCasa/Xiaomi subset of the selected 2.0 source. Retained runtime files keep their original bytes; the publication manifest records exact exclusions. No export or edition selection is needed before installation. Inspect it directly:

```bash
python3 tools/release.py verify
python3 tools/verify_results.py
python3 tools/prepare_robocasa_submission.py check
python3 tools/setup_robocasa_submission.py --check-locks
```

`check` returns **2** while submission items remain unresolved. It reports original protocol differences and source hashes for organizer review rather than silently altering them. It is not a certification. `SOURCE_PROVENANCE.json` identifies this single source; `publication/source-manifest.json` defines the public file boundary. Maintainers can make an additional clean package with `python3 tools/release.py export --output /tmp/harness-2.0-submission`; this optional packaging step is not required to install or inspect the root.

## Isolated Linux setup

The setup target is Linux x86_64 with a CUDA 12.8-compatible NVIDIA driver and system EGL/OpenGL libraries. Install Git and `uv==0.13.7` first. Provisioning requires Internet access, model storage and RoboCasa assets. Python 3.11 coordinator/simulator and Python 3.12 policy environments are separate. PyPI dependencies are pinned transitively with hashes; both simulator Git revisions and the HF model revision are fixed. The exact upstream FlashAttention wheel is installed separately without dependency resolution. Model weights and assets stay outside the source tree.

Review the argument-array plan first; this works on Mac without provisioning:

```bash
python3 tools/setup_robocasa_submission.py \
  --prefix /tmp/harness-2.0-runtime \
  --download-assets --download-checkpoint
```

On the Linux GPU host, the same command plus `--apply` provisions the environments and downloads the pinned assets/checkpoint in one invocation:

```bash
python3 tools/setup_robocasa_submission.py \
  --prefix /tmp/harness-2.0-runtime \
  --download-assets --download-checkpoint --apply
. /tmp/harness-2.0-runtime/environment.sh
```

Use a new persistent prefix for an installation instead of temporary storage. Existing prefixes are refused and never removed; failed partial installs remain available for inspection. Provisioning records stay in that external runtime prefix. The installer checks imports and static metadata without inference or an episode; installation and CI run records are not source-package deliverables.

## Workload configurations and evaluation entry point

The configuration files separate existing results from future reviewer execution:

| File | Purpose |
| --- | --- |
| [historical-selection.json](historical-selection.json) | Fixed r002 record hashes, four selected batches, split denominators and the excluded followup03. |
| [historical-parameters.json](historical-parameters.json) | Original planner configuration and all ten 600-second overrides, bound to the original selected attempt IDs. |
| [evaluation-profile.json](evaluation-profile.json) | Future reviewer workload from the frozen `robocasa365-leaderboard-v1` profile: 50 tasks, 50 rollouts each, `pretrain`, environment sequence seed 7 and original task horizons. |
| [planner.json](planner.json) | Configured planner identity, supported CLI settings, private API variable names and historical service-side settings. |

The historical manifests have no executable replay path. Original saved states
and policy RNG records are external; scene IDs, episode indices and seed 7 cannot
reconstruct them. No runner generates fresh resets and labels them historical.
The submission continues to report the original selected200 results.

Inspect a deterministic argument-array plan, without credentials, runtime
installation, simulator startup, policy prediction or provider calls:

```bash
python3 tools/run_robocasa_submission.py \
  --runtime-prefix /path/to/installed/harness-2.0-runtime \
  --output-dir /path/to/existing-parent/new-reviewer-evaluation
```

The command defaults to plan-only mode and writes no evaluation directory.
The profile owns tasks, seed, split, episode count and horizons. It selects the
unchanged Xiaomi policy with one worker, `recording-mode=none` and the original
`robocasa-horizon-gated-v1` planning behavior. The two object registries remain
`lightwheel,objaverse`; install all required assets before execution.

For a separately requested future run on the intended Linux x86_64 GPU host,
provide private `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment values and add
the explicit execution flags:

```bash
/path/to/installed/harness-2.0-runtime/coordinator/bin/python tools/run_robocasa_submission.py \
  --runtime-prefix /path/to/installed/harness-2.0-runtime \
  --output-dir /path/to/existing-parent/new-reviewer-evaluation \
  --execute --api-settings-confirmed
```

`--env-file /path/to/private/planner.env` can supply those two names explicitly
instead of exporting them. Use the installed coordinator interpreter, which
includes the pinned dotenv dependency. The wrapper reads that file only when execution is
requested; it does not print endpoint or key values. Runtime credentials are not
part of public JSON, command arguments, source or evidence archives.

The confirmation flag acknowledges the privately supplied API model/settings. The endpoint and key are configured outside the repository. Historical controls remain `gpt-6-astra`, service-side `xhigh`, a 330-second HTTP timeout, 180-second nested planning and an unset outer solve timeout; the [Planner API guide](PLANNER.md) defines the client interface.

Execution requires a new output directory and refuses partial or prior outputs.
It makes one original CLI invocation, with no automatic retry, reset, resume or
replacement of historical attempts. A process exit alone does not establish
native completion or qualification; inspect its retained terminal evidence and
qualification report. No new run has been launched by preparing these commands.

## Recompute and archive existing evidence

```bash
python3 tools/verify_results.py
python3 tools/run_robocasa_submission.py \
  --archive-historical-results /path/to/new/harness-2.0-selected200.tar.gz
```

The archive operation checks canonical hashes and derived manifest consistency,
then writes a new deterministic small-evidence archive. It refuses an existing
file and prints the archive SHA-256 plus per-file hashes. It includes the five
original records, maintained report, historical manifests, public planner identity
configuration/evidence, source provenance and offline verifier code. It contains no raw logs, images, states, weights or private
environment data, and does not claim to reconstruct historical trajectories.

## Submission JSON and numeric results

The `submission` object in [candidate.json](candidate.json) uses the current Agentic shape and numerical selected200 split percentages, with the historical scope and unchanged protocol in its notes. It omits training configuration because the base weights are unchanged. Complete the arXiv URL, public release SHA and actual submission date. Remove the `PREPARATION DRAFT, NOT SUBMITTED.` label from the final notes and retain the selected200 disclosures. Configured historical planner identity is distinguished from independently verified provider identity; dynamic cross-episode state has not been independently audited. Credentials follow the official private process.

Generate the checked-in draft from the verified historical records:

```bash
python3 tools/prepare_robocasa_submission.py selected200-json \
  --draft --output /tmp/RoboHarness-Zero_2026_11_15.json
```

Draft mode permits unresolved publication metadata and never means ready to submit. Once metadata is finalized, omit `--draft` and supply `--metadata /path/to/completed-metadata.json` to generate a final-format file. Final validation rejects `N/A` scores and requires an arXiv abstract. It does not certify scientific eligibility.

The inherited `--submission-metadata-json` path and `robots/robocasa/submission.py` retain an older leaderboard shape (`policy_family`, `submission_source`). They remain frozen for source provenance. **Use the preparation converter for current Agentic JSON.** Do not pass the new candidate object to that old CLI option.

Validate a completed metadata file with:

```bash
python3 tools/prepare_robocasa_submission.py validate-json \
  --input /tmp/submission-metadata.json
```

The historical selected200 converter is separate from the complete official-profile converter below. Do not relabel selected200 as a completed 50-task / 50-rollout evaluation. If a future separately authorized run produces complete official-profile results, recompute its actual batch ledgers with:

```bash
PYTHONPATH=. python3 tools/prepare_robocasa_submission.py convert-results \
  --input /path/to/evaluation \
  --metadata /tmp/submission-metadata.json \
  --output /tmp/harness-2.0-agentic.json
```

The converter checks the clean source commit, checkpoint, exact task/seed/horizon profile and complete accepted batch evidence. It ignores a hand-edited `official_metrics.json`, preserves failures in the denominator and refuses an existing output file. Its numeric export requires all 2,500 accepted rollouts; this is a local evidence rule for this export profile. Schema-valid JSON and recomputed metrics do not establish Agentic eligibility. No result or external submission is created by source preparation.

Accepted raw results contain canonical absolute artifact paths. Read or mount an archived evaluation at its recorded `runtime.evaluation_root`; the converter refuses to silently rewrite evidence after relocation.

## Unchanged protocol disclosed for organizer review

The source audit identifies three connected issues:

| Source path | Finding | Disclosure |
| --- | --- | --- |
| `rpent/context/projectors.py` | Compact tool context can include `environment.success`, including false values. | Original planner-visible native success feedback is retained. |
| `rpent/harness/hooks.py` | Native success can authorize verification and appear in feedback. | Original success-dependent verification and control are retained. |
| `robots/robocasa/toolkit.py` | `back_project` exposes RGB-D grounding beyond the base policy observations. | Original depth and derived coordinates are retained. |

The owner explicitly requested the original protocol. This packet preserves those behaviors and presents their differences for organizer consideration; it does not claim compliance with a policy-observations-only/no-success-feedback rule. `check` keeps unresolved findings visible. Fixed capability priors retain their original calibration provenance; no new timing-only priors are added. No new native run or protocol modification is part of this preparation.

For later request-boundary auditing, the scanner accepts JSONL records with `type: "model_request"` and a `request` object containing the complete normalized messages, tools and tool responses actually sent to the provider:

```bash
python3 tools/prepare_robocasa_submission.py audit-trace \
  --input /path/to/normalized-model-requests.jsonl
```

It rejects empty/unlabelled traces and reports known success/depth leaks, including nested JSON strings. It does not prove complete request coverage or inspect pixels, filesystem access, SDK internals or indirect control dependencies. Native SDK tool-delivery snapshots are tool responses, not complete provider histories. A clean scan alone must never be used as an eligibility certificate.
