# Harness 2.0 evaluation evidence

The included evaluation evidence is the existing **selected200 r002** panel for
unchanged historical Harness 2.0 / Xiaomi, originally internal version 1.2.3.5
at `c8911c64f24393e8665bd9ecc34f819173d9c3ba`. The panel contains four selected
episodes per each of the 50 tasks, with 141 successes and 59 failures.

The [result report](../results/2.0-selected200/REPORT.md) and five canonical source
records retain the selected attempts and accepted outcomes. Fresh50, followup01,
followup02 and followup04 remain included; followup03 remains excluded. Ten
selected attempts have the original user-authorized 600-second nested-planning
timeout override. This is a mixed-parameter historical summary, not a uniform
180-second run or a completed official 2,500-rollout evaluation.

Initial XML/state payloads, original prepared runtime overlays, policy RNG
snapshots, raw receipts, input images and physics captures are external. Source
and aggregate verification alone do not reproduce the original trajectories or
establish complete archive qualification. The fixed record hashes and result
denominators can be checked offline with:

```bash
python3 tools/verify_results.py
```

The [submission packet](robocasa365/2.0/README.md) includes the original source
identity and explicit publication exclusions, setup, numerical draft and explicit protocol disclosures. Planner-visible
native success feedback, success-dependent verification and RGB-D grounding are
retained for organizer review. No compliance waiver is claimed. Team and paper
metadata will be completed later.

The inherited CLI/evaluation interfaces remain available for separately configured
work. Publication does not launch an evaluation, resample scenes, retry failures,
change scientific inputs or alter budgets. Future evaluation scope and parameters
must be explicit rather than inferred from these historical records.

## Configuration and entry points

- [Historical selection](robocasa365/2.0/historical-selection.json): fixed evidence hashes and original selected batches, with `replay_supported=false`.
- [Historical parameters](robocasa365/2.0/historical-parameters.json): exact selected timeout overrides and attempt IDs; original state/RNG evidence remains external.
- [Future reviewer profile](robocasa365/2.0/evaluation-profile.json): the frozen 50-task, 50-rollout `pretrain` workload, with original task horizons and seed-7 reset sequence.
- [Planner configuration](robocasa365/2.0/planner.json): public settings and private environment names, without credentials.

`tools/run_robocasa_submission.py` prints an argument-array plan by default and
does not import or start the simulator, policy or planner. Future execution needs
`--execute`, private planner environment values, installed Linux/GPU environments,
a new output path and `--gateway-settings-confirmed` as a reviewer declaration.
It does not run historical replay or overwrite/retry existing evaluations.
See the [packet guide](robocasa365/2.0/README.md) for complete commands and the
distinction between gateway declarations and independent identity audits.

The same entry point's `--archive-historical-results` option creates a new
deterministic archive of the small public evidence and offline verifier after
checking the original hashes. It excludes raw logs, weights, states, images and
credentials. This operation does not create a new evaluation result.
