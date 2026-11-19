# RoboCasa365 submission preparation

This repository contains only **unchanged historical Harness 2.0 / Xiaomi**,
originally internal version 1.2.3.5 at
`c8911c64f24393e8665bd9ecc34f819173d9c3ba`, with its existing
**selected200 r002: 141/200 successes (70.5%)**. The root is directly installable;
there is no multi-edition catalog or source-selection step.

Start with the [2.0 packet](2.0/README.md), [result
report](../../results/2.0-selected200/REPORT.md), [JSON
draft](2.0/RoboHarness-Zero_2026_11_15.json) and [PR draft](2.0/PR_DRAFT.md).
The [project metadata](../../README.md) lists the submitting team, contact and paper authors. [BibTeX citation](../../CITATION.bib). The project source is
licensed under [Apache-2.0](../../LICENSE); [NOTICE](../../NOTICE) and
[third-party notices](../../THIRD_PARTY_NOTICES.md) retain upstream attribution.
Actual API connection details are supplied privately, outside the repository.

## Source and results

The source identity and runtime protocol remain original. Five canonical result
files preserve the selected attempts, four batches, excluded followup03 and ten
600-second planning overrides. The split rates are Atomic-Seen 62/72,
Composite-Seen 45/64 and Composite-Unseen 34/64. The historical panel has four
selected episodes per each of 50 tasks; it is not a completed official
2,500-rollout evaluation, full historical native replay or complete
workflow/canary/archive qualification. No other evaluation cohort is pooled into
these results.

From the source root, run `python3 tools/verify_results.py` to check fixed record
hashes and recompute the aggregate without models, credentials or a simulator.
The [packet setup guide](2.0/README.md) uses separate pinned coordinator, Xiaomi
policy and simulator environments. The setup instructions and public dependency locks are included in the source package.

## Applicable submission type and disclosures

The [official Agentic rules](https://github.com/robocasa-benchmark/leaderboard#agentic-submissions)
are the governing submission requirements. Harness planning, verification and
recovery suggest an Agentic entry; acceptance of novelty and eligibility remains
the organizers' decision. The [official leaderboard](https://robocasa.ai/leaderboard.html)
records reviewed entries and their disclosures.
[Xiaomi's RoboCasa evaluation setup](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/tree/main/eval_robocasa365)
is the relevant base-policy reference.

The owner requested the existing protocol. Native success ends physical execution;
the terminal status may then reach the Harness/model to complete `finish`
bookkeeping. It does not authorize further robot actions or alter the native
first-success score. RGB-D `back_project` grounding remains documented as part
of the observation scope. Preparation does not change the runtime.
The selected Xiaomi checkpoint and weights remain unchanged. Fixed capability
priors retain their original calibration provenance, and dynamic cross-episode
state has not been independently audited.

Historical configuration selected `codexgateway`, `gpt-6-astra`,
`reasoning_effort=xhigh` and `max_tokens=0`. The configured identity is not an
independently verified upstream provider identity. The API endpoint and key are supplied privately; only empty variable names are included in Git.

The [submission draft](2.0/RoboHarness-Zero_2026_11_15.json) includes numerical historical
split scores and scope disclosures. Final-format validation rejects `N/A`
scores and requires completed publication fields, including an arXiv abstract.
Draft validation is not organizer eligibility approval. Generic
[template JSON](agentic-submission.example.json) is a field reference only.

This source publication does not run a new evaluation, alter tasks or budgets,
replace fixed inputs, retry failed scenes, contact organizers or submit results.
