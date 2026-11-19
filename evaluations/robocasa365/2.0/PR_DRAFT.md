# [AGENTIC] RoboHarness-Zero Submission

Leaderboard file for the current draft date: `submissions/agentic/RoboHarness-Zero_2026_11_15.json`. For the final PR, use the submission JSON’s `date` for the filename suffix in `YYYY_MM_DD` format.

Polyadic Research Group prepared this submission for *RoboHarness-Zero: Any Frozen VLA, Zero Per-Task Preparation*. The paper URL is `<ARXIV_URL>` and the public release commit is `<PUBLIC_COMMIT_SHA>`. [Source repository](https://github.com/polyadic-research/roboharness0) · [BibTeX citation](../../../CITATION.bib). The project source is licensed under [Apache-2.0](../../../LICENSE); [NOTICE](../../../NOTICE) and [third-party notices](../../../THIRD_PARTY_NOTICES.md) retain upstream attribution. No leaderboard PR has been opened.

## Source and method

RoboHarness-Zero coordinates language planning, a fixed robot policy, verification and bounded recovery. The selected source is unchanged Harness 2.0, originally internal version 1.2.3.5, commit `c8911c64f24393e8665bd9ecc34f819173d9c3ba`, digest `751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc`. Base policy: `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, HF revision `0d1aa76d0d82debc9b611e4d1e231096434d5be4`, Apache-2.0, no weight edits. Runtime, prompt and capability-reference bytes remain original.

## Historical author-run results

selected200 r002 contains four episodes per each of 50 tasks: **141/200 successes (70.5%)**. Split rates: Atomic-Seen **62/72 (86.111111%)**, Composite-Seen **45/64 (70.3125%)**, Composite-Unseen **34/64 (53.125%)**.

The selected attempts come from fresh50, followup01, followup02 and followup04; followup03 is excluded. Ten selected episodes have a 600-second nested-planning override. Original attempt identities and mixed-parameter history remain in the attached index. This is not a completed official 2,500-rollout evaluation, a uniform unchanged-180-second panel or a substituted pure-policy result. All200 workflow/archive closure is not claimed.

## Original protocol disclosures

- Before success, the planner can see `environment.success=false` in tool responses. Native success stops physical execution. The terminal status can then reach the planner for `finish` bookkeeping.
- RGB-D `back_project` grounding remains enabled beyond the selected base policy's RGB interface.
- Fixed capability priors and their original calibration references remain. Dynamic cross-episode state has not been independently audited.
- Historical configuration identifies OpenAI Codex SDK via `codexgateway`, `gpt-6-astra`, reasoning effort `xhigh` and `max_tokens=0`; upstream provider/model identity was not independently verified.

We request organizer consideration of the original system and historical evidence, with these differences disclosed. No compliance approval or exemption is claimed. No protocol was changed to prepare this packet.

## Reproduction and access

The source includes separate coordinator, policy and simulator setup, pinned revisions, offline recomputation and per-attempt provenance. Raw native logs, images, states and weights are external; small result summaries and indexes are included.

The maintainer supplies the planner API endpoint and key privately. Configure the two API environment variables as described in the [Planner API guide](PLANNER.md); private values and service deployment files are outside this source package.

The source publication is the curated RoboCasa/Xiaomi subset of the original 2.0 source. Retained runtime and model-bound inputs preserve original bytes; the publication manifest records the fixed exclusion list. Other integrations, frontend assets, demos and deployment tooling are omitted.
