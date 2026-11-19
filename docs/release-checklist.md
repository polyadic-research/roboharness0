# Public repository and submission checklist

## Public repository

- [x] Curate the RoboCasa365 / Xiaomi subset of historical Harness 2.0 at `c8911c64`; preserve frozen runtime and model-bound input bytes.
- [x] Include Apache-2.0 LICENSE, NOTICE and upstream license texts; align package metadata and distributions.
- [x] Include pinned installation tools, configuration examples and client API instructions.
- [x] Preserve selected200 r002 records and the offline 141/200 verifier, including original attempts, followup03 exclusion and ten planning overrides.
- [x] Record the submitting team, contact, paper title and authors in the project metadata and publish the [BibTeX citation](../CITATION.bib).
- [x] Exclude actual API endpoints/keys, private configuration, bulk data and internal Git history from the verified source export.
- [x] Provide English and Chinese READMEs; the main README includes the project introduction, contributions, results, supervision flow, verification and evaluation entry points.
- [ ] At release, initialize the public repository from the verified export and record its exact release commit SHA. [Public repository](https://github.com/polyadic-research/roboharness0).
- [ ] Optional: add architecture figures, representative video and project branding.

## Where each publication field is used

| Field | Paper | Public repository | Project website | Leaderboard submission |
| --- | --- | --- | --- | --- |
| Team / submitter | — | README and BibTeX team note | — | JSON `submitter` and PR body |
| Contact email | — | README (optional) | — | — |
| Paper title and authors | Title page | README and BibTeX | — | — |
| Publication year | — | BibTeX | — | — |
| arXiv URL | — | README and BibTeX | Paper link | JSON `paper_link` and PR body |
| Public repository URL | Code link in the abstract | README | Code link | JSON `code_url` and PR body |
| Public release commit SHA | — | Record at release | — | JSON `commit_hash` and PR body |
| Submission date | — | — | — | JSON `date` and filename suffix |

A dash means the current material has no dedicated field. The website also reserves a BibTeX value for a future citation display.

## Formal submission

- [x] Prepare current Agentic JSON and PR drafts with the fixed source/checkpoint/license and historical Planner settings.
- [x] State the author-run selected200 scope and exact split denominators; do not relabel it as a completed official 2500-rollout result.
- [x] Disclose that the planner can see `environment.success=false` before terminal success. Native success stops physical execution, and the terminal status may then reach the planner for `finish` bookkeeping.
- [x] Document private API access through empty environment-variable names and the client interface; no gateway deployment is required in the source package.
- [ ] Fill in the arXiv URL, public release commit SHA and actual leaderboard submission date. Remove the draft label from the final notes and retain the selected200 disclosures.
- [x] Use the [PR title](../evaluations/robocasa365/2.0/PR_DRAFT.md) `[AGENTIC] RoboHarness-Zero Submission` and name the leaderboard JSON `submissions/agentic/RoboHarness-Zero_YYYY_MM_DD.json` using its `date` field.
- [ ] Optional: add a square leaderboard icon.
- [ ] Publish or submit only when the owner asks to proceed.

API service deployment, availability/quota receipts, new Linux/GPU qualification records and hosted-CI run records are outside the current repository preparation scope. Existing installation tools and the CI workflow remain available. This checklist does not launch new evaluations or require a new official 2500-rollout run as an automatic preparation step.
