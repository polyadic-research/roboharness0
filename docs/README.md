# Submission documentation

This branch contains focused documentation for the RoboCasa365 / Xiaomi 2.0 submission:

- [Installation](policy-setup.md): pinned environments, assets/checkpoint setup and import/static checks.
- [Evaluation and results](../evaluations/robocasa365/2.0/README.md): default-plan-only entry point, future workload and historical selected200 evidence.
- [Planner API](../evaluations/robocasa365/2.0/PLANNER.md): public client settings, private API configuration and historical evidence.
- [Project metadata](../README.md) and [BibTeX citation](../CITATION.bib).
- [Preserved RoboCasa protocol](robocasa365.md): execution and observation scope.
- [Runtime reference inputs](runtime-reference-data.md): fixed Xiaomi capability-prior provenance.
- [Source boundary and export](open-source-preparation.md): curated source, verification and publication history.
- [License status](license-status.md) and [third-party notices](../THIRD_PARTY_NOTICES.md): confirmed Apache-2.0 source license and upstream attribution.
- [Release checklist](release-checklist.md): validation and remaining handoff fields.
- [Harness authority](harness/README.md) and [change checklist](harness/change-checklist.md): retained lifecycle invariants and generated references.

The repository [English](../README.md) and [Chinese](../README.zh-CN.md) introductions provide the same submission entry points. The former general framework/Sphinx books, other robot guides, frontend demos and deployment manuals are excluded from this branch's submission scope. Maintained Markdown links and the publication file boundary are checked by `python3 tools/release.py verify`.
