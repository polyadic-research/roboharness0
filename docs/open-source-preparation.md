# Submission source boundary

This branch prepares the **RoboCasa365 / Xiaomi** portion of historical Harness **2.0**, internal/package version **1.2.3.5**, scientific source commit `c8911c64f24393e8665bd9ecc34f819173d9c3ba`.

The owner requested a focused submission branch. Unrelated integrations, frontend assets, demos, deployment tools and general framework documentation are removed. Required shared runtime dependencies remain when the selected CLI/evaluation path imports them. Retained scientific runtime, prompts, capability references and original release metadata keep their original bytes. This is a curated source subset; it does not claim to contain every file of the original full source snapshot.

The retained shared dependencies include the binding-identity helper in `rpent/policies/lerobot_smolvla_artifacts.py`, the catalogs imported by the frozen RoboCasa parser, and the Python evaluation repository/runtime-store modules imported by the CLI coordinator. These files retain their original bytes even where their names mention another integration. Independent non-Xiaomi policy services, other environment packages and browser assets are excluded. The supported entry point is the RoboCasa/Xiaomi submission runner.

`SOURCE_PROVENANCE.json` records original source identity. `publication/source-manifest.json` records the original frozen-file hashes, explicit exclusions and the current public payload. The historical full-snapshot digest identifies the original accepted snapshot; the current publication manifest inventories the curated tree separately. Original native-success handling, verification and RGB-D behavior remain unchanged.

## Verify and export

From the source root:

```bash
python3 tools/release.py verify
python3 tools/verify_results.py
python3 tools/release.py export --output /tmp/harness-2.0-submission-source
```

The exporter requires a new destination outside the checkout. It copies only registered verified files and excludes Git history, environments, caches, weights, assets and private configuration. Installation can use the source root directly; export is a maintainer packaging step.

This preparation branch still inherits the internal repository's Git history. Removing files from its current tree does not remove historical objects. Public hosting must start with new Git history in the verified export, without mirroring internal branches or tags. Other branches and worktrees remain outside this curation.

## Metadata and evidence scope

The project source is licensed under [Apache-2.0](../LICENSE), confirmed by the owner on 2026-11-17. Preserve the root [NOTICE](../NOTICE), [upstream attribution](../THIRD_PARTY_NOTICES.md) and all texts under `LICENSES/` in source and wheel distributions. The [license guide](license-status.md) covers external artifacts. The [project introduction](../README.md) lists the authors and team contact, and [CITATION.bib](../CITATION.bib) contains the paper citation. The API endpoint and key are supplied privately, outside the source package.

The selected200 panel has **141 successes and 59 failures (70.5%)**, with its original attempts, excluded followup03 and ten nested-planning overrides. It is a historical author-run panel. A completed official 2,500-rollout evaluation, historical trajectory replay and complete archive qualification are not claimed. The [submission packet](../evaluations/robocasa365/2.0/README.md) presents the original protocol differences for organizer review without claiming acceptance or a waiver.

Source curation and CPU checks do not authorize a new provider call, policy prediction, environment construction or evaluation.

## Refreshing reviewed publication files

Stage the reviewed changes, run `python3 tools/update_public_manifest.py`, then stage `publication/source-manifest.json`. The refresh verifies the original retained source bytes, explicit exclusion boundary and canonical result hashes before updating ordinary publication-file hashes. Run `python3 tools/release.py verify --staged` to check the exact files, contents and modes going into the commit. Run `python3 tools/release.py verify` to check the working tree before exporting.

## CPU verification

Use a Python 3.11 environment with the pinned coordinator dependencies and test tools:

```bash
python -m pip install --require-hashes -r evaluations/robocasa365/2.0/environments/coordinator.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -r requirements/test.txt
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/generate_harness_docs.py --check
```

Recording tests require FFmpeg from `imageio-ffmpeg`. If a system HTTP proxy is active, set `NO_PROXY=127.0.0.1,localhost,::1` and `no_proxy=127.0.0.1,localhost,::1` for the test process. The suite uses synthetic CPU fixtures.
