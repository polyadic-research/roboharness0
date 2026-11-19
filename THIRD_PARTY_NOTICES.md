# Third-party notices

RoboHarness-Zero source, including project additions, is distributed under [Apache-2.0](LICENSE), as confirmed by the owner on 2026-11-17. The root [NOTICE](NOTICE) identifies upstream contributions. This file supplies source links and the license boundaries for the curated RoboCasa365 / Xiaomi edition.

## Retained source and benchmark interfaces

| Project | Material used here | Upstream license and source |
| --- | --- | --- |
| [RPent](https://github.com/RLinf/RPent) | Retained framework, `rpent` namespace and robot integration lineage; extended by RoboHarness-Zero. | [Apache-2.0 text](LICENSES/RPent-Apache-2.0.txt), retained verbatim. Existing upstream notices remain intact. |
| [RoboCasa365](https://github.com/robocasa/robocasa) | Benchmark task definitions, groups, horizons, observations and native success-check interfaces used by `robots/robocasa`. Simulator code and kitchen assets are installed separately. | [MIT text](LICENSES/RoboCasa-MIT.txt), from the [pinned upstream LICENSE](https://github.com/robocasa/robocasa/blob/456174f62b89b8fca99eaaf33949c29fec9cfc2a/LICENSE). |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | Simulator and controller interfaces used through RoboCasa; implementation installed separately. | [MIT text](LICENSES/robosuite-MIT.txt), from the [pinned upstream LICENSE](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/LICENSE). |

The exact historical Harness source identity, retained file hashes and curation exclusions are in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) and the [publication manifest](publication/source-manifest.json). The frozen scientific files and original attribution remain byte-exact. The package name, original version and inherited author metadata preserve that source identity. The paper authors and project contact are listed in the [project metadata](README.md).

## Separately obtained policy, dependencies and assets

- **Xiaomi base policy:** `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, unchanged HF revision `0d1aa76d0d82debc9b611e4d1e231096434d5be4`. The [candidate](evaluations/robocasa365/2.0/candidate.json) records Apache-2.0 and the [checkpoint's own LICENSE](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365/blob/0d1aa76d0d82debc9b611e4d1e231096434d5be4/LICENSE). Processor/model code and weights are obtained separately, with their upstream notices.
- **Installed packages:** exact coordinator, simulator and policy versions are recorded in the [environment manifest and lock files](evaluations/robocasa365/2.0/environments/manifest.json). Package licenses and notices continue to apply. These dependency implementations are not vendored into the source export.
- **Simulation assets and datasets:** RoboCasa kitchen textures, Lightwheel/Objaverse objects and other downloaded assets have independent terms. The MIT license for simulator code and the project Apache-2.0 license do not replace asset licenses. See the [installation guide](docs/policy-setup.md) for the separate download boundary.
- **Planner service:** model endpoint access and provider terms are separate from the repository license; [Planner configuration](evaluations/robocasa365/2.0/PLANNER.md) documents that interface. No API credentials or model-service weights are distributed here.

Some frozen internal helpers and catalog filenames refer to other integrations. They remain required shared source dependencies; their presence does not bundle OpenPI, LeRobot, GR00T or LIBERO implementations or weights into this Xiaomi source edition.

## Presentation references

OpenPI's [public source and guides](https://github.com/Physical-Intelligence/openpi) informed the repository's presentation. The root `LICENSE` plus separate `NOTICE` arrangement follows [HexaShell's publication structure](https://github.com/HexaFuture/HexaShell). These are presentation references; this 2.0 edition does not include HexaShell runtime code, prompts or skills.
