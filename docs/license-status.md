# License status

On 2026-11-17 the owner confirmed **Apache-2.0** for the project source, including RoboHarness-Zero additions. This supersedes the earlier deferral. The full license is in the root [LICENSE](../LICENSE); [NOTICE](../NOTICE) records upstream contributions, following the `LICENSE` plus `NOTICE` structure used by [HexaShell](https://github.com/HexaFuture/HexaShell).

The inherited RPent [Apache-2.0 text](../LICENSES/RPent-Apache-2.0.txt) remains verbatim. Existing upstream copyright and license notices remain intact. The [RoboCasa MIT text](../LICENSES/RoboCasa-MIT.txt) and [robosuite MIT text](../LICENSES/robosuite-MIT.txt) are retained separately; [third-party notices](../THIRD_PARTY_NOTICES.md) identify their scope and pinned sources.

`pyproject.toml` declares the SPDX expression `Apache-2.0` and includes the root license, notice and third-party texts in source and wheel distributions. The [candidate metadata](../evaluations/robocasa365/2.0/candidate.json) and [source provenance](../SOURCE_PROVENANCE.json) record the same decision. Package name/version, retained scientific source and historical result bytes remain unchanged.

Weights, installed dependency implementations, datasets, simulator assets and API services retain their own upstream licenses or service terms. They are not relicensed by the project license. The source exporter does not bundle those external artifacts.

The [project metadata](../README.md) lists the team, contact and paper authors. Private API access is configured outside the source package. The [release checklist](release-checklist.md) covers source and submission materials.
