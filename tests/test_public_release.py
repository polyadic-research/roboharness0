"""Standalone publication preserves original source and excludes local output."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
SOURCE_DIGEST = "751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc"
MANIFEST = "publication/source-manifest.json"
CANONICAL_FROZEN = json.loads((ROOT / MANIFEST).read_text())["frozen_files"]
OUTSIDE_SUBMISSION_PATHS = (
    "apps/demo.py", "deploy/runpod/start.sh", "examples/demo.json", "ui/index.html",
    "robots/libero/new_backend.py", "robots/robocerebra/new_backend.py",
    "rpent/planner/new_planner.py", "rpent/skill_library/new_reader.py",
    "docs/source-en/index.rst", "docs/source-zh/index.rst",
    "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".pre-commit-config.yaml",
    "rpent/recording/narrated/README.zh-CN.md",
    "rpent/recording/narrated/README.automation.zh-CN.md",
)


def test_refresh_cannot_bless_changed_frozen_runtime(source) -> None:
    root, _ = source
    before = (root / MANIFEST).read_bytes()
    (root / "rpent/__init__.py").write_text('__version__ = "changed"\n')
    from tools.update_public_manifest import refresh
    with pytest.raises(ValueError, match="Frozen"):
        refresh(root)
    assert (root / MANIFEST).read_bytes() == before


def test_refresh_cannot_replace_historical_results(source) -> None:
    root, _ = source
    shutil.copytree(ROOT / "results/2.0-selected200", root / "results/2.0-selected200")
    before = (root / MANIFEST).read_bytes()
    result = root / "results/2.0-selected200/selected200-summary-20261107-r002.json"
    result.write_text('{"successes":200}')
    from tools.update_public_manifest import refresh
    with pytest.raises(ValueError, match="SHA-256"):
        refresh(root)
    assert (root / MANIFEST).read_bytes() == before


def test_refresh_tracks_reviewed_documentation_and_preserves_source(source) -> None:
    root, _ = source
    from tools.verify_selected200 import RECORDS
    result_dir = root / "results/2.0-selected200"
    result_dir.mkdir(parents=True)
    for name in RECORDS:
        shutil.copyfile(ROOT / "results/2.0-selected200" / name, result_dir / name)
    (root / "README.md").write_text("# Updated public documentation\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    from tools.update_public_manifest import refresh
    result = refresh(root)
    assert result["passed"] is True
    assert _release().verify(root)["frozen_files"] == 292


def _release():
    return importlib.import_module("tools.release")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def _register(root: Path, manifest: dict, name: str) -> None:
    path = root / name
    manifest["files"][name] = {
        "sha256": _sha(path), "bytes": path.stat().st_size,
        "mode": "100755" if path.stat().st_mode & 0o111 else "100644",
    }


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "source"
    root.mkdir()
    current = json.loads((ROOT / MANIFEST).read_text())
    frozen = dict(CANONICAL_FROZEN)
    excluded = set(current.get("excluded_frozen_files", []))
    for relative in frozen.keys() - excluded:
        path = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    assert len(frozen) == 292
    (root / "pyproject.toml").write_text('[project]\nname = "rpent"\nversion = "1.2.3.5"\n')
    (root / "README.md").write_text("# Standalone source\n")
    (root / "SOURCE_PROVENANCE.json").write_text(json.dumps({
        "edition": "2.0", "source_commits": {"2.0": SOURCE_COMMIT},
        "source_versions": {"2.0": "1.2.3.5"},
    }))
    manifest = {
        "schema_version": current["schema_version"], "edition": "2.0", "source_commit": SOURCE_COMMIT,
        "source_digest": SOURCE_DIGEST, "original_version": "1.2.3.5",
        "frozen_files": frozen, "files": {},
    }
    if current["schema_version"] == 2:
        manifest.update(source_scope=current["source_scope"], excluded_frozen_files=sorted(excluded))
    for path in root.rglob("*"):
        if path.is_file():
            _register(root, manifest, path.relative_to(root).as_posix())
    _write_manifest(root, manifest)
    return root, manifest


@pytest.fixture
def curated_source(source, monkeypatch):
    """A deliberately small additional omission tests the verifier's mechanism."""
    root, manifest = source
    removed = "rpent/utils/logging.py"
    excluded = sorted({*manifest.get("excluded_frozen_files", []), removed})
    manifest.update(schema_version=2, source_scope="robocasa365-xiaomi-submission",
                    excluded_frozen_files=excluded)
    encoded = json.dumps(excluded, separators=(",", ":")).encode()
    monkeypatch.setattr(_release(), "CURATION_EXCLUSIONS_SHA256", hashlib.sha256(encoded).hexdigest(), raising=False)
    (root / removed).unlink()
    manifest["files"].pop(removed)
    _write_manifest(root, manifest)
    return root, manifest, removed


def test_approved_scope_preserves_the_complete_original_hash_map(curated_source):
    root, manifest, _ = curated_source
    result = _release().verify(root)
    assert result["passed"] is True
    assert result["frozen_files"] == len(manifest["frozen_files"]) == 292
    assert result["retained_frozen_files"] == 292 - len(manifest["excluded_frozen_files"])
    assert result["source_scope"] == "robocasa365-xiaomi-submission"


def test_an_additional_missing_file_cannot_be_blessed_by_scope(curated_source):
    root, manifest, _ = curated_source
    extra = "robots/robocasa/toolkit.py"
    manifest["excluded_frozen_files"].append(extra)
    manifest["excluded_frozen_files"].sort()
    manifest["files"].pop(extra)
    (root / extra).unlink()
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="curation exclusion"):
        _release().verify(root)


@pytest.mark.parametrize("registered", [False, True])
def test_excluded_source_cannot_reenter_the_published_tree(curated_source, registered):
    root, manifest, removed = curated_source
    (root / removed).write_bytes((ROOT / removed).read_bytes())
    if registered:
        _register(root, manifest, removed)
        _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Excluded original source"):
        _release().verify(root)


def test_retained_source_still_cannot_be_modified_after_curating(curated_source):
    root, manifest, _ = curated_source
    name = "robots/robocasa/toolkit.py"
    (root / name).write_text("# Changed retained runtime fixture\n")
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Frozen source"):
        _release().verify(root)


def test_refresh_preserves_approved_omissions_without_enabling_new_ones(curated_source):
    root, manifest, _ = curated_source
    from tools.verify_selected200 import RECORDS
    result_dir = root / "results/2.0-selected200"
    result_dir.mkdir(parents=True)
    for name in RECORDS:
        shutil.copyfile(ROOT / "results/2.0-selected200" / name, result_dir / name)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    from tools.update_public_manifest import refresh
    assert refresh(root)["passed"] is True
    before = (root / MANIFEST).read_bytes()
    retained = "robots/robocasa/toolkit.py"
    (root / retained).unlink()
    subprocess.run(["git", "-C", str(root), "add", "-u"], check=True)
    with pytest.raises(ValueError, match="Frozen source"):
        refresh(root)
    assert (root / MANIFEST).read_bytes() == before


@pytest.mark.parametrize("relative", OUTSIDE_SUBMISSION_PATHS)
def test_scoped_verification_rejects_unrelated_registered_material(curated_source, relative):
    root, manifest, _ = curated_source
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Unrelated scoped-publication test fixture\n")
    _register(root, manifest, relative)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Outside submission scope"):
        _release().verify(root)


@pytest.mark.parametrize("relative", OUTSIDE_SUBMISSION_PATHS)
def test_scoped_refresh_cannot_bless_unrelated_staged_material(curated_source, relative):
    root, _, _ = curated_source
    from tools.verify_selected200 import RECORDS
    result_dir = root / "results/2.0-selected200"
    result_dir.mkdir(parents=True)
    for name in RECORDS:
        shutil.copyfile(ROOT / "results/2.0-selected200" / name, result_dir / name)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Unrelated scoped-publication test fixture\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    before = (root / MANIFEST).read_bytes()
    from tools.update_public_manifest import refresh
    with pytest.raises(ValueError, match="Outside submission scope"):
        refresh(root)
    assert (root / MANIFEST).read_bytes() == before


def test_legacy_scope_keeps_the_original_runtime_documentation_exception():
    for relative in OUTSIDE_SUBMISSION_PATHS:
        _release().check_source_scope({"schema_version": 1}, relative)


def test_standalone_verification_needs_no_edition_catalog(source) -> None:
    root, _ = source
    result = _release().verify(root)
    assert result["passed"] is True
    assert result["edition"] == "2.0"
    assert result["frozen_files"] == 292
    assert not (root / "editions").exists()


@pytest.mark.parametrize("field,value", [
    ("schema_version", 3), ("edition", "2.1"),
    ("source_commit", "1836faeaa909edf328d6f445f546a40775064264"),
    ("source_digest", "0" * 64), ("original_version", "2.0"),
])
def test_other_source_identity_cannot_be_published(source, field, value) -> None:
    root, manifest = source
    manifest[field] = value
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="source identity"):
        _release().verify(root)


def test_rehashing_a_modified_runtime_does_not_bless_it(source) -> None:
    root, manifest = source
    name = "robots/robocasa/toolkit.py"
    with (root / name).open("a") as stream:
        stream.write("\n# Modified runtime fixture\n")
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Frozen source"):
        _release().verify(root)


def test_missing_frozen_path_cannot_reduce_the_verified_source(source) -> None:
    root, manifest = source
    manifest["frozen_files"].pop("robots/robocasa/toolkit.py")
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Frozen source map"):
        _release().verify(root)


@pytest.mark.parametrize("name", ["rpent/extension.py", "robots/robocasa/extension.py"])
def test_registered_extra_runtime_cannot_expand_the_original_protocol(source, name) -> None:
    root, manifest = source
    (root / name).write_text("# Registered extra runtime fixture\n")
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Extra runtime"):
        _release().verify(root)


def test_publication_only_narration_readme_is_allowed(source) -> None:
    root, manifest = source
    name = "rpent/recording/narrated/README.zh-CN.md"
    (root / name).write_text("# Maintained recording documentation\n")
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    if manifest["schema_version"] == 1:
        assert _release().verify(root)["passed"] is True
    else:
        with pytest.raises(ValueError, match="Outside submission scope"):
            _release().verify(root)


def test_registered_payload_corruption_is_rejected(source) -> None:
    root, _ = source
    (root / "README.md").write_text("Changed without updating the manifest\n")
    with pytest.raises(ValueError, match="Payload"):
        _release().verify(root)


def test_wrong_package_version_is_rejected_even_after_rehash(source) -> None:
    root, manifest = source
    (root / "pyproject.toml").write_text('[project]\nversion = "2.0"\n')
    _register(root, manifest, "pyproject.toml")
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Package version"):
        _release().verify(root)


def test_payload_modes_are_checked(source) -> None:
    root, manifest = source
    manifest["files"]["README.md"]["mode"] = "100755"
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="mode"):
        _release().verify(root)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "a/../../outside", r"a\outside", ".git/config"])
def test_manifest_paths_cannot_escape_or_copy_git_control(source, relative) -> None:
    root, manifest = source
    manifest["files"][relative] = manifest["files"]["README.md"]
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Unsafe|Forbidden"):
        _release().verify(root)


def test_symlink_payload_is_rejected(source) -> None:
    root, _ = source
    (root / "README.md").unlink()
    (root / "README.md").symlink_to(root / "RELEASE.json")
    with pytest.raises(ValueError, match="Symlink"):
        _release().verify(root)


def test_unlisted_tracked_material_is_rejected(source) -> None:
    root, _ = source
    (root / "debug-note.txt").write_text("Unregistered local note\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    with pytest.raises(ValueError, match="Unlisted tracked"):
        _release().verify(root)
    with pytest.raises(ValueError, match="Unlisted tracked"):
        _release().verify_staged(root)


@pytest.mark.parametrize("missing", ["README.md", MANIFEST])
def test_release_file_present_only_in_worktree_is_rejected(source, missing) -> None:
    root, _ = source
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "rm", "--quiet", "--cached", missing], check=True)
    with pytest.raises(ValueError, match="missing from Git index") as error:
        _release().verify(root)
    assert missing in str(error.value)


def test_staged_verification_reads_the_index_independently_of_worktree_edits(source) -> None:
    root, _ = source
    readme = root / "README.md"
    original = readme.read_bytes()
    readme.write_text("Stale staged contents\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    readme.write_bytes(original)
    assert _release().verify(root)["passed"]
    with pytest.raises(ValueError, match="Payload SHA or size mismatch: README.md"):
        _release().verify_staged(root)
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    readme.unlink()
    assert _release().verify_staged(root)["passed"]


def test_untracked_local_output_is_not_exported(source, tmp_path) -> None:
    root, _ = source
    (root / "build").mkdir()
    (root / "build/temporary.py").write_text("# Local build output\n")
    destination = tmp_path / "export"
    result = _release().export(root, destination)
    assert result["passed"] is True
    assert not (destination / "build").exists()
    assert not (destination / ".git").exists()
    assert (destination / MANIFEST).is_file()
    assert _release().verify(destination)["passed"] is True


def test_existing_export_destination_is_preserved(source, tmp_path) -> None:
    root, _ = source
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        _release().export(root, destination)
    assert (destination / "sentinel").read_text() == "keep"


def test_export_inside_source_is_rejected(source) -> None:
    root, _ = source
    with pytest.raises(ValueError, match="outside"):
        _release().export(root, root / "public")
    assert not (root / "public").exists()


def test_other_edition_export_is_rejected(source, tmp_path) -> None:
    root, _ = source
    with pytest.raises(ValueError, match="only.*2.0"):
        _release().export(root, tmp_path / "public", "2.1")


@pytest.mark.parametrize("text,match", [
    ("[Missing](missing.md)\n", "Missing documentation link"),
    ("Local machine /Users/researcher/private/report\n", "Operator record"),
])
def test_maintained_docs_are_checked(source, text, match) -> None:
    root, manifest = source
    (root / "README.md").write_text(text)
    _register(root, manifest, "README.md")
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match=match):
        _release().verify(root)


def test_frozen_result_readme_keeps_its_historical_link(source) -> None:
    root, manifest = source
    name = "results/2.0-selected200/README.zh-CN.md"
    (root / name).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / name, root / name)
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    assert _release().verify(root)["passed"] is True


def test_unverified_result_readme_cannot_use_the_frozen_exemption(source) -> None:
    root, manifest = source
    name = "results/2.0-selected200/README.zh-CN.md"
    (root / name).parent.mkdir(parents=True)
    (root / name).write_text("[Missing](missing.md)\n")
    _register(root, manifest, name)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="Frozen result"):
        _release().verify(root)


@pytest.mark.parametrize("relative", [
    ".env", ".env.local", "nested/.env", "nested/.env.production",
    "auth.json", "nested/credentials.json", ".aws/credentials",
    ".claude/settings.local.json", ".codex/auth.json", ".ssh/id_ed25519",
    ".npmrc", "nested/.netrc", ".pypirc", "nested/.git/config",
])
def test_credential_and_local_control_paths_cannot_be_registered(tmp_path, relative) -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        _release().safe_path(tmp_path, relative)


def test_environment_template_remains_publishable(tmp_path) -> None:
    assert _release().safe_path(tmp_path, ".env.example") == tmp_path / ".env.example"


def test_manifest_refresh_rejects_force_staged_credentials_without_rewriting(source) -> None:
    root, _ = source
    before = (root / MANIFEST).read_bytes()
    shutil.copytree(ROOT / "results/2.0-selected200", root / "results/2.0-selected200")
    (root / ".gitignore").write_text(".env\n")
    (root / ".env").write_text("EXAMPLE_ONLY=not-a-real-secret\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--force", ".env"], check=True)
    updater = importlib.import_module("tools.update_public_manifest")
    with pytest.raises(ValueError, match="Forbidden"):
        updater.refresh(root)
    assert (root / MANIFEST).read_bytes() == before
