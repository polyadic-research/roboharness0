"""Verify and export the standalone historical Harness 2.0 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

MANIFEST = "publication/source-manifest.json"
SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
SOURCE_DIGEST = "751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc"
SOURCE_VERSION = "1.2.3.5"
# Canonical sorted JSON of the original 291 runtime/model-input hashes plus
# RELEASE.json. This pins the whole map, including its path coverage.
FROZEN_MAP_SHA256 = "4f988f0887f7df42ca3974265c5f45aa952f19f4569ff10515671f5d886cb572"
# Only the authorized, dependency-reviewed omission set may use schema 2.
# Pinned after dependency review; None fails closed.
CURATION_EXCLUSIONS_SHA256: str | None = "fb7e817276f1a31cb45db0e50db57519f82f981064c834a67b095ff10a8caad4"
SOURCE_SCOPE = "robocasa365-xiaomi-submission"
PUBLICATION_RUNTIME_DOCS = {
    "rpent/recording/narrated/README.zh-CN.md",
    "rpent/recording/narrated/README.automation.zh-CN.md",
}
SCOPED_EXCLUDED_PREFIXES = (
    "apps/", "deploy/", "examples/", "ui/", "robots/libero/",
    "robots/robocerebra/", "rpent/planner/", "rpent/skill_library/",
    "docs/source-en/", "docs/source-zh/",
)
SCOPED_EXCLUDED_FILES = {
    "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".pre-commit-config.yaml",
    *PUBLICATION_RUNTIME_DOCS,
}
FORBIDDEN = {
    ".git", "editions", "log data", "archives", "experiments", ".venv",
    "node_modules", "logs", "outputs", "ci-reports", "build", "dist",
    ".cache", ".pytest_cache", "__pycache__", ".ruff_cache",
}
LOCAL_CONTROL_COMPONENTS = {".git", ".aws", ".claude", ".codex", ".ssh"}
CREDENTIAL_FILENAMES = {"auth.json", "credentials.json", ".npmrc", ".netrc", ".pypirc"}


def safe_path(root: Path, relative: str) -> Path:
    """Require a canonical relative file path with no symlink components."""
    if not isinstance(relative, str):
        raise ValueError("Unsafe release path")
    parts = PurePosixPath(relative)
    if (not relative or parts.is_absolute() or ".." in parts.parts or
            "\\" in relative or relative != parts.as_posix() or relative == "."):
        raise ValueError(f"Unsafe release path: {relative!r}")
    normalized = tuple(component.lower() for component in parts.parts)
    filename = normalized[-1]
    private_environment = filename == ".env" or (
        filename.startswith(".env.") and filename != ".env.example"
    )
    if (parts.parts[0] in FORBIDDEN or LOCAL_CONTROL_COMPONENTS.intersection(normalized) or
            filename in CREDENTIAL_FILENAMES or private_environment):
        raise ValueError(f"Forbidden release path: {relative}")
    path = root
    for component in parts.parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"Symlink in release path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe release path: {relative}")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(root: Path) -> dict:
    """Keep original provenance and pin the explicitly reviewed source scope."""
    data = json.loads(safe_path(root, MANIFEST).read_text())
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or (
        data.get("edition"), data.get("source_commit"),
        data.get("source_digest"), data.get("original_version"),
    ) != ("2.0", SOURCE_COMMIT, SOURCE_DIGEST, SOURCE_VERSION):
        raise ValueError("Unexpected standalone source identity")
    if data["schema_version"] == 1:
        if "source_scope" in data or "excluded_frozen_files" in data:
            raise ValueError("Unexpected standalone source identity")
    elif data["schema_version"] == 2:
        if data.get("source_scope") != SOURCE_SCOPE or "excluded_frozen_files" not in data:
            raise ValueError("Unexpected standalone source identity")
    else:
        raise ValueError("Unexpected standalone source identity")
    frozen = data.get("frozen_files")
    if not isinstance(frozen, dict):
        raise ValueError("Frozen source map is missing")
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != FROZEN_MAP_SHA256:
        raise ValueError("Frozen source map differs from the original 292-file map")
    if data["schema_version"] == 2:
        excluded = data["excluded_frozen_files"]
        if (not isinstance(excluded, list) or not excluded or
                any(not isinstance(path, str) for path in excluded) or
                excluded != sorted(set(excluded)) or
                any(path not in frozen or not path.startswith(("rpent/", "robots/"))
                    for path in excluded)):
            raise ValueError("Invalid approved curation exclusion set")
        encoded = json.dumps(excluded, separators=(",", ":")).encode()
        if (CURATION_EXCLUSIONS_SHA256 is None or
                hashlib.sha256(encoded).hexdigest() != CURATION_EXCLUSIONS_SHA256):
            raise ValueError("The approved curation exclusion set changed")
    if not isinstance(data.get("files"), dict) or not data["files"]:
        raise ValueError("The registered source payload is missing")
    if MANIFEST in data["files"]:
        raise ValueError("The source manifest must exclude its own hash")
    return data


def retained_frozen_files(manifest: dict) -> dict[str, str]:
    """Return required original bytes without altering the canonical hash map."""
    excluded = set(manifest.get("excluded_frozen_files", []))
    return {path: sha for path, sha in manifest["frozen_files"].items() if path not in excluded}


def check_source_scope(manifest: dict, relative: str) -> None:
    """Keep removed framework support out of the scoped submission payload."""
    if manifest["schema_version"] != 2:
        return
    normalized = relative.lower()
    if (normalized in {path.lower() for path in SCOPED_EXCLUDED_FILES} or
            any(normalized == prefix.rstrip("/") or normalized.startswith(prefix)
                for prefix in SCOPED_EXCLUDED_PREFIXES)):
        raise ValueError(f"Outside submission scope: {relative}")


def verify_frozen_source(root: Path, manifest: dict, registered: set[str] | None = None) -> None:
    """Require retained bytes and refuse even untracked reintroduced omissions."""
    for relative in manifest.get("excluded_frozen_files", []):
        if (registered is not None and relative in registered) or safe_path(root, relative).exists():
            raise ValueError(f"Excluded original source must remain absent: {relative}")
    for relative, expected in retained_frozen_files(manifest).items():
        path = safe_path(root, relative)
        if ((registered is not None and relative not in registered) or
                not path.is_file() or digest(path) != expected):
            raise ValueError(f"Frozen source differs from the original: {relative}")


def check_public_documentation(root: Path, relative: str, frozen: dict) -> None:
    """Check maintained Markdown while retaining exact historical references."""
    if relative == "results/2.0-selected200/README.zh-CN.md":
        try:
            from tools.verify_selected200 import RECORDS
        except ModuleNotFoundError:
            from verify_selected200 import RECORDS
        if digest(safe_path(root, relative)) != RECORDS["README.zh-CN.md"]:
            raise ValueError("Frozen result README differs from its historical bytes")
        return
    if not relative.endswith(".md") or relative in frozen or relative.startswith("tests/"):
        return
    path = safe_path(root, relative)
    text = path.read_text()
    if re.search(r"/Users/(?!<|your-user|username)[^/\s]+|\b(?:root|ubuntu)@\d+\.", text):
        raise ValueError(f"Operator record in public documentation: {relative}")
    if "VLM/reviews/" in text or ".superpowers/sdd/" in text:
        raise ValueError(f"Operator record in public documentation: {relative}")
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
        target = target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if not (path.parent / unquote(parsed.path)).is_file():
            raise ValueError(f"Missing documentation link: {relative} -> {target}")


def verify(root: Path, *, tracked_files: set[str] | None = None) -> dict:
    """Verify immutable source, package identity and every registered payload."""
    root = root.resolve()
    data = load_manifest(root)
    registered = set(data["files"])
    for relative in registered:
        safe_path(root, relative)
        check_source_scope(data, relative)
        if (relative.startswith(("rpent/", "robots/")) and
                relative not in data["frozen_files"] and relative not in PUBLICATION_RUNTIME_DOCS):
            raise ValueError(f"Extra runtime path outside the frozen source: {relative}")
    verify_frozen_source(root, data, registered)
    for relative, entry in data["files"].items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "mode"}:
            raise ValueError(f"Invalid payload metadata: {relative}")
        if (not isinstance(entry["sha256"], str) or
                re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None or
                type(entry["bytes"]) is not int or entry["bytes"] < 0 or
                not isinstance(entry["mode"], str) or entry["mode"] not in {"100644", "100755"}):
            raise ValueError(f"Invalid payload metadata: {relative}")
        path = safe_path(root, relative)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Payload file is missing: {relative}")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Payload SHA or size mismatch: {relative}")
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if mode != entry["mode"]:
            raise ValueError(f"Payload mode mismatch: {relative}")
    if "pyproject.toml" not in registered or "SOURCE_PROVENANCE.json" not in registered:
        raise ValueError("Package source identity files are not registered")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if project.get("project", {}).get("version") != SOURCE_VERSION:
        raise ValueError("Package version must retain the original 1.2.3.5 identity")
    release_info = json.loads((root / "RELEASE.json").read_text())
    if release_info.get("harness_version") != SOURCE_VERSION:
        raise ValueError("The original RELEASE.json version changed")
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    if (provenance.get("edition") != "2.0" or
            provenance.get("source_commits") != {"2.0": SOURCE_COMMIT} or
            provenance.get("source_versions") != {"2.0": SOURCE_VERSION}):
        raise ValueError("The source identity provenance differs from the selected source")
    if "source_digests" in provenance and provenance["source_digests"] != {"2.0": SOURCE_DIGEST}:
        raise ValueError("The source identity digest differs from the selected source")
    if tracked_files is None and (root / ".git").exists():
        indexed = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        tracked_files = {os.fsdecode(name) for name in indexed.split(b"\0") if name}
    if tracked_files is not None:
        expected = registered | {MANIFEST}
        extra = tracked_files - expected
        if extra:
            raise ValueError(f"Unlisted tracked files: {', '.join(sorted(extra))}")
        missing = expected - tracked_files
        if missing:
            raise ValueError(f"Release files missing from Git index: {', '.join(sorted(missing))}")
    for relative in registered:
        check_public_documentation(root, relative, data["frozen_files"])
    return {
        "passed": True, "edition": "2.0", "source_commit": SOURCE_COMMIT,
        "original_version": SOURCE_VERSION, "files": len(registered),
        "frozen_files": len(data["frozen_files"]),
        "retained_frozen_files": len(retained_frozen_files(data)),
        "excluded_frozen_files": len(data.get("excluded_frozen_files", [])),
        "source_scope": data.get("source_scope", "complete-original-source"),
    }


def verify_staged(root: Path) -> dict:
    """Verify the file modes and raw Git blobs that the next commit will contain."""
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError("Staged verification requires a Git worktree root")
    git = ["git", "-C", str(root)]
    tree = subprocess.check_output([*git, "write-tree"], text=True).strip()
    entries = subprocess.check_output([*git, "ls-tree", "-rz", "--full-tree", tree])
    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory).resolve()
        tracked_files = set()
        for entry in entries.split(b"\0")[:-1]:
            metadata, name = entry.split(b"\t", 1)
            mode, _, oid = metadata.split()
            relative = os.fsdecode(name)
            path = safe_path(snapshot, relative)
            if mode not in {b"100644", b"100755"}:
                raise ValueError(f"Unsupported staged release file mode: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                subprocess.run([*git, "cat-file", "blob", oid.decode("ascii")], stdout=stream, check=True)
            path.chmod(0o755 if mode == b"100755" else 0o644)
            tracked_files.add(relative)
        return verify(snapshot, tracked_files=tracked_files)


def export(root: Path, output: Path, edition: str | None = None) -> dict:
    """Copy only verified registered files into a new tree without Git control."""
    if edition not in {None, "2.0"}:
        raise ValueError("This standalone source exports only edition 2.0")
    root = root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Export destination already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Export outside the source checkout")
    verify(root)
    data = load_manifest(root)
    output.mkdir(parents=True)
    for relative, entry in data["files"].items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(root, relative), destination)
        destination.chmod(0o755 if entry["mode"] == "100755" else 0o644)
    manifest = output / MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(safe_path(root, MANIFEST), manifest)
    manifest.chmod(0o644)
    verify(output)
    return {
        "passed": True, "edition": "2.0", "files": len(data["files"]) + 1,
        "git_history_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "export"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--edition")
    parser.add_argument("--staged", action="store_true", help="Verify the exact Git index snapshot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        if args.edition not in {None, "2.0"}:
            parser.error("this source contains only edition 2.0")
        result = verify_staged(args.root) if args.staged else verify(args.root)
    else:
        if args.staged:
            parser.error("--staged applies only to verify")
        if args.output is None:
            parser.error("export requires --output")
        result = export(args.root, args.output, args.edition)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
