"""Refresh reviewed publication files without changing frozen source or results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from tools import release
    from tools.verify_selected200 import verify as verify_results
except ModuleNotFoundError:
    import release
    from verify_selected200 import verify as verify_results


def refresh(root: Path) -> dict:
    """Inventory the Git index after checking the original immutable records."""
    root = root.resolve()
    manifest = release.load_manifest(root)
    release.verify_frozen_source(root, manifest)
    verify_results(root)
    indexed = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=True,
    ).split("\0")
    files = {}
    for relative in sorted(set(indexed) - {"", release.MANIFEST}):
        path = release.safe_path(root, relative)
        release.check_source_scope(manifest, relative)
        if not path.is_file():
            raise ValueError(f"Stage reviewed additions and removals first: {relative}")
        if (relative.startswith(("rpent/", "robots/")) and
                relative not in manifest["frozen_files"] and
                relative not in release.PUBLICATION_RUNTIME_DOCS):
            raise ValueError(f"Extra runtime path: {relative}")
        files[relative] = {
            "sha256": release.digest(path), "bytes": path.stat().st_size,
            "mode": "100755" if path.stat().st_mode & 0o111 else "100644",
        }
    if not set(release.retained_frozen_files(manifest)).issubset(files):
        raise ValueError("The retained frozen source must remain in the index")
    previous = (root / release.MANIFEST).read_bytes()
    manifest["files"] = files
    (root / release.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        return release.verify(root)
    except Exception:
        (root / release.MANIFEST).write_bytes(previous)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(refresh(args.root), indent=2))


if __name__ == "__main__":
    main()
