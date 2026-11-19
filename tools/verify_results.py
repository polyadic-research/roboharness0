"""Recompute this repository's preserved Harness 2.0 selected200 results."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from tools.verify_selected200 import verify as verify_selected200
except ModuleNotFoundError:
    from verify_selected200 import verify as verify_selected200


def verify(root: Path) -> dict:
    """Read and verify the fixed records without running a model or simulator."""
    return verify_selected200(root)


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
