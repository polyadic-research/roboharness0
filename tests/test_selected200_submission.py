"""Historical selection checks must reject changed evidence and score claims."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.verify_selected200 import verify

ROOT = Path(__file__).resolve().parents[1]


def test_recomputes_the_exact_historical_selection():
    report = verify(ROOT)
    assert report["passed"] is True
    assert (report["cases"], report["successes"], report["failures"]) == (200, 141, 59)
    assert report["source_commit"] == "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
    assert report["source_digest"] == "751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc"
    assert {key: (row["successes"], row["cases"]) for key, row in report["splits"].items()} == {
        "atomic_seen": (62, 72),
        "composite_seen": (45, 64),
        "composite_unseen": (34, 64),
    }
    assert report["planning_timeout_override_cases"] == 10
    assert report["official_full_evaluation_claimed"] is False
    assert report["raw_archives_reverified"] is False


def _copy_evidence(tmp_path):
    folder = tmp_path / "results/2.0-selected200"
    shutil.copytree(ROOT / "results/2.0-selected200", folder)
    return folder


def test_changed_result_cannot_be_relabelled_as_the_original_selection(tmp_path):
    folder = _copy_evidence(tmp_path)
    path = folder / "selected200-index-20261107-r002.json"
    data = json.loads(path.read_text())
    data["selected_results"][0]["native_status"] = "success"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path)


def test_changed_source_identity_cannot_pass_result_verification(tmp_path):
    folder = _copy_evidence(tmp_path)
    path = folder / "version-identity.json"
    data = json.loads(path.read_text())
    data["scientific_source"]["commit"] = "1836faeaa909edf328d6f445f546a40775064264"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path)


def test_missing_original_selection_record_fails_closed(tmp_path):
    folder = _copy_evidence(tmp_path)
    (folder / "selected200-summary-20261107-r002.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
