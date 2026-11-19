from __future__ import annotations

from rpent.evaluation.startup import initialize_evaluation_startup
from rpent.evaluation.startup_log import read_startup_events
from tests.synthetic_tasks import synthetic_manifest as _manifest


def test_repeated_startup_initialization_preserves_existing_journal(tmp_path):
    manifest = _manifest(tmp_path)

    first = initialize_evaluation_startup(manifest, tmp_path)
    before = (tmp_path / "preparation.json").read_bytes()
    second = initialize_evaluation_startup(manifest, tmp_path)

    assert first.preparation_created is True
    assert second.preparation_created is False
    assert second.manifest.manifest_id == first.manifest.manifest_id
    assert (tmp_path / "preparation.json").read_bytes() == before
    assert [
        (event.phase, event.event)
        for event in read_startup_events(tmp_path / "startup.jsonl")
    ] == [("evaluation_created", "execution_started")]
