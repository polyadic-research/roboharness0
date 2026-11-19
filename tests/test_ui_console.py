from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from rpent.evaluation.manifest import EvaluationManifest
from rpent.ui.console import (
    MAX_CONSOLE_PAGE_BYTES,
    iter_console_download,
    open_console_download,
    read_console_page,
    read_console_text,
)
from rpent.ui.repository import EvaluationRepository
from rpent.ui.schemas import ConsolePageView

FIXTURE = Path(__file__).parent / "fixtures/ui/completed-evaluation"
CASE_ID = "d7fe6815553e4d3fe549627de494598f33f101ef2661c70ee792aedff1bb1ae0"
ATTEMPT_ID = "fixture-attempt"


def _indexed_repository(
    root: Path,
    *,
    boundary: Path | None = None,
    **kwargs,
) -> EvaluationRepository:
    repository = EvaluationRepository(
        ((root.parent if boundary is None else boundary),),
        **kwargs,
    )
    repository.register_evaluation(
        root,
        EvaluationManifest.load(root / "evaluation_manifest.json"),
    )
    return repository


def grouped_attempt_id(batch_attempt_id: str) -> str:
    return f"{batch_attempt_id}-000-{CASE_ID[:12]}"


def test_console_page_reads_byte_offsets_and_bounds_limit(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_bytes(b"alpha\nbeta\ngamma\n")

    page = read_console_page(path, offset=6, limit=5, complete=False)

    assert page == ConsolePageView(
        offset=6,
        next_offset=11,
        text="beta\n",
        truncated=False,
        complete=False,
    )


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, -1), (0, MAX_CONSOLE_PAGE_BYTES + 1)],
)
def test_console_page_rejects_invalid_bounds(
    tmp_path: Path, offset: int, limit: int
) -> None:
    path = tmp_path / "console.log"
    path.write_text("output", encoding="utf-8")

    with pytest.raises(ValueError, match="offset|limit"):
        read_console_page(path, offset=offset, limit=limit, complete=False)


def test_console_page_preserves_utf8_byte_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_text("A😀B", encoding="utf-8")

    first = read_console_page(path, offset=0, limit=3, complete=False)
    second = read_console_page(path, offset=first.next_offset, limit=4, complete=False)
    third = read_console_page(path, offset=second.next_offset, limit=1, complete=True)

    assert (first.text, first.next_offset) == ("A", 1)
    assert (second.text, second.next_offset) == ("😀", 5)
    assert (third.text, third.next_offset, third.complete) == ("B", 6, True)
    with pytest.raises(ValueError, match="UTF-8 boundary"):
        read_console_page(path, offset=2, limit=4, complete=False)

    emoji_only = tmp_path / "emoji.log"
    emoji_only.write_text("😀", encoding="utf-8")
    smallest = read_console_page(
        emoji_only, offset=0, limit=1, complete=False
    )
    assert (smallest.text, smallest.next_offset) == ("😀", 4)
    assert len(smallest.text.encode("utf-8")) <= MAX_CONSOLE_PAGE_BYTES


def test_console_page_resets_after_file_truncation(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_text("new\n", encoding="utf-8")

    page = read_console_page(path, offset=20, limit=10, complete=True)

    assert page.offset == 0
    assert page.next_offset == 4
    assert page.text == "new\n"
    assert page.truncated is True
    assert page.complete is True


def test_console_page_observes_file_growth_from_prior_next_offset(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_text("one\n", encoding="utf-8")
    first = read_console_page(path, offset=0, limit=32, complete=False)
    with path.open("ab") as stream:
        stream.write("二\n".encode())

    second = read_console_page(
        path, offset=first.next_offset, limit=32, complete=False
    )

    assert first.text == "one\n"
    assert second.text == "二\n"
    assert second.next_offset == len("one\n二\n".encode())


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_console_reader_rejects_non_regular_paths(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "console.log"
    if kind == "symlink":
        target = tmp_path / "target.log"
        target.write_text("secret", encoding="utf-8")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)

    with pytest.raises(OSError, match="regular|symlink"):
        read_console_page(path, offset=0, limit=32, complete=False)


def test_console_reader_rejects_paths_outside_attempt_root(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="attempt root"):
        read_console_page(
            outside, offset=0, limit=32, complete=False, root=attempt
        )


def test_console_reader_rejects_invalid_retained_utf8(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_bytes(b"valid\xffinvalid")

    with pytest.raises(ValueError, match="UTF-8"):
        read_console_page(path, offset=0, limit=32, complete=False)
    page = read_console_page(path, offset=6, limit=32, complete=False)
    assert page.text == "invalid"
    with pytest.raises(ValueError, match="UTF-8"):
        read_console_text(path)
    with pytest.raises(ValueError, match="UTF-8"):
        open_console_download(path)


def test_console_download_streams_a_large_validated_file_in_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "console.log"
    retained = ("worker 😀 output\n" * 20_000).encode("utf-8")
    path.write_bytes(retained)

    stream, size = open_console_download(path)
    chunks = tuple(iter_console_download(stream, size))

    assert size == len(retained)
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 64 * 1024
    assert b"".join(chunks) == retained


def test_console_download_fails_if_validated_bytes_disappear(tmp_path: Path) -> None:
    path = tmp_path / "console.log"
    path.write_text("validated output\n", encoding="utf-8")
    stream, size = open_console_download(path)
    path.write_bytes(b"")

    with pytest.raises(OSError, match="changed"):
        tuple(iter_console_download(stream, size))


def test_console_page_schema_is_strict_and_frozen() -> None:
    page = ConsolePageView(
        offset=0, next_offset=0, text="", truncated=False, complete=False
    )

    with pytest.raises(ValidationError):
        page.offset = 1
    with pytest.raises(ValidationError):
        ConsolePageView.model_validate(
            {
                "offset": "0",
                "next_offset": 0,
                "text": "",
                "truncated": False,
                "complete": False,
            }
        )


def test_repository_reads_only_the_selected_canonical_attempt(tmp_path: Path) -> None:
    root = tmp_path / "evaluations" / "completed"
    shutil.copytree(FIXTURE, root)
    selected = root / "cases" / CASE_ID / "attempts" / ATTEMPT_ID
    selected.mkdir(parents=True)
    (selected / "console.log").write_text("selected 😀\n", encoding="utf-8")
    other = root / "cases" / CASE_ID / "attempts" / "other-attempt"
    other.mkdir(parents=True)
    (other / "console.log").write_text("other\n", encoding="utf-8")
    repository = _indexed_repository(root)

    page = repository.console_page(
        "fixture-evaluation", CASE_ID, offset=0, limit=256
    )
    download = repository.console_text("fixture-evaluation", CASE_ID)

    assert page.text == "selected 😀\n"
    assert page.complete is True
    assert download == "selected 😀\n"


def test_repository_returns_empty_live_page_before_log_exists(tmp_path: Path) -> None:
    root = tmp_path / "evaluations" / "completed"
    shutil.copytree(FIXTURE, root)
    repository = _indexed_repository(root)

    page = repository.console_page(
        "fixture-evaluation", CASE_ID, offset=7, limit=256
    )

    assert page == ConsolePageView(
        offset=0,
        next_offset=0,
        text="",
        truncated=True,
        complete=True,
    )
    with pytest.raises(FileNotFoundError):
        repository.console_text("fixture-evaluation", CASE_ID)


def test_repository_returns_incomplete_empty_page_for_active_attempt_without_log(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations" / "active"
    shutil.copytree(FIXTURE, root)
    manifest = json.loads(
        (root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    (root / "results.jsonl").write_text("", encoding="utf-8")
    attempt = root / "cases" / CASE_ID / "attempts" / "active-attempt"
    attempt.mkdir(parents=True)
    (attempt / "case.json").write_text(
        json.dumps({
            "schema_version": 1,
            "evaluation_id": manifest["evaluation_id"],
            "manifest_id": manifest["manifest_id"],
            "case_id": CASE_ID,
            "attempt_id": "active-attempt",
            "case": manifest["cases"][0],
            "recording": manifest["recording"],
            "qualification": manifest["qualification"],
        }),
        encoding="utf-8",
    )

    page = _indexed_repository(root).console_page(
        "fixture-evaluation", CASE_ID, offset=0, limit=256
    )

    assert page.text == ""
    assert page.next_offset == 0
    assert page.complete is False


def test_repository_rejects_a_symlinked_selected_attempt(tmp_path: Path) -> None:
    root = tmp_path / "evaluations" / "completed"
    shutil.copytree(FIXTURE, root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "console.log").write_text("secret", encoding="utf-8")
    attempt_parent = root / "cases" / CASE_ID / "attempts"
    attempt_parent.mkdir(parents=True)
    (attempt_parent / ATTEMPT_ID).symlink_to(outside, target_is_directory=True)

    with pytest.raises((OSError, ValueError), match="attempt|root|canonical"):
        _indexed_repository(root).console_page(
            "fixture-evaluation", CASE_ID, offset=0, limit=256
        )


def test_repository_reads_the_selected_grouped_batch_console(tmp_path: Path) -> None:
    root = tmp_path / "evaluations" / "completed"
    shutil.copytree(FIXTURE, root)
    historical = json.loads((root / "results.jsonl").read_text(encoding="utf-8"))
    historical["task_batch_attempt_id"] = "batch-selected"
    historical["attempt_id"] = grouped_attempt_id("batch-selected")
    historical["artifact_dir"] = historical["artifact_dir"].replace(
        ATTEMPT_ID, grouped_attempt_id("batch-selected")
    )
    current = {
        **historical,
        "attempt_id": grouped_attempt_id("batch-current"),
        "task_batch_attempt_id": "batch-current",
        "artifact_dir": historical["artifact_dir"].replace(
            grouped_attempt_id("batch-selected"),
            grouped_attempt_id("batch-current"),
        ),
    }
    (root / "results.jsonl").write_text(
        json.dumps(historical) + "\n" + json.dumps(current) + "\n",
        encoding="utf-8",
    )
    selected = root / "task_batches" / "pick mug" / "batch-selected"
    selected.mkdir(parents=True)
    (selected / "console.log").write_text("grouped output\n", encoding="utf-8")
    current_root = root / "task_batches" / "pick mug" / "batch-current"
    current_root.mkdir(parents=True)
    (current_root / "console.log").write_text("current output\n", encoding="utf-8")
    unrelated = root / "task_batches" / "pick mug" / "batch-other"
    unrelated.mkdir(parents=True)
    (unrelated / "console.log").write_text("wrong output\n", encoding="utf-8")
    repository = _indexed_repository(root)

    latest = repository.console_page(
        "fixture-evaluation", CASE_ID, offset=0, limit=256
    )
    historical_page = repository.console_page(
        "fixture-evaluation",
        CASE_ID,
        attempt_id=grouped_attempt_id("batch-selected"),
        offset=0,
        limit=256,
    )

    assert latest.text == "current output\n"
    assert historical_page.text == "grouped output\n"
    assert repository.console_text(
        "fixture-evaluation",
        CASE_ID,
        attempt_id=grouped_attempt_id("batch-selected"),
    ) == (
        "grouped output\n"
    )


def test_repository_rejects_a_symlinked_selected_batch_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations" / "completed"
    shutil.copytree(FIXTURE, root)
    result = json.loads((root / "results.jsonl").read_text(encoding="utf-8"))
    result["task_batch_attempt_id"] = "batch-selected"
    result["attempt_id"] = grouped_attempt_id("batch-selected")
    result["artifact_dir"] = result["artifact_dir"].replace(
        ATTEMPT_ID, grouped_attempt_id("batch-selected")
    )
    (root / "results.jsonl").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    outside = tmp_path / "outside-batch"
    outside.mkdir()
    (outside / "console.log").write_text("secret", encoding="utf-8")
    selected = root / "task_batches" / "pick mug" / "batch-selected"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical"):
        _indexed_repository(root).console_page(
            "fixture-evaluation", CASE_ID, offset=0, limit=256
        )


@pytest.mark.parametrize("terminal_status", [None, "complete", "failed", "cancelled"])
def test_repository_prefers_active_grouped_batch_over_old_central_result(
    tmp_path: Path, terminal_status: str | None
) -> None:
    root = tmp_path / "evaluations" / "active-grouped"
    shutil.copytree(FIXTURE, root)
    manifest = json.loads(
        (root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    old = json.loads((root / "results.jsonl").read_text(encoding="utf-8"))
    old["attempt_id"] = grouped_attempt_id("batch-old")
    old["task_batch_attempt_id"] = "batch-old"
    old["artifact_dir"] = old["artifact_dir"].replace(
        ATTEMPT_ID, grouped_attempt_id("batch-old")
    )
    (root / "results.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    old_batch = root / "task_batches" / "pick mug" / "batch-old"
    old_batch.mkdir(parents=True)
    (old_batch / "console.log").write_text("old output\n", encoding="utf-8")

    active_attempt_id = grouped_attempt_id("batch-active")
    active_attempt = root / "cases" / CASE_ID / "attempts" / active_attempt_id
    active_attempt.mkdir(parents=True)
    registration = {
        "schema_version": 1,
        "evaluation_id": manifest["evaluation_id"],
        "manifest_id": manifest["manifest_id"],
        "case_id": CASE_ID,
        "attempt_id": active_attempt_id,
        "registered_at": "2026-08-09T00:00:00+00:00",
    }
    (active_attempt / "case.json").write_text(
        json.dumps({
            **registration,
            "started_at": "2026-08-09T00:00:00+00:00",
            "case": manifest["cases"][0],
            "recording": manifest["recording"],
            "qualification": manifest["qualification"],
        }) + "\n",
        encoding="utf-8",
    )
    (active_attempt.parents[1] / "current_attempt.json").write_text(
        json.dumps(registration) + "\n", encoding="utf-8"
    )
    active_batch = root / "task_batches" / "pick mug" / "batch-active"
    active_batch.mkdir(parents=True)
    running = {
        "manifest_id": manifest["manifest_id"],
        "task_name": "pick mug",
        "batch_attempt_id": "batch-active",
        "status": "running",
        "expected_case_ids": [CASE_ID],
        "completed_case_ids": [],
        "started_at": "2026-08-09T00:00:00+00:00",
        "finished_at": None,
    }
    records = [running]
    if terminal_status is not None:
        records.append(
            {
                **running,
                "status": terminal_status,
                "completed_case_ids": [CASE_ID],
                "finished_at": "2026-08-09T00:01:00+00:00",
            }
        )
    (active_batch / "batch.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (active_batch / "console.log").write_text("active output\n", encoding="utf-8")
    repository = _indexed_repository(root)

    latest = repository.console_page(
        "fixture-evaluation", CASE_ID, offset=0, limit=256
    )
    explicit = repository.console_page(
        "fixture-evaluation",
        CASE_ID,
        attempt_id=active_attempt_id,
        offset=0,
        limit=256,
    )

    assert latest.text == explicit.text == "active output\n"
    assert latest.complete is explicit.complete is False
