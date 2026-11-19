"""Shared synchronous startup lifecycle for CLI and UI evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation_progress import (
    PreparationError,
    PreparationPhaseStatus,
    PreparationProgress,
    PreparationSnapshot,
    load_preparation,
)
from rpent.evaluation.startup_log import record_startup_event

PersistedObserver = Callable[[EvaluationManifest, Path], None]


@dataclass(frozen=True, slots=True)
class EvaluationStartup:
    """Persisted identity and journal shared by every launch adapter."""

    manifest: EvaluationManifest
    root: Path
    preparation: PreparationProgress
    preparation_created: bool

    @property
    def worker_environment(self) -> Mapping[str, str]:
        return {
            "RPENT_PREPARATION_PATH": str(self.preparation.path.resolve()),
            "RPENT_EVALUATION_ID": self.manifest.evaluation_id,
            "RPENT_MANIFEST_ID": self.manifest.manifest_id,
        }

    def snapshot(self) -> PreparationSnapshot:
        snapshot = load_preparation(
            self.preparation.path,
            evaluation_id=self.manifest.evaluation_id,
            manifest_id=self.manifest.manifest_id,
        )
        if snapshot is None:  # pragma: no cover - guarded by initialization
            raise FileNotFoundError(self.preparation.path)
        return snapshot

    def advance(
        self,
        phase: str,
        status: PreparationPhaseStatus,
        detail: str | None = None,
    ) -> None:
        self.preparation.advance(phase, status, detail=detail)

    def prepare_dependencies(
        self,
        inference_lease: Any | None,
        cancelled: Callable[[], bool],
    ) -> bool:
        """Prepare inference dependencies and open the simulator phase."""
        snapshot = self.snapshot()
        if snapshot.status != "preparing":
            return False
        if snapshot.current_phase != "evaluation_created":
            return True
        if cancelled():
            self.cancel()
            return False
        try:
            if inference_lease is None:
                self.advance("sam3_loading", "skipped", "not required")
                self.advance("vla_loading", "skipped", "not required")
            else:
                inference_lease.ensure_ready(self.advance, cancelled)
                for timing_phase, duration_ms in getattr(
                    inference_lease, "startup_timings", {}
                ).items():
                    self.record(
                        timing_phase,
                        "completed",
                        duration_ms=int(duration_ms),
                    )
        except CancelledError:
            self.cancel()
            return False
        except Exception as exc:
            self.fail(exc)
            raise
        if cancelled():
            self.cancel()
            return False
        self.advance("simulator_starting", "loading")
        self.record("simulator_starting", "phase_started")
        return True

    def fail(self, exc: BaseException) -> None:
        snapshot = self.snapshot()
        if snapshot.status != "preparing":
            return
        phase = _active_or_next_phase(self.preparation, snapshot)
        message = " ".join(str(exc).splitlines()).strip() or "preparation failed"
        self.preparation.advance(
            phase,
            "failed",
            error=PreparationError(
                (type(exc).__name__ or "PreparationError")[:4096],
                message[:4096],
            ),
        )

    def cancel(self) -> None:
        snapshot = self.snapshot()
        if snapshot.status != "preparing":
            return
        phase = _active_or_next_phase(self.preparation, snapshot)
        self.preparation.advance(
            phase,
            "cancelled",
            detail="preparation cancelled",
        )

    def record(self, phase: str, event: str, **fields: Any) -> bool:
        return record_startup_event(
            self.root,
            self.manifest.evaluation_id,
            phase,
            event,
            **fields,
        )


def persist_evaluation_manifest(
    manifest: EvaluationManifest,
    root: str | Path,
) -> tuple[EvaluationManifest, Path]:
    """Create or confirm one manifest and its authoritative output root."""
    selected_root = Path(root).resolve()
    persisted = manifest.write_new(selected_root / "evaluation_manifest.json")
    declared_root = Path(str(persisted.runtime["evaluation_root"])).resolve()
    if declared_root != selected_root:
        raise ValueError("manifest evaluation_root does not match prepared root")
    return persisted, selected_root


def initialize_evaluation_startup(
    manifest: EvaluationManifest,
    root: str | Path,
    *,
    on_persisted: PersistedObserver | None = None,
) -> EvaluationStartup:
    """Persist identity and atomically create or confirm its preparation journal."""
    persisted, selected_root = persist_evaluation_manifest(manifest, root)
    if on_persisted is not None:
        on_persisted(persisted, selected_root)
    path = selected_root / "preparation.json"
    try:
        preparation = PreparationProgress.create(
            path,
            persisted.evaluation_id,
            persisted.manifest_id,
        )
    except FileExistsError:
        snapshot = load_preparation(
            path,
            evaluation_id=persisted.evaluation_id,
            manifest_id=persisted.manifest_id,
        )
        if snapshot is None:  # pragma: no cover - path existed under the lock
            raise FileNotFoundError(path)
        preparation = PreparationProgress(path)
        created = False
    else:
        created = True
        record_startup_event(
            selected_root,
            persisted.evaluation_id,
            "evaluation_created",
            "execution_started",
        )
    return EvaluationStartup(persisted, selected_root, preparation, created)


def _active_or_next_phase(
    preparation: PreparationProgress,
    snapshot: PreparationSnapshot,
) -> str:
    current = next(
        phase for phase in snapshot.phases if phase.phase == snapshot.current_phase
    )
    if current.status == "loading":
        return current.phase
    next_phase = next(
        (phase for phase in snapshot.phases if phase.status == "pending"),
        None,
    )
    if next_phase is None:
        return current.phase
    preparation.advance(next_phase.phase, "loading")
    return next_phase.phase


__all__ = [
    "EvaluationStartup",
    "initialize_evaluation_startup",
    "persist_evaluation_manifest",
]
