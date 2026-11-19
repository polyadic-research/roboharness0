"""Synthetic CPU fixtures for environment-independent evaluation contracts.

These records and pixels are test inputs, never benchmark or policy evidence.
"""

from dataclasses import dataclass
from pathlib import Path

from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.recording.contracts import CameraDescriptor
from rpent.recording.display import camera_frame, ordered_frames


@dataclass(frozen=True, slots=True)
class SyntheticTask:
    suite: str
    index: int
    name: str
    language: str
    bddl_identifier: str
    upstream_horizon: int


class SyntheticDisplayAdapter:
    def __init__(self, *, width=256, height=256, camera_ids=("main_images", "wrist_images")):
        self._descriptors = tuple(
            CameraDescriptor(camera_id, "Agent" if index == 0 else "Wrist",
                             "primary" if index == 0 else "wrist", width, height, 3, 20.0)
            for index, camera_id in enumerate(camera_ids)
        )

    @property
    def descriptors(self):
        return self._descriptors

    def frames(self, observation):
        return ordered_frames(self.descriptors, observation)

    def frame(self, camera_id, observation):
        return camera_frame(self.descriptors, camera_id, observation)


def synthetic_manifest(
    root: Path,
    *,
    evaluation_id: str = "evaluation-1",
    task_name: str = "task-1",
) -> EvaluationManifest:
    case = EpisodeCase(
        ordinal=0,
        environment="synthetic",
        preset_name="preset",
        task_name=task_name,
        task_index=0,
        instruction="do it",
        seed=0,
        episode_index=0,
        horizon=10,
        checkpoint_repo_id="repo/model",
        checkpoint_revision="a" * 40,
        binding_name="binding",
    )
    return EvaluationManifest(
        rpent_commit="b" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "synthetic"},
        simulator={"revision": "c" * 40},
        policy={"revision": "d" * 40},
        task_selector="all",
        cases=(case,),
        schedule={"episodes_per_task": 1, "base_seed": 0},
        contracts={"action": {"shape": [7]}},
        runtime={
            "main_interpreter": "/python",
            "evaluation_root": str(root),
            "policy_endpoint": "http://policy",
        },
        evaluation_id=evaluation_id,
        created_at="2026-08-07T00:00:00+00:00",
    )
