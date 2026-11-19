"""Independent native HD cameras, reconstructed from timestamped physics states.

Capture is cheap and read-only. Rendering uses a separate mjModel/mjData after
the run, so policy pixels, physics and measured LLM/VLA waits stay authoritative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from bisect import bisect_right
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

CAMERAS = {
    "robot0_agentview_left": (1920, 1920),
    "robot0_agentview_center": (1920, 1080),
    "robot0_eye_in_hand": (1920, 1080),
}


class StateCapture:
    """Observe native substeps at 60 Hz simulation time and every returned state."""

    def __init__(self, root, *, physics=None, wall_clock=time.time):
        if physics is None:
            import mujoco as physics
        self.physics, self.wall_clock = physics, wall_clock
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.values = (self.root / "states.bin").open("xb")
        self.journal = (self.root / "states.jsonl").open("x")
        self.sim = None
        self.hooks = {}
        self.index = 0
        self.next_sample_s = -math.inf
        self.closed = False
        self.failed = False

    def observe(self, sim):
        if self.sim is None:
            self.sim = sim
            self.model, self.data = sim.model._model, sim.data._data
            self.spec = self.physics.mjtState.mjSTATE_INTEGRATION
            self.size = self.physics.mj_stateSize(self.model, self.spec)
            self.state = np.empty(self.size, dtype="<f8")
            (self.root / "model.xml").write_text(sim.model.get_xml())
            (self.root / "capture.json").write_text(
                json.dumps(
                    {
                        "state_spec": int(self.spec),
                        "state_size": self.size,
                        "dtype": "<f8",
                        "simulation_sample_hz": 60,
                        "geomgroup": list(
                            map(
                                int,
                                getattr(
                                    getattr(
                                        getattr(sim, "_render_context_offscreen", None),
                                        "vopt",
                                        None,
                                    ),
                                    "geomgroup",
                                    [0, 1, 1, 1, 1, 1],
                                ),
                            )
                        ),
                        "sampling": "native substeps plus every confirmed observation; wall-clock timestamps",
                    },
                    indent=2,
                )
                + "\n"
            )

            def wrap(original):
                def step(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if float(self.data.time) >= self.next_sample_s - 1e-9:
                        self._sample_safe()
                    return result

                return step

            # robosuite's default lite_physics uses step1/step2, not step.
            for name in ("step", "step2"):
                original = getattr(sim, name, None)
                if callable(original):
                    hook = wrap(original)
                    self.hooks[name] = (original, hook)
                    setattr(sim, name, hook)
        elif sim is not self.sim:
            raise ValueError("a state capture belongs to exactly one simulator")
        self._sample_safe()

    def _sample_safe(self):
        if self.failed:
            return
        try:
            self._sample()
        except Exception as exc:
            self.failed = True
            logging.getLogger(__name__).warning(
                "HD state capture failed (%s)", type(exc).__name__
            )
            try:
                (self.root / "error.json").write_text(
                    json.dumps({"error_type": type(exc).__name__})
                )
            except OSError:
                pass

    def _sample(self):
        wall_s = self.wall_clock()
        self.physics.mj_getState(self.model, self.data, self.state, self.spec)
        self.values.write(self.state.tobytes())
        self.journal.write(
            json.dumps(
                {
                    "index": self.index,
                    "wall_s": wall_s,
                    "simulation_s": float(self.data.time),
                }
            )
            + "\n"
        )
        self.values.flush()
        self.journal.flush()
        self.next_sample_s = (math.floor(float(self.data.time) * 60 + 1e-9) + 1) / 60
        self.index += 1

    def close(self):
        if self.closed:
            return
        if self.sim is not None:
            for name, (original, hook) in self.hooks.items():
                if getattr(self.sim, name) is hook:
                    setattr(self.sim, name, original)
        self.values.close()
        self.journal.close()
        self.closed = True


def capture_environment_state(adapter):
    """Opt in on the simulator machine with RPENT_NARRATION_STATES_DIR."""
    configured = os.environ.get("RPENT_NARRATION_STATES_DIR")
    if not configured or getattr(adapter, "_narration_capture_disabled", False):
        return
    capture = getattr(adapter, "_narration_capture", None)
    try:
        if capture is None or capture.sim is not adapter.env.sim:
            if capture is not None:
                capture.close()
            folder = Path(configured) / f"{os.getpid()}-{uuid4().hex[:8]}"
            capture = adapter._narration_capture = StateCapture(folder)
        capture.observe(adapter.env.sim)
    except Exception as exc:
        adapter._narration_capture_disabled = True
        if capture is not None:
            capture.close()
        logging.getLogger(__name__).warning(
            "HD state capture disabled (%s)", type(exc).__name__
        )


def render_states(root, output, *, start_wall_s, duration_s, fps=60):
    """Render only states visible on the final 60-fps wall-time grid, on SDS."""
    import mujoco

    root, output = Path(root), Path(output)
    if (root / "error.json").exists():
        raise ValueError("HD state capture failed; refusing an incomplete export")
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((root / "capture.json").read_text())
    rows = [json.loads(s) for s in (root / "states.jsonl").read_text().splitlines()]
    times = [r["wall_s"] for r in rows]
    if not times or times != sorted(times) or times[0] > start_wall_s:
        raise ValueError(
            "capture must have ordered timestamps and an initial state before planning"
        )
    states = np.memmap(root / "states.bin", dtype="<f8", mode="r")
    if states.size != len(rows) * metadata["state_size"]:
        raise ValueError("state journal and binary have different lengths")
    states = states.reshape(len(rows), metadata["state_size"])
    indices = sorted(
        {
            max(0, bisect_right(times, start_wall_s + i / fps) - 1)
            for i in range(math.ceil(duration_s * fps))
        }
    )
    model = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    # This clone alone needs a larger framebuffer; live policy model is untouched.
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1920
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    # Match robosuite's visual-only default instead of displaying collision meshes.
    options.geomgroup[:] = metadata.get("geomgroup", [0, 1, 1, 1, 1, 1])
    output.mkdir(parents=True)
    renderers = {}
    rendered = []
    try:
        for dimensions in set(CAMERAS.values()):
            w, h = dimensions
            renderers[dimensions] = mujoco.Renderer(model, width=w, height=h)
        for n, index in enumerate(indices):
            # MuJoCo's Python binding requires a writable buffer even for setState.
            mujoco.mj_setState(
                model, data, np.array(states[index], copy=True), metadata["state_spec"]
            )
            mujoco.mj_forward(model, data)
            files = {}
            for camera, dimensions in CAMERAS.items():
                renderer = renderers[dimensions]
                renderer.update_scene(data, camera=camera, scene_option=options)
                rgb = renderer.render()
                w, h = dimensions
                if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
                    raise ValueError("native HD renderer returned invalid pixels")
                name = f"{n:06d}-{camera}.jpg"
                Image.fromarray(rgb).save(output / name, quality=95, subsampling=0)
                files[camera] = name
            rendered.append(
                {
                    "wall_s": rows[index]["wall_s"],
                    "state_index": index,
                    "simulation_s": rows[index]["simulation_s"],
                    "files": files,
                }
            )
            if n % 50 == 0:
                print(f"native HD states {n}/{len(indices)}", flush=True)
    finally:
        for renderer in renderers.values():
            renderer.close()
    manifest = {
        "cameras": {c: {"width": w, "height": h} for c, (w, h) in CAMERAS.items()},
        "frames": rendered,
        "fps": fps,
        "start_wall_s": start_wall_s,
        "source": "native MuJoCo render from captured integration state; no image upscaling",
        "captured_states": len(rows),
        "rendered_states": len(rendered),
        "model_sha256": hashlib.sha256((root / "model.xml").read_bytes()).hexdigest(),
        "states_sha256": hashlib.sha256((root / "states.bin").read_bytes()).hexdigest(),
        "geomgroup": list(map(int, options.geomgroup)),
    }
    (output / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class HDFrameSource:
    """Verified native pixels selected by wall time, independently of policy frames."""

    def __init__(self, root, origin_wall_s, cameras):
        from rpent.recording.narrated.render import safe_path

        self.root = Path(root)
        manifest = self.root / "frames.json"
        metadata = json.loads(manifest.read_text())
        self.hashes = {str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()}
        self.dimensions = {}
        self.frames = metadata["frames"]
        self.times = [r["wall_s"] - origin_wall_s for r in self.frames]
        if not self.times or self.times != sorted(self.times) or self.times[0] > 0:
            raise ValueError(
                "HD capture must contain ordered times and a verified initial state"
            )
        self.change_times = sorted({0, *(t for t in self.times if t > 0)})
        for camera in cameras:
            declared = metadata["cameras"][camera]
            size = (declared["width"], declared["height"])
            if size[0] < 1920 or size[1] < 1080:
                raise ValueError(f"{camera}: native camera must be at least 1920×1080")
            self.dimensions[camera] = size
            for row in self.frames:
                path = safe_path(self.root, row["files"][camera])
                with Image.open(path) as image:
                    if image.size != size:
                        raise ValueError(
                            f"{camera}: native image size {image.size} differs from {size}"
                        )
                self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def at(self, camera, time_s):
        from rpent.recording.narrated.render import safe_path

        index = bisect_right(self.times, time_s) - 1
        if index < 0:
            return None
        return safe_path(self.root, self.frames[index]["files"][camera])

    def path(self, camera, frame):
        return self.at(camera, frame["time_s"] if frame else 0)

    @staticmethod
    def fitted(path, width, height):
        from rpent.recording.narrated.render import FrameSource

        return FrameSource.fitted(path, width, height)
