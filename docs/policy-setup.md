# Harness 2.0 / Xiaomi installation

Run installation on **Linux x86_64**, from this repository root. The retained scientific source is the historical 2.0 / Xiaomi integration: internal/package version `1.2.3.5`, commit `c8911c64f24393e8665bd9ecc34f819173d9c3ba`. Coordinator and simulator use Python 3.11; the policy uses Python 3.12.

## Requirements and offline preview

The target needs Git, `uv==0.13.7`, glibc 2.28 or newer, system EGL/OpenGL libraries and an NVIDIA driver compatible with the pinned Torch CUDA 12.8 build. Install the fixed uv tool with your normal Python tooling, for example `python3 -m pip install --user uv==0.13.7`, and ensure its executable is on PATH. The setup script does not install system packages or change driver settings. Provisioning needs Internet access and storage for three environments, CUDA libraries, checkpoint and kitchen assets. Preflight reads `nvidia-smi` without allocating a GPU or interrupting a job.

These previews work on the Mac without installing environments or downloading dependencies, assets or weights:

```bash
python3 tools/setup_robocasa_submission.py --check-locks
python3 tools/setup_robocasa_submission.py \
  --prefix /path/to/new/harness-2.0-runtime \
  --download-assets --download-checkpoint
```

The JSON includes argument-array commands, source/lock hashes, a plan digest and a separate verification command. Preview creates no prefix.

## One-command installation and verification

On the intended Linux GPU host:

```bash
python3 tools/setup_robocasa_submission.py \
  --prefix /workspace/harness-2.0-runtime \
  --download-assets --download-checkpoint --apply
```

This installs hashed dependencies into three isolated environments, installs this source without resolving additional dependencies, checks out the fixed upstream commits, installs the exact FlashAttention wheel, downloads the requested assets/checkpoint, then verifies imports and static metadata in all three interpreters. It never loads a model, constructs an environment, starts a runtime server, calls a provider or executes an action.

`--download-assets` explicitly requests the upstream download and answers its single confirmation prompt. Requested types are `tex`, `tex_generative`, `fixtures_lw`, `objs_lw` and **`objs_objaverse`**. Both Lightwheel and Objaverse are required by the unchanged [evaluation profile](../evaluations/robocasa365/2.0/evaluation-profile.json). Exact flag names and the prompt are in the [pinned upstream downloader](https://github.com/robocasa/robocasa/blob/456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/scripts/download_kitchen_assets.py).

An existing prefix is refused and never removed or repaired automatically. On failure, partial environments and `setup-receipt.json` remain for inspection; choose a new prefix for a fresh installation. A failed check produces no successful activation file.

After successful installation:

```bash
. /workspace/harness-2.0-runtime/environment.sh
python tools/check_robocasa_submission_environment.py \
  --prefix /workspace/harness-2.0-runtime \
  --require-assets --require-checkpoint
```

The activation file selects `RPENT_MAIN_PYTHON`, `RPENT_ROBOCASA_PYTHON`, `RPENT_XIAOMI_MIBOT_PYTHON`, the downloaded `RPENT_XIAOMI_MIBOT_SNAPSHOT` and EGL. If checkpoint download was omitted, no nonexistent snapshot path is exported. `setup-receipt.json` and `verification.json` stay under the external prefix and contain no credentials. Rechecking prints JSON without overwriting original receipts.

## Dependency and evidence boundaries

Every package in each PyPI lock has an exact version and artifact hashes. Coordinator API pins match the recorded local CPU compatibility surface: Pydantic AI 2.54.0, Pydantic 2.14.3, HTTPX 0.28.1, OpenAI 3.24.0, FastAPI 0.142.2, Uvicorn 0.54.0 and imageio-ffmpeg 0.6.0. The [compatibility record](../evaluations/robocasa365/2.0/environments/coordinator-tested-versions.json) describes a Mac CPU publication environment, not the historical evaluation environment. The Linux coordinator lock was resolved for Python 3.11 with uv 0.13.7 and prereleases disallowed. Simulator and policy locks retain their original contents.

The FlashAttention 2.8.3 wheel is pinned to the exact official CUDA 12/Torch 2.8/CPython 3.12/CXX11-ABI release URL. No independently verified wheel digest is available in this preparation, so it is explicitly outside the PyPI hash-lock claim. Verification checks installed version, direct-URL origin, CUDA build, ABI and import; the report retains available installer hash metadata.

The checkpoint is `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` at immutable HF revision `0d1aa76d0d82debc9b611e4d1e231096434d5be4`, with unchanged weights. Optional checkpoint verification reads only configuration/index metadata and checks that referenced shards exist; it never loads weights. Packages, assets and checkpoints remain outside Git under their upstream terms.

Import/static checks are offline and report **zero provider/model calls, zero environment constructors, zero predictions and zero control actions**. The submission tree is a curated subset; retained scientific source bytes, protocol and selected200 results remain unchanged.
