from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_simulator_worker_imports_do_not_require_main_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    script = """
import builtins

real_import = builtins.__import__

def isolated_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'dotenv', 'pydantic'}:
        raise ImportError(f'{name} is unavailable in the simulator worker')
    return real_import(name, *args, **kwargs)

builtins.__import__ = isolated_import
import robots.robocasa.dependencies
import robots.robocasa.env_server
import rpent.utils.daemon
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
