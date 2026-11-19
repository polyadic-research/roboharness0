"""The selected CLI must work without the excluded integrations and SDKs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_selected_entrypoints_import_without_optional_provider_sdks():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import socket
import sys
class ExcludedOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'anthropic', 'mcp', 'mcp_types', 'claude_agent_sdk',
            'openai_codex', 'openai_codex_cli_bin', 'sse_starlette',
        }:
            raise ModuleNotFoundError('Excluded optional dependency')
sys.meta_path.insert(0, ExcludedOptionalDependencies())
def deny_network(*args, **kwargs):
    raise AssertionError('Import inspection attempted a network request')
socket.socket.connect = deny_network
socket.create_connection = deny_network
import rpent.cli.main
import rpent.runtime.codexgateway
import rpent.evaluation.front_controller
import robots.robocasa.runtime
from robots.robocasa import _add_cli_args
from robots.robocasa.policy_selection import policy_binding
parser = rpent.cli.main._build_argparser()
_add_cli_args(parser, use_dashboard=False)
args = parser.parse_args([
    '--env', 'robocasa', '--vla', 'xiaomi-mibot', '--planner', 'api',
    '--model', 'codexgateway:gpt-6-astra',
    '--planning-profile', 'robocasa-horizon-gated-v1',
])
assert args.vla == 'xiaomi-mibot'
assert policy_binding('xiaomi-mibot').name == 'robocasa-xiaomi-mibot-v1'
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
