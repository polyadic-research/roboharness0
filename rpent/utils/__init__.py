"""Utility helpers with lazy public re-exports for isolated workers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RpcClient": ("rpent.utils.rpc", "RpcClient"),
    "RpcError": ("rpent.utils.socket_rpc", "RpcError"),
    "SocketRpcClient": ("rpent.utils.socket_rpc", "SocketRpcClient"),
    "SocketRpcServer": ("rpent.utils.socket_rpc", "SocketRpcServer"),
    "default_variables": ("rpent.utils.templates", "default_variables"),
    "get_logger": ("rpent.utils.logging", "get_logger"),
    "get_output_dir": ("rpent.utils.logging", "get_output_dir"),
    "init_output_dir": ("rpent.utils.logging", "init_output_dir"),
    "substitute": ("rpent.utils.templates", "substitute"),
    "substitute_text": ("rpent.utils.templates", "substitute_text"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "RpcClient",
    "RpcError",
    "SocketRpcClient",
    "SocketRpcServer",
    "default_variables",
    "get_logger",
    "get_output_dir",
    "init_output_dir",
    "substitute",
    "substitute_text",
]
