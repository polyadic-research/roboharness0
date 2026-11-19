"""Policy-session adapters used by RPent episode runtimes."""

from importlib import import_module

__all__ = ["Pi05PolicySession", "SmolVLAPolicySession", "XiaomiMiBoTPolicySession"]


def __getattr__(name):
    """Keep isolated policy servers free of unrelated main-runtime imports."""
    modules = {
        "Pi05PolicySession": "pi05",
        "SmolVLAPolicySession": "smolvla_client",
        "XiaomiMiBoTPolicySession": "xiaomi_mibot_client",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
