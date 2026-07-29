"""RVC inference pipeline modules.

Attributes resolve on first access (PEP 562) instead of at import time.
Eager imports here dragged torch and the whole model stack into *any* import
below this package -- including ``rcwx.pipeline.realtime_config``, the module
that exists precisely so GUI widgets and tests can avoid that cost.  Nothing
in the tree imports these names from the package itself; they stay exported
for outside callers.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rcwx.pipeline.inference import RVCPipeline
    from rcwx.pipeline.realtime_unified import RealtimeVoiceChangerUnified

__all__ = ["RVCPipeline", "RealtimeVoiceChangerUnified"]

_LAZY = {
    "RVCPipeline": "rcwx.pipeline.inference",
    "RealtimeVoiceChangerUnified": "rcwx.pipeline.realtime_unified",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


# Machinery for the lazy hook; an implementation detail.
_INTERNAL = frozenset({"import_module", "TYPE_CHECKING", "Any", "_LAZY"})


def __dir__() -> list[str]:
    # Keep the real module attributes (__file__, __path__, __spec__, and the
    # submodules the import system binds here) alongside the lazy names.
    return sorted((set(globals()) | set(__all__)) - _INTERNAL)
