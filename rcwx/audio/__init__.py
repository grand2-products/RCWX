"""Audio I/O and processing modules.

Attributes resolve on first access (PEP 562).  Eager imports here pulled the
denoiser -- and through it torch -- into every import below this package,
including numpy-only modules such as ``rcwx.audio.sola``.  Nothing in the
tree imports these names from the package itself; they stay exported for
outside callers.

``denoise`` and ``resample`` name both a submodule and a function, and the
eager version bound the function last, so that is what callers see.  A
module-level ``__getattr__`` cannot preserve that: the import system setattrs
each submodule onto its parent, and ``__getattr__`` only runs for names that
are ABSENT.  Loading ``rcwx.audio.denoise`` -- which any of the other lazy
names does, and which ``import rcwx.pipeline.inference`` does -- would then
shadow the function for the rest of the process.  ``__getattribute__`` runs
ahead of the instance dict, so those two names resolve there instead.
"""

import sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rcwx.audio.buffer import RingOutputBuffer
    from rcwx.audio.denoise import (
        DenoiseConfig,
        MLDenoiser,
        SpectralGateDenoiser,
        denoise,
        is_ml_denoiser_available,
    )
    from rcwx.audio.input import AudioInput
    from rcwx.audio.output import AudioOutput
    from rcwx.audio.resample import resample

__all__ = [
    "AudioInput",
    "AudioOutput",
    "RingOutputBuffer",
    "DenoiseConfig",
    "MLDenoiser",
    "SpectralGateDenoiser",
    "denoise",
    "is_ml_denoiser_available",
    "resample",
]

_LAZY = {
    "AudioInput": "rcwx.audio.input",
    "AudioOutput": "rcwx.audio.output",
    "RingOutputBuffer": "rcwx.audio.buffer",
    "DenoiseConfig": "rcwx.audio.denoise",
    "MLDenoiser": "rcwx.audio.denoise",
    "SpectralGateDenoiser": "rcwx.audio.denoise",
    "denoise": "rcwx.audio.denoise",
    "is_ml_denoiser_available": "rcwx.audio.denoise",
    "resample": "rcwx.audio.resample",
}


# Names a submodule of this package would otherwise shadow.
_SHADOWED = {name: _LAZY[name] for name in ("denoise", "resample")}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache; __getattr__ only fires for absent names
    return value


# Machinery for the lazy hooks; an implementation detail, not part of the
# package surface.
_INTERNAL = frozenset(
    {"sys", "import_module", "ModuleType", "TYPE_CHECKING", "Any", "_LAZY", "_SHADOWED"}
)


def __dir__() -> list[str]:
    # Keep the real module attributes -- __file__, __path__, __spec__, and the
    # submodules bound by the import system.  Returning only __all__ made
    # ``'__path__' in dir(pkg)`` false, so package-detection idioms and REPL
    # completion both broke.
    return sorted((set(globals()) | set(__all__)) - _INTERNAL)


class _FunctionsShadowSubmodules(ModuleType):
    """Resolve the submodule-shadowed names to their functions."""

    def __getattribute__(self, name: str) -> Any:
        module_path = _SHADOWED.get(name)
        if module_path is not None:
            return getattr(import_module(module_path), name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _FunctionsShadowSubmodules
