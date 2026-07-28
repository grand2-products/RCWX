"""Audio I/O and processing modules.

Attributes resolve on first access (PEP 562).  Eager imports here pulled the
denoiser -- and through it torch -- into every import below this package,
including numpy-only modules such as ``rcwx.audio.sola``.  Nothing in the
tree imports these names from the package itself; they stay exported for
outside callers.

``denoise`` and ``resample`` name both a submodule and a function.  Binding
either function eagerly reintroduces the import being avoided (torch for one,
scipy for the other), so both stay lazy: they resolve to the FUNCTION as
before, except when their submodule was imported before the package attribute
is first touched, where the module wins.  Nothing in the tree relies on these
re-exports; import from the submodule to be unambiguous.
"""

from importlib import import_module
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


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    # Bind it here, which matters beyond caching: ``denoise`` and ``resample``
    # name both a submodule and a function, and importing the submodule just
    # set this attribute to the module.  The eager version bound the function
    # last and callers rely on that, so restore it.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
