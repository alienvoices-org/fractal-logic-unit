"""Internal compatibility - optional dependency handling."""
import importlib, sys, warnings, math, random

class _MissingDependency:
    def __init__(self, name, reason=""):
        self._name = name; self._reason = reason
    def __getattr__(self, _):
        raise ImportError(f"Optional dependency '{self._name}' not installed. {self._reason}")
    def __call__(self, *_, **__):
        raise ImportError(f"Optional dependency '{self._name}' not installed. {self._reason}")

def _try_import(name, reason="", warn=False):
    try:
        return importlib.import_module(name)
    except ImportError:
        if warn:
            warnings.warn(f"Optional dependency '{name}' missing. {reason}")
        return _MissingDependency(name, reason)

_NP = _try_import("numpy", "Pure-Python fallback.", warn=True)
_TORCH = _try_import("torch", "PyTorch layers disabled.", warn=True)
_SAFETENSORS = _try_import("safetensors.torch", "Conversion disabled.", warn=True)
_PL = _try_import("pytorch_lightning", "Lightning wrapper disabled.", warn=True)
_MSGPACK = _try_import("msgpack", "Binary delta serialisation disabled.", warn=True)

HAS_NUMPY = not isinstance(_NP, _MissingDependency)
HAS_TORCH = not isinstance(_TORCH, _MissingDependency)
HAS_SAFETENSORS = not isinstance(_SAFETENSORS, _MissingDependency)
HAS_PL = not isinstance(_PL, _MissingDependency)
HAS_MSGPACK = not isinstance(_MSGPACK, _MissingDependency)

def _is_odd(n): return n % 2 == 1
def _md5_hash(s):
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16)
