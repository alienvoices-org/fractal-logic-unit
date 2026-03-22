# Minimal pytest shim — makes pytest.raises, pytest.mark etc. available
# without installing pytest, for the standalone runner.
import sys, unittest

class _Mark:
    def __getattr__(self, name):
        def decorator(fn=None, **kw):
            if fn is not None: return fn
            return lambda f: f
        return decorator
    def parametrize(self, *a, **kw):
        return lambda f: f

class _Raises:
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"Expected {self.exc} but no exception raised")
        return issubclass(et, self.exc)

class _Pytest:
    mark = _Mark()
    @staticmethod
    def raises(exc, *a, **kw): return _Raises(exc)
    @staticmethod
    def skip(reason=""): raise unittest.SkipTest(reason)
    @staticmethod
    def approx(val, **kw): return val
    @staticmethod
    def warns(*a, **kw):
        import contextlib
        return contextlib.nullcontext()
    @staticmethod
    def fail(msg=""): raise AssertionError(msg)

import types
pytest_mod = types.ModuleType("pytest")
_p = _Pytest()
pytest_mod.mark     = _p.mark
pytest_mod.raises   = _p.raises
pytest_mod.skip     = _p.skip
pytest_mod.approx   = _p.approx
pytest_mod.warns    = _p.warns
pytest_mod.fail     = _p.fail
sys.modules["pytest"] = pytest_mod

# fixture decorator support
def _fixture(fn=None, scope="function", **kw):
    if fn is not None: return fn
    return lambda f: f

pytest_mod.fixture = _fixture
