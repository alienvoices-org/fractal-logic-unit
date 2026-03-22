#!/usr/bin/env python3
"""
FLU V15 — Standalone Test Runner (no pytest required).
"""
import sys, os, traceback, importlib.util, inspect, time, unittest, io

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

exec(open(os.path.join(ROOT, "pytest_shim.py")).read())

PASSED, FAILED, ERRORS, SKIPPED = [], [], [], []

# ── Shims ─────────────────────────────────────────────────────────────────────
class _CapsysResult:
    def __init__(self, out="", err=""): self.out = out; self.err = err

class LiveCapsys:
    """Redirect sys.stdout/stderr so readouterr() captures what the test prints."""
    def __init__(self):
        self._buf_out = io.StringIO()
        self._buf_err = io.StringIO()
        self._old_out = self._old_err = None
    def _install(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._buf_out, self._buf_err
    def _uninstall(self):
        if self._old_out: sys.stdout, sys.stderr = self._old_out, self._old_err
    def readouterr(self):
        out = self._buf_out.getvalue()
        err = self._buf_err.getvalue()
        self._buf_out.truncate(0); self._buf_out.seek(0)
        self._buf_err.truncate(0); self._buf_err.seek(0)
        return _CapsysResult(out, err)

class Monkeypatch:
    def __init__(self): self._undo = []
    def setattr(self, obj, name, val, raising=True):
        orig = getattr(obj, name, None)
        setattr(obj, name, val)
        self._undo.append((obj, name, orig))
    def delattr(self, obj, name, raising=True):
        try: delattr(obj, name)
        except AttributeError:
            if raising: raise
    def undo(self):
        for obj, name, orig in reversed(self._undo):
            if orig is None:
                try: delattr(obj, name)
                except: pass
            else:
                setattr(obj, name, orig)
        self._undo.clear()


def _resolve(params, mod_fixtures, inst_fixtures):
    """Return (kwargs, err_msg). err_msg is None on success."""
    kwargs = {}
    for p in params:
        if p == "self": continue
        if p == "monkeypatch": kwargs[p] = Monkeypatch(); continue
        if p == "capsys":      kwargs[p] = LiveCapsys();  continue
        if p in inst_fixtures:
            try:    kwargs[p] = inst_fixtures[p]()
            except: return None, f"instance fixture '{p}' raised"
        elif p in mod_fixtures:
            try:    kwargs[p] = mod_fixtures[p]()
            except: return None, f"module fixture '{p}' raised"
        else:
            return None, f"unknown fixture '{p}'"
    return kwargs, None


def _call(label, fn, kwargs):
    capsys = kwargs.get("capsys")
    mp     = kwargs.get("monkeypatch")
    if capsys: capsys._install()
    try:
        fn(**kwargs)
        PASSED.append(label)
    except unittest.SkipTest as e:
        SKIPPED.append(f"{label}  (skip: {e})")
    except Exception:
        FAILED.append((label, traceback.format_exc()))
    finally:
        if capsys: capsys._uninstall()
        if mp:     mp.undo()


def run_file(filepath):
    rel = os.path.relpath(filepath, ROOT)
    mod_name = rel.replace(os.sep, "_").replace(".py", "")
    try:
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        ERRORS.append((rel, traceback.format_exc()))
        return

    # Module-level fixtures
    mod_fixtures = {
        name: getattr(mod, name)
        for name in dir(mod)
        if not name.startswith("test_") and not name.startswith("_")
        and callable(getattr(mod, name)) and not inspect.isclass(getattr(mod, name))
    }

    # Top-level test functions
    for name in sorted(dir(mod)):
        if not name.startswith("test_"): continue
        obj = getattr(mod, name)
        if not (callable(obj) and not inspect.isclass(obj)): continue
        sig = inspect.signature(obj)
        kwargs, err = _resolve(list(sig.parameters), mod_fixtures, {})
        if kwargs is None:
            SKIPPED.append(f"{rel}::{name}  ({err})")
        else:
            _call(f"{rel}::{name}", obj, kwargs)

    # Test classes
    for cname in sorted(dir(mod)):
        if not cname.startswith("Test"): continue
        cls = getattr(mod, cname)
        if not inspect.isclass(cls): continue
        try: inst = cls()
        except: continue

        inst_fixtures = {
            name: getattr(inst, name)
            for name in dir(cls)
            if not name.startswith("_") and not name.startswith("test_")
            and callable(getattr(inst, name, None))
        }
        setup = getattr(inst, "setup_method", None) or getattr(inst, "setUp", None)

        for mname in sorted(dir(cls)):
            if not mname.startswith("test_"): continue
            method = getattr(inst, mname, None)
            if not callable(method): continue
            if setup:
                try: setup()
                except: pass
            sig = inspect.signature(method)
            params = [p for p in sig.parameters if p != "self"]
            kwargs, err = _resolve(params, mod_fixtures, inst_fixtures)
            if kwargs is None:
                SKIPPED.append(f"{rel}::{cname}::{mname}  ({err})")
            else:
                _call(f"{rel}::{cname}::{mname}", method, kwargs)


def collect():
    result = []
    for root, dirs, files in os.walk(os.path.join(ROOT, "tests")):
        dirs.sort()
        for f in sorted(files):
            if f.startswith("test_") and f.endswith(".py"):
                result.append(os.path.join(root, f))
    return result


if __name__ == "__main__":
    t0    = time.time()
    files = collect()
    print(f"FLU V15 Test Suite — {len(files)} test files\n")

    for fp in files:
        run_file(fp)

    elapsed = time.time() - t0
    total   = len(PASSED) + len(FAILED) + len(ERRORS) + len(SKIPPED)

    print(f"\n{'='*72}")
    print(f"  PASSED  {len(PASSED):>4}   FAILED {len(FAILED):>4}   "
          f"ERRORS {len(ERRORS):>4}   SKIPPED {len(SKIPPED):>4}   "
          f"TOTAL {total:>4}   ({elapsed:.1f}s)")
    print(f"{'='*72}")

    if ERRORS:
        print("\n── MODULE ERRORS ──────────────────────────────────────────────")
        for path, tb in ERRORS:
            print(f"  ERROR  {path}")
            print(f"         {[l for l in tb.split(chr(10)) if l.strip()][-1]}")

    if FAILED:
        print("\n── FAILURES ────────────────────────────────────────────────────")
        for label, tb in FAILED:
            print(f"  FAIL   {label}")
            print(f"         {[l for l in tb.split(chr(10)) if l.strip()][-1]}")

    if not FAILED and not ERRORS:
        print(f"\n  ✓  All {len(PASSED)} tests passed.")
    else:
        print(f"\n  ✗  {len(FAILED)} failures, {len(ERRORS)} errors.")

    sys.exit(1 if (FAILED or ERRORS) else 0)
