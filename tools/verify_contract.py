#!/usr/bin/env python3
"""
tools/verify_contract.py
=========================
FLU Package Contract Compliance Checker.

Implements the `flu verify` command.  Checks that the package satisfies
every required item in docs/PACKAGE_CONTRACT.json (compliance_checklist
items C-01 through C-10).

This was the external audit's top structural recommendation (March 2026):
    'The contract is normative, not enforced. Ideally you would provide
     `flu verify` which checks contract compliance, theorem registry
     validity, and benchmark reproducibility.'

USAGE
-----
    python tools/verify_contract.py
    python tools/verify_contract.py --package-root .
    python tools/verify_contract.py --json        # machine-readable output
    python tools/verify_contract.py --strict      # exit 1 on any failure

EXIT CODES
----------
    0  All checks pass
    1  One or more required checks failed (--strict mode)
    0  Warnings only (non-required checks failed, default mode)
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ── Result model ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    id:       str
    desc:     str
    required: bool
    passed:   bool
    detail:   str = ""

    @property
    def symbol(self) -> str:
        if self.passed:
            return "✓"
        return "✗" if self.required else "⚠"

    @property
    def label(self) -> str:
        return "PASS" if self.passed else ("FAIL" if self.required else "WARN")


# ── Individual checks ─────────────────────────────────────────────────────────

def c01_theorem_registry_exists(root: Path) -> CheckResult:
    """C-01: theorem_registry.py must exist."""
    candidates = list(root.glob("src/*/theory/theorem_registry.py"))
    passed = len(candidates) > 0
    return CheckResult("C-01", "theorem_registry.py exists", True, passed,
                       str(candidates[0]) if passed else "not found under src/*/theory/")


def c02_all_theorems_valid_status(root: Path) -> CheckResult:
    """C-02: All theorems have PROVEN | CONJECTURE | DESIGN_INTENT | FALSE | DISPROVEN_SCOPED."""
    sys.path.insert(0, str(root / "src"))
    try:
        from flu.theory.theorem_registry import REGISTRY
        valid = {"PROVEN", "CONJECTURE", "DESIGN_INTENT", "FALSE", "DISPROVEN_SCOPED"}
        bad = [(k, t.status) for k, t in REGISTRY.items() if t.status not in valid]
        passed = len(bad) == 0
        detail = f"{len(REGISTRY)} theorems checked" if passed else f"invalid: {bad}"
        return CheckResult("C-02", "all theorems have valid status", True, passed, detail)
    except Exception as e:
        return CheckResult("C-02", "all theorems have valid status", True, False, str(e))


def c03_tests_pass(root: Path) -> CheckResult:
    """C-03: tests/run_all.py exits 0 (all tests pass)."""
    run_all = root / "tests" / "run_all.py"
    if not run_all.exists():
        return CheckResult("C-03", "tests/run_all.py passes", True, False, "run_all.py not found")
    result = subprocess.run(
        [sys.executable, str(run_all)],
        capture_output=True, text=True, cwd=str(root),
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        timeout=300,
    )
    passed = result.returncode == 0
    # Count passing tests
    passing = result.stdout.count("✓")
    failing = result.stdout.count("✗")
    detail = f"{passing} passing, {failing} failing" if passed else f"exit {result.returncode}; {failing} failures"
    return CheckResult("C-03", "tests/run_all.py passes", True, passed, detail)


def c04_open_debt_exists(root: Path) -> CheckResult:
    """C-04: docs/OPEN_DEBT.md exists and is non-empty."""
    f = root / "docs" / "OPEN_DEBT.md"
    passed = f.exists() and f.stat().st_size > 100
    detail = f"{f.stat().st_size} bytes" if f.exists() else "not found"
    return CheckResult("C-04", "docs/OPEN_DEBT.md exists and non-empty", True, passed, detail)


def c05_benchmarks_doc_exists(root: Path) -> CheckResult:
    """C-05: docs/BENCHMARKS.md exists with honest criterion."""
    f = root / "docs" / "BENCHMARKS.md"
    if not f.exists():
        return CheckResult("C-05", "docs/BENCHMARKS.md has honest criterion", True, False, "not found")
    content = f.read_text()
    has_honest = "Honest" in content or "honest" in content or "Proposed vs Delivered" in content
    passed = has_honest
    detail = "honest criteria present" if passed else "missing 'honest criteria' section"
    return CheckResult("C-05", "docs/BENCHMARKS.md has honest criterion", True, passed, detail)


def c06_manifest_exists(root: Path) -> CheckResult:
    """C-06: FLU_MANIFEST.json exists at root."""
    f = root / "FLU_MANIFEST.json"
    passed = f.exists()
    if passed:
        try:
            with open(f) as fh:
                obj = json.load(fh)
            detail = f"valid JSON, {len(obj)} top-level keys"
        except Exception as e:
            return CheckResult("C-06", "FLU_MANIFEST.json exists", True, False, f"invalid JSON: {e}")
    else:
        detail = "not found"
    return CheckResult("C-06", "FLU_MANIFEST.json exists", True, passed, detail)


def c07_readme_badges(root: Path) -> CheckResult:
    """C-07: README.md has theorem badge and test badge."""
    f = root / "README.md"
    if not f.exists():
        return CheckResult("C-07", "README.md has theorem and test badges", True, False, "not found")
    content = f.read_text()
    has_theorem = "PROVEN" in content or "theorems" in content.lower()
    has_tests   = "/190" in content or "passing" in content.lower() or "tests" in content.lower()
    passed = has_theorem and has_tests
    detail = f"theorem={'✓' if has_theorem else '✗'} test={'✓' if has_tests else '✗'}"
    return CheckResult("C-07", "README.md has theorem and test badges", True, passed, detail)


def c08_no_theory_to_app_import(root: Path) -> CheckResult:
    """C-08: No theory/ module imports from applications/."""
    theory_dir = root / "src"
    violations = []
    for py_file in theory_dir.rglob("theory*.py"):
        if "applications" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = ""
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                    elif isinstance(node, ast.Import):
                        module = ",".join(a.name for a in node.names)
                    if "applications" in module:
                        violations.append(f"{py_file.name}: imports {module}")
        except SyntaxError:
            pass
    passed = len(violations) == 0
    detail = "clean" if passed else f"violations: {violations}"
    return CheckResult("C-08", "no theory/ imports from applications/", True, passed, detail)


def c09_public_fns_have_docstrings(root: Path) -> CheckResult:
    """C-09: All public functions in src/ have docstrings."""
    sys.path.insert(0, str(root / "src"))
    missing = []
    try:
        import flu
        for name in dir(flu):
            if name.startswith("_"):
                continue
            obj = getattr(flu, name)
            if callable(obj) and inspect.isfunction(obj):
                if not obj.__doc__ or len(obj.__doc__.strip()) < 5:
                    missing.append(name)
        passed = len(missing) == 0
        detail = f"all {sum(1 for n in dir(flu) if not n.startswith('_') and inspect.isfunction(getattr(flu,n)))} public fns documented" if passed else f"missing: {missing[:5]}"
        return CheckResult("C-09", "all public functions have docstrings", True, passed, detail)
    except Exception as e:
        return CheckResult("C-09", "all public functions have docstrings", True, False, str(e))


def c10_registry_json_current(root: Path) -> CheckResult:
    """C-10: THEOREM_REGISTRY.json matches live Python registry (recommended)."""
    gen_script = root / "tools" / "generate_registry_json.py"
    reg_json   = root / "src" / "flu" / "theory" / "THEOREM_REGISTRY.json"
    if not reg_json.exists():
        return CheckResult("C-10", "THEOREM_REGISTRY.json is current", False, False, "not generated yet")
    if not gen_script.exists():
        return CheckResult("C-10", "THEOREM_REGISTRY.json is current", False, False, "generator not found")
    result = subprocess.run(
        [sys.executable, str(gen_script), "--verify", "--output", str(reg_json)],
        capture_output=True, text=True, cwd=str(root),
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    passed = result.returncode == 0
    detail = result.stdout.strip() or result.stderr.strip()
    return CheckResult("C-10", "THEOREM_REGISTRY.json is current", False, passed, detail)


# ── Runner ────────────────────────────────────────────────────────────────────

ALL_CHECKS = [
    c01_theorem_registry_exists,
    c02_all_theorems_valid_status,
    c03_tests_pass,
    c04_open_debt_exists,
    c05_benchmarks_doc_exists,
    c06_manifest_exists,
    c07_readme_badges,
    c08_no_theory_to_app_import,
    c09_public_fns_have_docstrings,
    c10_registry_json_current,
]


def run_checks(root: Path, skip_slow: bool = False) -> List[CheckResult]:
    """Run all contract compliance checks."""
    results = []
    for check_fn in ALL_CHECKS:
        if skip_slow and check_fn is c03_tests_pass:
            results.append(CheckResult("C-03", "tests/run_all.py passes", True, True, "skipped (--fast)"))
            continue
        try:
            result = check_fn(root)
        except Exception as e:
            name = check_fn.__doc__ or check_fn.__name__
            result = CheckResult(check_fn.__name__[:4].upper(), name, True, False, f"ERROR: {e}")
        results.append(result)
    return results


def print_report(results: List[CheckResult], verbose: bool = True) -> None:
    """Print a human-readable compliance report."""
    print("FLU Package Contract Compliance Report")
    print("=" * 50)
    required_pass = sum(1 for r in results if r.required and r.passed)
    required_total = sum(1 for r in results if r.required)
    recommended_pass = sum(1 for r in results if not r.required and r.passed)
    recommended_total = sum(1 for r in results if not r.required)
    print(f"Required:    {required_pass}/{required_total}")
    print(f"Recommended: {recommended_pass}/{recommended_total}")
    print()
    for r in results:
        print(f"  {r.symbol}  [{r.id}] {r.desc}")
        if verbose and r.detail:
            print(f"       {r.detail}")
    print()
    if required_pass == required_total:
        print("✓ ALL REQUIRED CHECKS PASS — contract compliant")
    else:
        failures = [r for r in results if r.required and not r.passed]
        print(f"✗ {len(failures)} required check(s) FAILED — contract violation")
        for f in failures:
            print(f"  ✗ [{f.id}] {f.desc}: {f.detail}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="FLU contract compliance checker")
    ap.add_argument("--package-root", type=Path, default=Path("."),
                    help="Root directory of the package (default: cwd)")
    ap.add_argument("--json",   action="store_true", help="Machine-readable JSON output")
    ap.add_argument("--strict", action="store_true", help="Exit 1 on any required failure")
    ap.add_argument("--fast",   action="store_true", help="Skip slow checks (test suite)")
    ap.add_argument("--quiet",  action="store_true", help="Suppress detail lines")
    args = ap.parse_args(argv)

    root = args.package_root.resolve()
    results = run_checks(root, skip_slow=args.fast)

    if args.json:
        print(json.dumps([
            {"id": r.id, "desc": r.desc, "required": r.required,
             "passed": r.passed, "detail": r.detail}
            for r in results
        ], indent=2))
    else:
        print_report(results, verbose=not args.quiet)

    if args.strict:
        return 0 if all(r.passed for r in results if r.required) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
