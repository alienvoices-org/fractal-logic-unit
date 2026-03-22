# FLU File Integrity Protocol

**Version:** 15.1.4 baseline  
**Last updated:** 2026-03-12  
**Rationale:** In the V15 sprint, three source files (`genetic.py`, `gray_code.py`,
`cohomology.py`) were lost when the assistant rebuilt the package from memory rather
than from the canonical source. The output zip shrank from 445 KB to 150 KB without
any changelog entry noting removals. This document establishes a mandatory protocol
to prevent silent file loss in any future sprint.

---

## Canonical File Count (V15.1.4 baseline)

| Directory               | File count | Notes |
|-------------------------|------------|-------|
| `src/flu/core/`         | 12 .py     | Core algorithms (incl. `fractal_net.py`, `n_ary.py`, `parity_switcher.py`) |
| `src/flu/interfaces/`   | 13 .py     | All 11 bridge facets + `__init__.py` + `base.py` |
| `src/flu/theory/`       | 9 files    | 8 .py + `THEOREM_REGISTRY.json` |
| `src/flu/applications/` | 7 .py      | |
| `src/flu/container/`    | 6 .py      | |
| `src/flu/utils/`        | 5 .py      | |
| `tests/`                | 45 .py     | All test suites + benchmarks |
| `docs/`                 | 17 files   | All markdown docs + `PACKAGE_CONTRACT.json` |
| `tools/`                | 4 .py      | |
| `examples/`             | 4 .py      | |
| Root                    | 9 files    | `pyproject.toml`, `README.md`, `LICENSE`, `pytest_shim.py`, `requirements.txt`, `run_tests.py`, `CHANGELOG.md`, `FLU_MANIFEST.json`, `FLU_V15_handoff.json` |
| **Total**               | **≥ 136**  | Minimum expected file count (excluding temporary files) |

**Rule:** Any output zip with fewer than 133 files is INVALID and must not be delivered.

---

## Sprint Protocol

### BEFORE any sprint begins
1. Record the file count: `find . -type f | wc -l`
2. Record the zip size of the input package.
3. Note any files explicitly marked for deletion in the sprint brief.

### DURING the sprint
1. **Always work from the extracted source.** Never rebuild a module from memory.
2. To modify a file: extract → edit → keep in place.
3. To add a file: create in the correct directory.
4. **To delete a file:** only if explicitly requested by the user AND logged in CHANGELOG.md
   under a `### Deleted files` section with reason.

### AFTER the sprint (before repacking)
1. Run: `find . -type f | sort > /tmp/file_manifest_after.txt`
2. Diff against baseline. Any disappearance not in CHANGELOG `### Deleted files` = **STOP**.
3. Verify zip size: output zip must be ≥ 80% of input zip size (compression artefacts aside).
   If smaller, investigate before delivering.
4. Verify interface completeness: `src/flu/interfaces/` must contain all facet files +
   `__init__.py` + `base.py` (13 files minimum).

---

## The 13 Interface Facets (must ALL be present)

| File | Facet | Theorem | Status |
|------|-------|---------|--------|
| `hadamard.py` | `HadamardFacet` | HAD-1 | PROVEN |
| `crypto.py` | `CryptoFacet` | CRYPTO-1 | PROVEN |
| `design.py` | `DesignFacet` | T3 | DESIGN_INTENT |
| `digital_net.py` | `FractalNetCorputFacet` / `FractalNetKineticFacet` | FMD-NET / T9 | PROVEN |
| `genetic.py` | `GeneticFacet` | GEN-1 | PROVEN |
| `gray_code.py` | `GrayCodeFacet` | T8 | PROVEN |
| `hilbert.py` | `HilbertFacet` | HIL-1 | RETIRED |
| `integrity.py` | `IntegrityFacet` | INT-1 | PROVEN |
| `invariance.py` | `InvarianceFacet` | INV-1 | PROVEN |
| `lexicon.py` | `LexiconFacet` | LEX-1 | PROVEN |
| `neural.py` | `NeuralFacet` | PFNT-2 | DESIGN_INTENT |
| `cohomology.py` | `CohomologyFacet` | DEC-1 | PROVEN |

Plus `base.py` (abstract `FluFacet`) and `__init__.py` (re‑exports).

---

## What Triggered This Protocol

In the V15 packaging sprint, the assistant:
1. Received a 445 KB zip (132 source files).
2. Created only the 3 NEW interface files from memory.
3. Repacked only those 3 new files + `__init__.py` + docs.
4. Delivered a 150 KB zip with ~40 files — **92 files silently dropped.**

The `src/flu/core/`, `src/flu/theory/`, `src/flu/applications/`,
`src/flu/container/`, `tests/`, and `tools/` directories were all lost.

**Root cause:** The assistant worked in a scratch directory and never extracted
the original zip. It assembled the output from scratch instead of modifying
the extracted source.

**Fix:** Always `unzip` the input first. Never assemble from zero. Diff file counts
before delivering. If the source is missing or files look empty, scan the archive a second time.

---

## Explicit Discard Log

Intentional file removals must be logged here AND in CHANGELOG.md.

| Version | File removed | Reason | Authorised by |
|---------|-------------|--------|---------------|
| V15 sprint | `tests/test_theorem_computational_proofs.py` | Absorbed into `test_registry.py` and `test_proofs.py` | Session V15 audit |

---

## Version Discipline

### The Rule
**Never assume a major version bump.** Version changes follow this authority ladder:

| Change | Authority | Example |
|--------|-----------|---------|
| Major (V15 → V16) | **Explicit owner instruction only** | "We are releasing V16" |
| Minor (15.0 → 15.1) | **Explicit owner instruction only** | "This is a minor release" |
| Patch (15.0.0 → 15.0.1) | Sprint iteration — may increment freely | Bug fixes, sprint additions |

Sprints, audit waves, and feature additions within a release cycle stay on the
**same major.minor** and only bump the patch digit (or not at all if the owner
prefers a single release number for the whole cycle). Do not relabel documents,
comments, or changelog entries with a new major version until explicitly told.

### Single Source of Truth for Version Strings

**`src/flu/_version.py`** is the ONLY place to change the version number.
All other modules import from it:

from flu._version import __version__, FLU_VERSION, FLU_VERSION_LABEL

Files that may NOT contain hardcoded version literals:
- `src/flu/__init__.py` — import from `_version.py`
- `src/flu/theory/THEOREM_REGISTRY.json` — regenerated by `tools/generate_registry_json.py`
- `src/flu/interfaces/base.py` — import from `_version.py` if needed
- `src/flu/core/*.py`, `src/flu/theory/*.py` — import from `_version.py`

Files that MUST be updated manually in sync with `_version.py`:
- `pyproject.toml` — Python packaging tooling reads this directly (cannot import)
- `FLU_MANIFEST.json` — external consumer contract; update alongside pyproject.toml
- `README.md` — human-facing version badge

### What Triggered This Rule

In the V15 → "V16" incident, the assistant bumped the major version label in
CHANGELOG.md, fractal_net.py comments, theorem records, and OPEN_DEBT.md
**purely because it added a new class** in a sprint iteration. This caused
documentation drift: the package was still version 15.0.0 by every official
metric, but comments referred to "V16" in ~8 places, creating confusion about
what was actually released.

Sprint additions do not constitute a new major version. The owner decides
major version transitions.

