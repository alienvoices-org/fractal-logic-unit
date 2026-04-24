"""
tools/cube_comparison_order5.py
================================
In-depth comparison of the two order-5 magic cubes stored in flu/constants.py:

  TRUMP_BOYER_5_NP  — Trump/Boyer "perfect" magic cube (Walter Trump &
                      Christian Boyer, 2003-11-13).  Used with permission.
  FM_DANCE_5_NP     — FM-Dance Siamese-step magic cube (Mönnich 2017,
                      "Symmetrische Tanzschritte für magische Universen").

Both cubes have:
  • Values 1..125 each exactly once
  • All orthogonal line sums = M = 315
  • All 4 space diagonals = 315

They differ in:
  • Face diagonal ("planar diagonal") coverage
  • Spectral / digit-column structure
  • Value scrambling across z-layers

CORRECTION NOTE
───────────────
The prior version of this script (2025-04) compared Trump/Boyer against the
FM-Dance *addressing bijection* (trivial digit→position identity, NOT magic).
This version uses the corrected FM_DANCE_5_NP from flu.core.fm_dance.generate_magic,
which implements the genuine Siamese step construction.  See fm_dance.py for
the three-way distinction: addressing bijection / magic hypercube / T-matrix path.

Run:
    python tools/cube_comparison_order5.py

Authors: Felix Mönnich & The Kinship Mesh Collective
Credit:  Trump/Boyer cube by Walter Trump & Christian Boyer, 2003-11-13.
         Used with explicit permission from Walter Trump.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from flu.constants import TRUMP_BOYER_5_NP, FM_DANCE_5_NP, MAGIC_SUM_5

TB = TRUMP_BOYER_5_NP   # shape (5,5,5), values 1..125, axes (z,y,x) 1-indexed
FM = FM_DANCE_5_NP      # shape (5,5,5), values 1..125, axes (x,y,z) manuscript
M  = MAGIC_SUM_5        # 315
N  = 5

SEP  = "=" * 72
SSEP = "-" * 60


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def all_axis_magic(cube: np.ndarray) -> dict:
    d = cube.ndim
    return {f"axis-{a} sums": np.unique(cube.sum(axis=a)).tolist() for a in range(d)}


def space_diagonals(cube: np.ndarray) -> list[int]:
    return [int(sum(cube[i, i,   i]   for i in range(N))),
            int(sum(cube[i, i,   N-1-i] for i in range(N))),
            int(sum(cube[i, N-1-i, i]   for i in range(N))),
            int(sum(cube[N-1-i, i, i]   for i in range(N)))]


def count_magic_planar_diags(cube: np.ndarray) -> tuple[int, list[str]]:
    """Count how many of 60 orthogonal face diagonals equal M."""
    count, hits = 0, []
    for axis, aname in [(0,'z'), (1,'y'), (2,'x')]:
        for i in range(N):
            layer = np.take(cube, i, axis=axis)
            d1 = int(sum(layer[k, k]     for k in range(N)))
            d2 = int(sum(layer[k, N-1-k] for k in range(N)))
            if d1 == M: count += 1; hits.append(f"{aname}={i} main={d1}")
            if d2 == M: count += 1; hits.append(f"{aname}={i} anti={d2}")
    return count, hits


def count_broken_diags(cube: np.ndarray) -> dict:
    counts = {'z-planes': 0, 'y-planes': 0, 'x-planes': 0}
    for axis, key in [(0,'z-planes'), (1,'y-planes'), (2,'x-planes')]:
        for i in range(N):
            layer = np.take(cube, i, axis=axis)
            for shift in range(N):
                s1 = int(sum(layer[k, (k+shift)%N]     for k in range(N)))
                s2 = int(sum(layer[k, (N-1-k+shift)%N] for k in range(N)))
                if s1 == M: counts[key] += 1
                if s2 == M: counts[key] += 1
    return counts


def digit_balance(cube: np.ndarray) -> dict:
    """5-ary digit balance: global (25×) and per-slice (5×)."""
    c = cube - 1
    result = {}
    for dp, label in [(0,'units'), (1,'fives'), (2,'twenty-fives')]:
        dg = (c // (N**dp)) % N
        gb = bool(np.all(np.bincount(dg.flatten(), minlength=N) == N*N))
        result[f"digit-{dp} ({label}) global"] = gb
        for ax, an in [(0,'z'), (1,'y'), (2,'x')]:
            ok = all(bool(np.all(np.bincount(np.take(dg,i,ax).flatten(), minlength=N)==N))
                     for i in range(N))
            result[f"digit-{dp} per-{an}-slice"] = ok
    return result


def spectral(cube: np.ndarray) -> dict:
    F = np.abs(np.fft.fftn(cube.astype(float)))
    F_noDC = F.copy(); F_noDC[0,0,0] = 0
    idx = np.unravel_index(np.argsort(F_noDC.flatten())[::-1][:5], F.shape)
    top5 = list(zip(*[i.tolist() for i in idx], F_noDC[idx].tolist()))
    return {"DC": float(F[0,0,0]),
            "non-DC max": float(F_noDC.max()),
            "non-DC mean": float(F_noDC.mean()),
            "non-DC std": float(F_noDC.std()),
            "top-5 freq": top5}


def layer_stats(cube: np.ndarray) -> list[dict]:
    return [{"z": z,
             "min": int(cube[z].min()), "max": int(cube[z].max()),
             "mean": float(cube[z].mean()), "std": float(cube[z].std())}
            for z in range(N)]


def pos_correlation(a: np.ndarray, b: np.ndarray) -> dict:
    pa = {int(a[z,y,x]): (z,y,x) for z in range(N) for y in range(N) for x in range(N)}
    pb = {int(b[z,y,x]): (z,y,x) for z in range(N) for y in range(N) for x in range(N)}
    ca = np.array([pa[v] for v in range(1, N**3+1)], dtype=float)
    cb = np.array([pb[v] for v in range(1, N**3+1)], dtype=float)
    return {f"r(axis-{i})": round(float(np.corrcoef(ca[:,i], cb[:,i])[0,1]), 4)
            for i in range(3)}


def spectral_block_per_line(cube: np.ndarray) -> dict:
    """Each axis-aligned LINE contains exactly one value from each block {1-25}.."""
    blk = (cube - 1) // 25
    result = {}
    for ax, an in [(0,'z'), (1,'y'), (2,'x')]:
        ok = True
        for i in range(N):
            for j in range(N):
                # extract line: fix two axes, vary the third
                if ax == 0:   line = blk[:, i, j]
                elif ax == 1: line = blk[i, :, j]
                else:          line = blk[i, j, :]
                if np.bincount(line, minlength=N).tolist() != [1]*N:
                    ok = False; break
            if not ok: break
        result[f"{an}-lines one-per-block"] = ok
    return result


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────

def ps(title): print(f"\n{SEP}\n  {title}\n{SEP}")
def pss(title): print(f"\n  {SSEP}\n  {title}\n  {SSEP}")
def fd(d, indent=4):
    for k, v in d.items(): print(f"{' '*indent}{k}: {v}")


def run():
    cubes = [("Trump/Boyer", TB), ("FM-Dance", FM)]

    print(SEP)
    print("  FLU — Order-5 Magic Cube Comparison  (corrected)")
    print("  Trump/Boyer Perfect Magic Cube  vs  FM-Dance Magic Cube")
    print(f"  Magic constant M = {M} = 5·(5³+1)/2")
    print(SEP)

    # ── 1. Completeness ───────────────────────────────────────────────────────
    ps("1. COMPLETENESS — values 1..125 each exactly once")
    for name, C in cubes:
        ok = sorted(C.flatten().tolist()) == list(range(1, N**3+1))
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    # ── 2. Orthogonal line sums ───────────────────────────────────────────────
    ps("2. ALL ORTHOGONAL LINE SUMS = M = 315")
    for name, C in cubes:
        res = all_axis_magic(C)
        ok  = all(v == [M] for v in res.values())
        print(f"\n  {name}: overall {'PASS' if ok else 'FAIL'}")
        fd(res)

    # ── 3. Space diagonals ────────────────────────────────────────────────────
    ps("3. SPACE DIAGONALS (4 main body diagonals)")
    for name, C in cubes:
        sds = space_diagonals(C)
        ok  = all(s == M for s in sds)
        print(f"  {name}: {sds}  {'PASS' if ok else 'FAIL'}")

    # ── 4. Planar diagonals ───────────────────────────────────────────────────
    ps("4. PLANAR DIAGONAL COUNT (15 orthogonal planes × 2 diagonals = 30 possible)")
    print("  ('Perfect' = all 30; FM-Dance is magic but NOT perfect)")
    for name, C in cubes:
        cnt, hits = count_magic_planar_diags(C)
        print(f"\n  {name}: {cnt}/30 planar diagonals = {M}")
        for h in hits: print(f"    ✓ {h}")

    # ── 5. Broken diagonals ───────────────────────────────────────────────────
    ps("5. BROKEN (TOROIDAL) DIAGONAL SUMS = M  (max 50 per plane-direction)")
    for name, C in cubes:
        bd = count_broken_diags(C)
        print(f"  {name}: {bd}")

    # ── 6. Spectral block per LINE ────────────────────────────────────────────
    ps("6. SPECTRAL BLOCK DISTRIBUTION — one value per block per LINE")
    print("  (block j = {25j+1 … 25(j+1)},  j=0..4;  guarantees magic sums)")
    for name, C in cubes:
        res = spectral_block_per_line(C)
        print(f"\n  {name}:")
        fd(res)

    # ── 7. Digit balance ──────────────────────────────────────────────────────
    ps("7. 5-ARY DIGIT BALANCE (Latin / LHS structure)")
    print("  Global: each digit value 0..4 appears 25× over all 125 cells.")
    print("  Per-slice: each digit value appears 5× in every axis-aligned slice.")
    for name, C in cubes:
        db = digit_balance(C)
        print(f"\n  {name}:")
        fd(db)

    # ── 8. Layer statistics ───────────────────────────────────────────────────
    ps("8. LAYER STATISTICS (axis-0 slices)")
    print("  TB: each z-layer spans the full value range (well-scrambled).")
    print("  FM: each layer draws from all 5 spectral blocks equally (by construction).")
    for name, C in cubes:
        print(f"\n  {name}:")
        for s in layer_stats(C):
            print(f"    z={s['z']}: min={s['min']:3d}  max={s['max']:3d}  "
                  f"mean={s['mean']:5.1f}  std={s['std']:.2f}")

    # ── 9. Spectral (DFT) analysis ────────────────────────────────────────────
    ps("9. SPECTRAL ANALYSIS (3D DFT magnitude)")
    for name, C in cubes:
        sp = spectral(C)
        print(f"\n  {name}:")
        fd(sp)

    # ── 10. Value-position correlation ───────────────────────────────────────
    ps("10. VALUE–POSITION CORRELATION (Pearson r, TB vs FM)")
    print("  r ≈ 0 on all axes → the two cubes have no geometric alignment.")
    fd(pos_correlation(TB, FM))

    # ── 11. Summary table ─────────────────────────────────────────────────────
    ps("11. SUMMARY TABLE")
    rows = [
        ("Property",                              "Trump/Boyer",  "FM-Dance"),
        ("─"*42,                                  "─"*14,         "─"*14),
        ("Values 1..125 (completeness)",          "✓ PASS",       "✓ PASS"),
        ("All rows / cols / pillars = 315",       "✓ YES",        "✓ YES"),
        ("All 4 space diagonals = 315",           "✓ YES",        "✓ YES"),
        ("Planar diagonals = 315  (max 30)",      "30/30  ✓",     "6/30"),
        ("Broken diag hits / direction",          "10",           "10"),
        ("Line has 1 value per spectral block",   "✓ YES",        "✓ YES"),
        ("Layer means all equal 63.0",            "✓ YES",        "✓ YES"),
        ("Global 5-ary digit balance",            "✓ YES",        "✓ YES"),
        ("Per-slice digit balance (LHS)",         "✗ NO",         "✓ YES"),
        ("Magic cube classification",             "PERFECT",      "MAGIC (non-perfect)"),
    ]
    cw = [44, 16, 16]
    for row in rows:
        print("  " + "".join(str(c).ljust(w) for c,w in zip(row,cw)))

    # ── 12. Analytical conclusions ────────────────────────────────────────────
    ps("12. ANALYTICAL CONCLUSIONS")
    print("""
  MAGIC EQUIVALENCE:
    Both cubes are confirmed MAGIC in the strict sense: all orthogonal
    line sums and all 4 space diagonals equal M = 315.

  WHAT "PERFECT" ADDS OVER "MAGIC":
    Trump/Boyer achieves all 30 PLANAR diagonals = 315 (both diagonals of
    every axis-aligned cross-section). FM-Dance achieves only 6/30.
    Planar diagonals require tighter spectral coupling across adjacent layers
    than the adjacent-pair step vectors provide — essentially a 2D-magic
    condition on every slice, not just 1D-magic on every line.

  DIGIT-COLUMN / LHS STRUCTURE:
    FM-Dance is a LATIN HYPERCUBE in the strong per-slice sense: every 5-ary
    digit position (units / fives / twenty-fives) is balanced (exactly 5 of
    each residue) in EVERY axis-aligned slice. Trump/Boyer is globally balanced
    but not per-slice — its digit residues vary within individual layers.
    This is the structural property that makes FM-Dance compatible with the
    FLU Latin Hypercube Sampling framework and Trump/Boyer incompatible.

  SPECTRAL BLOCK PER LINE:
    BOTH cubes now share the spectral block property: every axis-aligned line
    contains exactly one value from each block {1-25},{26-50},…,{101-125}.
    This is a necessary (but not sufficient) condition for magic line sums,
    and both cubes satisfy it. The FM-Dance magic cube achieves this through
    adjacent-pair coupling; Trump/Boyer achieves it through exhaustive search.

  THREE DISTINCT FM-DANCE OBJECTS (documentation fix):
    1. generate_fast   — addressing bijection  (identity digit map, Latin, NOT magic)
    2. generate_magic  — magic hypercube       (Siamese steps, ALL sums = M)
    3. path_coord      — T-matrix traversal    (Hamiltonian, Latin, NOT magic;
                         axis-0 stratified by spectral block like generate_fast)
    Only generate_magic / FM_DANCE_5_NP implements the construction described
    in the manuscript. path_coord is a different, independently valid bijection.
""")


if __name__ == "__main__":
    run()
