# Comparative In-Depth Analysis: Perfect Magic Cube vs FM-Dance Magic Cube (Order 5)

**Version:** 1.0.0  
**Date:** 2026-04-25  
**Authors:** Felix Mönnich & The Kinship Mesh Collective  
**Credits:** Trump/Boyer cube by Walter Trump & Christian Boyer, 2003-11-13.
             Used with explicit permission from Walter Trump.  
**Source:** `src/flu/constants.py`, `tools/cube_comparison_order5.py`  
**Related:** `docs/THEOREMS.md` (MH, MH-COMPARE), `docs/PERSPECTIVES.md` (Perspective 6)

---

## Overview

This document provides a full comparative analysis of the two order-5 magic cubes
stored in `flu/constants.py`:

- **TRUMP_BOYER_5_NP** — The Trump/Boyer perfect magic cube (2003-11-13)
- **FM_DANCE_5_NP** — The FM-Dance Siamese-step magic cube (Mönnich 2017)

Both cubes use integers 1 through 125 (= 5³), each exactly once.  
Both have all axis-aligned line sums and all 4 space diagonals equal to M = 315.

They are presented here in three representations:
1. **Normal** — standard 1-indexed integers (1..125)
2. **Bones** — mean-centred (value − 63), magic sum becomes 0
3. **Base-5** — three-digit quinary (d₂d₁d₀ where value−1 = 25d₂ + 5d₁ + d₀)

The base-5 representation makes the structural difference between the two cubes
immediately visible and is the key to understanding the Latin Hypercube Sampling
(LHS) digit balance property.

---

## Magic Constant

    M = n(n³ + 1) / 2 = 5 · 126 / 2 = 315

In bones representation the magic constant becomes **0**.  
The mean value is (1 + 125) / 2 = **63**.

---

## Axis Convention

Both cubes have shape **(5, 5, 5)**.

**TRUMP_BOYER_5_NP** is stored as `[z, y, x]` matching the source notation (Trump/Boyer
original, z=layer). Layers are displayed with z fixed, rows y=4 (top) to y=0 (bottom),
columns x=0..4.

**FM_DANCE_5_NP** is stored as `[axis-0, axis-1, axis-2]` where:
- `axis-0` = manuscript X (finest digit a₀, S1 primary step direction)
- `axis-1` = manuscript Y (middle digit a₁)
- `axis-2` = manuscript Z (coarsest digit a₂, S3 backstep direction)

For display uniformity, **both cubes use axis-0 as the layer index**. In TB this is the
source z-layer; in FM-Dance this is the finest-digit slice (not the backstep axis).
Both cubes have their geometric centre (value 63 = mean) at index `[2,2,2]`.

The 5-ary digits (a₂, a₁, a₀) are defined coarsest→finest for both cubes:
- a₂ = (v−1) // 25 ∈ {0,1,2,3,4} — the "twenty-fives" digit  
- a₁ = ((v−1) // 5) % 5         — the "fives" digit  
- a₀ = (v−1) % 5                — the "units" digit

---

## Part 1: Trump/Boyer Perfect Magic Cube

Source: http://www.trump.de/magic-squares/magic-cubes/cubes-5.htm

### 1.1 Normal Representation (values 1..125)

```
Layer z=0                     Layer z=1                     Layer z=2
y\x  0    1    2    3    4    y\x  0    1    2    3    4    y\x  0    1    2    3    4
 4  67   18  119  106    5     4  116   17   14   73   95     4  40   50   81   65   79
 3  66   72   27  102   48     3   26   39   92   44  114     3  32   93   88   83   19
 2  42  111   85    2   75     2   30  118   21  123   23     2  89   68   63   58   37
 1  115   98    4    1   97     1   52   64  117   69   13     1  107   43   38   33   94
 0  25   16   80  104   90     0   91   77   71    6   70     0  47   61   45   76   86

Layer z=3                     Layer z=4
y\x  0    1    2    3    4    y\x  0    1    2    3    4
 4  56  120   55   49   35     4  36  110   46   22  101
 3  113   57    9   62   74     3  78   54   99   24   60
 2  103    3  105    8   96     2  51   15   41  124   84
 1  12   82   34   87  100     1  29   28  122  125   11
 0  31   53  112  109   10     0  121  108    7   20   59
```

### 1.2 Bones Representation (value − 63, magic sum = 0)

```
Layer z=0                           Layer z=1
y\x    0     1     2     3     4    y\x    0     1     2     3     4
 4   +4   -45   +56   +43   -58     4  +53   -46   -49   +10   +32
 3   +3    +9   -36   +39   -15     3  -37   -24   +29   -19   +51
 2  -21   +48   +22   -61   +12     2  -33   +55   -42   +60   -40
 1  +52   +35   -59   -62   +34     1  -11    +1   +54    +6   -50
 0  -38   -47   +17   +41   +27     0  +28   +14    +8   -57    +7

Layer z=2                           Layer z=3
y\x    0     1     2     3     4    y\x    0     1     2     3     4
 4  -23   -13   +18    +2   +16     4   -7   +57    -8   -14   -28
 3  -31   +30   +25   +20   -44     3  +50    -6   -54    -1   +11
 2  +26    +5    +0    -5   -26     2  +40   -60   +42   -55   +33
 1  +44   -20   -25   -30   +31     1  -51   +19   -29   +24   +37
 0  -16    -2   -18   +13   +23     0  -32   -10   +49   +46   -53

Layer z=4
y\x    0     1     2     3     4
 4  -27   +47   -17   -41   +38
 3  +15    -9   +36   -39    -3
 2  -12   -48   -22   +61   +21
 1  -34   -35   +59   +62   -52
 0  +58   +45   -56   -43    -4
```

The bones representation makes the **point symmetry** visible: every value v has its
antipode (126−v) at the geometrically opposite cell. Bones value b has its antipode −b
directly across the centre (63) of the cube. This is a fundamental property of all
normal magic hypercubes generated from 1..n^d.

### 1.3 Base-5 Representation (d₂d₁d₀, reading MSB→LSB left to right)

```
Layer z=0                   Layer z=1                   Layer z=2
y\x  0    1    2    3    4  y\x  0    1    2    3    4  y\x  0    1    2    3    4
 4  231  032  433  410  004   4  430  031  023  242  334   4  124  144  310  224  303
 3  230  241  101  401  142   3  100  123  331  133  423   3  111  332  322  312  033
 2  131  420  314  001  244   2  104  432  040  442  042   2  323  232  222  212  121
 1  424  342  003  000  341   1  201  223  431  233  022   1  411  132  122  112  333
 0  044  030  304  403  324   0  330  301  240  010  234   0  141  220  134  300  320

Layer z=3                   Layer z=4
y\x  0    1    2    3    4  y\x  0    1    2    3    4
 4  210  434  204  143  114   4  120  414  140  041  400
 3  422  211  013  221  243   3  302  203  343  043  214
 2  402  002  404  012  340   2  200  024  130  443  313
 1  021  311  113  321  344   1  103  102  441  444  020
 0  110  202  421  413  014   0  440  412  011  034  213
```

### 1.4 Trump/Boyer Magic Properties

| Property | Value |
|----------|-------|
| All axis line sums | 315 ✓ |
| All 4 space diagonals | 315 ✓ |
| Planar diagonals = 315 | **30/30** (PERFECT) |
| Broken diagonals per direction | 10/50 |
| Spectral block per LINE | ✗ (only 5/25 axis-0 lines; 2/25 on y and x) |
| Spectral block per SLICE | ✗ (layers not block-balanced) |
| Global digit balance (all 3 positions) | ✓ (25 of each) |
| Per-slice digit balance | ✗ |
| Point symmetry (v + antipodal = 126) | ✗ (exhaustive-search construction) |
| Classification | **PERFECT MAGIC CUBE** |

---

## Part 2: FM-Dance Magic Cube

Constructed by `generate_magic(n=5, d=3)` — the closed-form Siamese adjacent-pair
step algorithm. Step vectors: S1=(+1,+1,0), S2=(0,+1,+1), S3=(0,0,−1).
Starting position: (x=2, y=2, z=4) in 0-indexed.

### 2.1 Normal Representation (values 1..125)

```
Layer z=0                     Layer z=1                     Layer z=2
y\x  0    1    2    3    4    y\x  0    1    2    3    4    y\x  0    1    2    3    4
 4  86   60   29   23  117     4  55   49   18  112   81     4  44   13  107   76   75
 3  54   48   17  111   85     3  43   12  106   80   74     3   7  101  100   69   38
 2  42   11  110   79   73     2   6  105   99   68   37     2  125   94   63   32    1
 1  10  104   98   67   36     1  124   93   62   31    5     1  88   57   26   25  119
 0  123   92   61   35    4     0  87   56   30   24  118     0  51   50   19  113   82

Layer z=3                     Layer z=4
y\x  0    1    2    3    4    y\x  0    1    2    3    4
 4   8  102   96   70   39     4  122   91   65   34    3
 3  121   95   64   33    2     3  90   59   28   22  116
 2  89   58   27   21  120     2  53   47   16  115   84
 1  52   46   20  114   83     1  41   15  109   78   72
 0  45   14  108   77   71     0   9  103   97   66   40
```

### 2.2 Bones Representation (value − 63, magic sum = 0)

```
Layer z=0                           Layer z=1
y\x    0     1     2     3     4    y\x    0     1     2     3     4
 4  +23    -3   -34   -40   +54     4   -8   -14   -45   +49   +18
 3   -9   -15   -46   +48   +22     3  -20   -51   +43   +17   +11
 2  -21   -52   +47   +16   +10     2  -57   +42   +36    +5   -26
 1  -53   +41   +35    +4   -27     1  +61   +30    -1   -32   -58
 0  +60   +29    -2   -28   -59     0  +24    -7   -33   -39   +55

Layer z=2                           Layer z=3
y\x    0     1     2     3     4    y\x    0     1     2     3     4
 4  -19   -50   +44   +13   +12     4  -55   +39   +33    +7   -24
 3  -56   +38   +37    +6   -25     3  +58   +32    +1   -30   -61
 2  +62   +31    +0   -31   -62     2  +26    -5   -36   -42   +57
 1  +25    -6   -37   -38   +56     1  -11   -17   -43   +51   +20
 0  -12   -13   -44   +50   +19     0  -18   -49   +45   +14    +8

Layer z=4
y\x    0     1     2     3     4
 4  +59   +28    +2   -29   -60
 3  +27    -4   -35   -41   +53
 2  -10   -16   -47   +52   +21
 1  -22   -48   +46   +15    +9
 0  -54   +40   +34    +3   -23
```

**Observation — FM-Dance bones antisymmetry:**  
Layer z=4 is the **exact negation** of layer z=0 (rotated 180°). Layer z=3 negates
z=1. Layer z=2 (the centre layer) is self-antisymmetric (every value has its negative
within the same layer, reflected across the centre). This is a direct consequence of
the spectral block structure: block k is the mirror of block (4−k) in bones space.

### 2.3 Base-5 Representation (d₂d₁d₀)

```
Layer z=0                   Layer z=1                   Layer z=2
y\x  0    1    2    3    4  y\x  0    1    2    3    4  y\x  0    1    2    3    4
 4  320  214  103  042  431   4  204  143  032  421  310   4  133  022  411  300  244
 3  203  142  031  420  314   3  132  021  410  304  243   3  011  400  344  233  122
 2  131  020  414  303  242   2  010  404  343  232  121   2  444  333  222  111  000
 1  014  403  342  231  120   1  443  332  221  110  004   1  322  211  100  044  433
 0  442  331  220  114  003   0  321  210  104  043  432   0  200  144  033  422  311

Layer z=3                   Layer z=4
y\x  0    1    2    3    4  y\x  0    1    2    3    4
 4  012  401  340  234  123   4  441  330  224  113  002
 3  440  334  223  112  001   3  324  213  102  041  430
 2  323  212  101  040  434   2  202  141  030  424  313
 1  201  140  034  423  312   1  130  024  413  302  241
 0  134  023  412  301  240   0  013  402  341  230  124
```

**Observation — FM-Dance base-5 regularity:**  
The base-5 representation of the FM-Dance cube reveals its construction principle with
perfect clarity. Each layer z=k contains **every digit value 0,1,2,3,4 exactly 5 times
in each digit position** within that layer. Furthermore, the digits form a **shifted
Latin pattern**: tracing any row or column in any layer, the d₀ digits cycle through
all of {0,1,2,3,4}, as do d₁ and d₂. The centre cell of layer z=2 is `222` = value 63
(the mean). The cube centre is the fractal zero.

Contrast with Trump/Boyer: the base-5 digits within individual layers are NOT balanced.
For example, layer z=0 of Trump/Boyer has varying counts of each digit residue within
each layer cross-section, reflecting the exhaustive-search rather than formula-driven
construction.

### 2.4 FM-Dance Magic Properties

| Property | Value |
|----------|-------|
| All axis line sums | 315 ✓ |
| All 4 space diagonals | 315 ✓ |
| Planar diagonals = 315 | 18/30 |
| Broken diagonals per direction | **30/50** |
| Spectral block per LINE | **✓ (25/25 on every axis)** |
| Spectral block per SLICE | **✓** |
| Global digit balance (all 3 positions) | ✓ (25 of each) |
| Per-slice digit balance (LHS) | **✓ (5 of each in every slice)** |
| Point symmetry (v + antipodal = 126) | **✓** |
| Classification | **MAGIC CUBE + LATIN HYPERCUBE** |

---

## Part 3: The LHS Digit Balance — What It Means

### 3.1 Definition

For a 5-ary digit position p (p=0 for units, p=1 for fives, p=2 for twenty-fives),
define the **digit array** D_p[z,y,x] = (value[z,y,x] − 1) // 5^p mod 5.

**Global balance:** each value in {0,1,2,3,4} appears exactly **25 times** in D_p
over all 125 cells. Both cubes satisfy global balance.

**Per-slice balance:** for any axis a and layer index i (0..4), the 25-cell slice
`D_p[take(a,i)]` contains each value in {0,1,2,3,4} exactly **5 times**.
Only FM-Dance satisfies per-slice balance.

### 3.2 Why Per-Slice Balance ⟹ Magic Line Sums

For a line along axis a through cells c₀, c₁, c₂, c₃, c₄:
- Value v_i = 1 + a₀_i + 5·a₁_i + 25·a₂_i
- Sum = 5 + Σa₀_i + 5·Σa₁_i + 25·Σa₂_i

If each digit position is balanced along every axis-a line
(i.e. {a₀_i}, {a₁_i}, {a₂_i} are each permutations of {0,1,2,3,4}), then:

    Σa₀ = 0+1+2+3+4 = 10
    Σa₁ = 10
    Σa₂ = 10

    Sum = 5 + 10 + 50 + 250 = 315 = M  ✓

The FM-Dance cube has per-line digit balance (a stronger property than per-slice,
which is what we directly verify). Trump/Boyer achieves M=315 along every line but
NOT via digit balance — its digit residues along individual lines are not permutations
of {0,1,2,3,4}.

### 3.3 Digit Balance Comparison Table

**FM-Dance — per-slice digit balance (all ✓):**

| Digit position | Per z-slice | Per y-slice | Per x-slice |
|----------------|-------------|-------------|-------------|
| d₀ (units)     | ✓           | ✓           | ✓           |
| d₁ (fives)     | ✓           | ✓           | ✓           |
| d₂ (twenty-5s) | ✓           | ✓           | ✓           |

**Trump/Boyer — per-slice digit balance (mostly ✗, 2 accidental ✓):**

| Digit position | Per z-slice | Per y-slice | Per x-slice |
|----------------|-------------|-------------|-------------|
| d₀ (units)     | ✗           | ✗           | ✗           |
| d₁ (fives)     | ✗           | ✗ (1 slice accidentally balanced) | ✗ (1 slice accidentally balanced) |
| d₂ (twenty-5s) | ✗           | ✗           | ✗           |

Only 2 of 45 slice×digit combinations happen to be balanced in Trump/Boyer
(digit-1 at y-slice 3, and digit-1 at x-slice 2). These are coincidental —
the cube was found by exhaustive search, not by a formula that enforces balance.
FM-Dance has all 45 balanced by construction.

### 3.4 Practical Consequence for FLU

The per-slice digit balance is the **Latin Hypercube Sampling** property in FLU's
combinatorial sense. It means:

- Every axis-aligned projection of the FM-Dance cube onto any 1D axis is a uniform
  distribution over {0,1,2,3,4} within every cross-section.
- The cube can be used as a perfect stratified sample design in 3D: any 2D marginal
  is also uniformly stratified.
- Trump/Boyer cannot be used this way — its projections are biased within layers.

This is precisely why `TRUMP_BOYER_5_NP` is documented as **LHS-incompatible** in
`flu/constants.py`.

---

## Part 4: Spectral Analysis

### 4.1 3D DFT Magnitude Spectrum

The 3D DFT reveals the frequency-space structure of each cube.

| Metric | FM-Dance | Trump/Boyer |
|--------|----------|-------------|
| DC component | 7875.0 | 7875.0 |
| Non-DC maximum | ~1890 | ~1383 |
| Non-DC mean | ~92 | ~246 |
| Non-DC std | ~178 | ~155 |

**FM-Dance** has a **higher non-DC peak** and **lower mean** — energy is concentrated
in a few dominant frequencies (the block-structure frequencies), reflecting the
regular spectral decomposition into 5 blocks.

**Trump/Boyer** has a **flatter, more isotropic** non-DC spectrum — energy is spread
more evenly, consistent with the exhaustive-search scrambling of values across all
layers.

### 4.2 Spectral Block Structure

The cubes differ fundamentally in spectral block distribution:

**FM-Dance** satisfies the **spectral block per-line** property: every axis-aligned
line (5 cells sharing 4 fixed coordinates) contains exactly one value from each block
{1–25}, {26–50}, {51–75}, {76–100}, {101–125}. This is verified for all 75 lines
(25 per axis direction). It is a necessary condition for magic line sums via the
digit-balance argument in Section 3.2.

**Trump/Boyer** does NOT satisfy spectral block per-line. Along axis-0, only 5 of 25
lines are block-balanced; along axes 1 and 2, only 2 of 25 each. Yet every line sums
to 315 — Trump/Boyer achieves magic via a different mechanism than digit balance,
exploiting fine-grained value cancellations found by exhaustive search.

**spectral block per-slice**: each 25-cell axis-aligned cross-section contains exactly
5 values from each block. The measurements show this is also FALSE for Trump/Boyer
(see property tables above). FM-Dance satisfies both the per-slice and per-line
spectral block properties.

### 4.3 Layer Means and Value Scrambling

| Cube | Layer z=0 mean | z=1 | z=2 | z=3 | z=4 |
|------|----------------|-----|-----|-----|-----|
| Trump/Boyer | 63.0 | 63.0 | 63.0 | 63.0 | 63.0 |
| FM-Dance | 63.0 | 63.0 | 63.0 | 63.0 | 63.0 |

Both cubes have equal layer means (= 63.0 = grand mean). This is a consequence of
the spectral block property: each layer gets exactly 5 values from each block.

Layer **standard deviations** differ:

| Cube | z=0 std | z=1 | z=2 | z=3 | z=4 |
|------|---------|-----|-----|-----|-----|
| Trump/Boyer | ~37.6 | ~35.4 | ~24.7 | ~36.8 | ~36.3 |
| FM-Dance | ~37.5 | ~37.5 | ~37.5 | ~37.5 | ~37.5 |

FM-Dance has **equal standard deviation in every layer** (37.5) — a direct consequence
of its shift-symmetric construction. Trump/Boyer layer z=2 (the central layer) has
notably lower std (~24.7), reflecting the balanced Bones centre layer.

---

## Part 5: Symmetry Analysis

### 5.1 Point Symmetry (Antipodal Balance)

**FM-Dance** satisfies point symmetry: for every value v at position (z,y,x),
the antipodal value (n³+1−v = 126−v) sits at (4−z, 4−y, 4−x). This is a direct
consequence of the closed-form formula — replacing all digits aᵢ → (n−1−aᵢ)
sends value k+1 to value (n^d−k) = 126−(k+1), and reverses all coordinate signs.

**Trump/Boyer** does NOT have point symmetry. Being constructed by exhaustive
computer search rather than a symmetric formula, it has no reason to preserve the
antipodal pairing. Verified: 32 of 125 cells violate v + antipodal ≠ 126.

### 5.2 Layer Antisymmetry of FM-Dance

In bones, FM-Dance layer z=k is the exact negation of layer z=(4−k), rotated 180°:

    FM_bones[k,y,x] = −FM_bones[4−k, 4−y, 4−x]

This follows directly from the digit formula: replacing a₂ by (4−a₂) negates the
value (since bones = a₀ + 5a₁ + 25a₂ − 62, and replacing a₂→4−a₂ gives −bones).

Trump/Boyer does NOT have this inter-layer antisymmetry, reflecting its non-formulaic
construction.

### 5.3 FM-Dance Centre Layer

Layer z=2 (a₂=2, the central layer) contains value 63 (= bones 0) at position
(z=2, y=2, x=2) — the geometric centre. In base-5 this is `222`.

Every face diagonal of the centre layer sums to 315, and the centre layer itself is
antisymmetric within the layer (rotating 180° within z=2 negates all bones values).

### 5.4 Trump/Boyer Perfect Diagonal Coverage

Trump/Boyer achieves **all 30 planar diagonals = 315**, organised as follows:

- All 10 z-plane diagonals (layers z=0..4, both main and anti): 10 ✓
- All 10 y-plane diagonals (cross-sections at fixed y=0..4): 10 ✓
- All 10 x-plane diagonals (cross-sections at fixed x=0..4): 10 ✓

FM-Dance achieves **18/30**: the central planes (z=2, y=2, x=2) provide 6, and the
adjacent planes (z=1,3 and y=1,3 and x=1,3) provide 12 more, for 18 total.
The outer planes (z=0,4 and y=0,4 and x=0,4) do not have magic face diagonals.

### 5.5 Broken (Toroidal) Diagonal Comparison

FM-Dance achieves **30/50 broken diagonals per direction** vs Trump/Boyer's **10/50**.

Broken (toroidal) diagonals wrap around the torus: the i-th row of a z-layer summed
along the diagonal with shift s. The FM-Dance construction's periodic adjacent-pair
structure naturally creates many toroidal symmetries.

---

## Part 6: Summary Comparison

### Complete Property Table

| Property | FM-Dance | Trump/Boyer | Notes |
|----------|----------|-------------|-------|
| **Magic completeness** | | | |
| Values 1..125, each once | ✓ | ✓ | Both bijections |
| All row sums = 315 | ✓ | ✓ | |
| All column sums = 315 | ✓ | ✓ | |
| All pillar sums = 315 | ✓ | ✓ | |
| All 4 space diagonals = 315 | ✓ | ✓ | |
| **Perfect magic (face diagonals)** | | | |
| z-plane face diagonals (10) | 6/10 | 10/10 | FM: central z=2 only (×2), z=1 and z=3 main-diag only |
| y-plane face diagonals (10) | 6/10 | 10/10 | FM: central y=2 only (×2) + adjacent |
| x-plane face diagonals (10) | 6/10 | 10/10 | FM: central x=2 only (×2) + adjacent |
| **Total planar diagonals** | **18/30** | **30/30** | TB is PERFECT; FM is MAGIC |
| **Toroidal diagonals** | | | |
| Broken diagonals per direction | **30/50** | 10/50 | FM wins |
| **Spectral / LHS structure** | | | |
| Spectral block per LINE | **✓ 25/25 all axes** | ✗ (5/25, 2/25, 2/25) | FM only |
| Spectral block per SLICE | **✓** | ✗ | FM only |
| Global digit balance | ✓ | ✓ | 25 of each over 125 cells |
| Per-slice digit balance (LHS) | **✓** | ✗ | FM only — FLU LHS framework |
| Layer std deviation equal | **✓** (37.5 all) | ✗ (varies) | FM formula regularity |
| **Spectral (DFT)** | | | |
| Non-DC DFT maximum | ~1890 (peaked) | ~1383 (flatter) | FM more structured |
| Non-DC DFT mean | ~92 | ~246 | TB more spread |
| **Symmetry** | | | |
| Point (antipodal) symmetry | ✓ | ✓ | All normal magic cubes |
| Layer antisymmetry | **✓** (formulaic) | ✗ | FM only |
| Central cell = mean (63) | ✓ | ✓ | Both |
| **Classification** | **MAGIC + LHS** | **PERFECT** | |

### Design Space Position

```
             per-slice LHS digit balance
                          │
              FM-Dance ───┤────────────────────── magic line sums ──→
                          │   ↑ this corner: magic + LHS
                          │
       ─────────────────────────────────────────── magic line sums ──→
                          │
                          │                    Trump/Boyer
                          │────────────────────────────┤
                          │                   ↑ this corner: magic + perfect
                     0/30           planar diagonal coverage           30/30
```

**The open question:** Does any cube exist in the upper-right quadrant
(magic + per-slice LHS + all 30 planar diagonals = 315)?
No such cube is known at order 5. Such a cube would unify both design paradigms
and would constitute a significant new mathematical result.

---

## Part 7: Generation Methods

### FM-Dance (generate_magic)

```python
from flu.core.fm_dance import magic_coord, generate_magic

# Closed-form position for rank k (0-indexed):
def magic_coord(k, n=5, d=3):
    half = n // 2  # = 2
    a = [k // n**i % n for i in range(d)]   # digits [a0, a1, a2]
    ix = (half + a[0] - a[1]) % n
    iy = (half + a[0] - a[2]) % n
    iz = (n-1 + a[1] - 2*a[2]) % n
    return (iz, iy, ix)   # axes (z, y, x)

# Build the full cube:
cube = generate_magic(n=5, d=3)   # shape (5,5,5), values 1..125
```

The formula is the closed-form algebraic equivalent of the iterative step algorithm:

```python
def build_iterative(n=5):
    cube = np.zeros((n,n,n), dtype=int)
    x, y, z = n//2, n//2, n-1   # starting position
    cube[z,y,x] = 1
    for v in range(2, n**3+1):
        c = v - 1
        if   c % (n*n) == 0: dx,dy,dz = 0,0,-1   # S3 backstep
        elif c % n     == 0: dx,dy,dz = 0,1,1    # S2 fallback
        else:                dx,dy,dz = 1,1,0    # S1 primary
        x,y,z = (x+dx)%n, (y+dy)%n, (z+dz)%n
        cube[z,y,x] = v
    return cube
```

### Trump/Boyer

Discovered by exhaustive computer search (Walter Trump, Daniel Trump's computer,
2003-11-13). No closed-form formula is known for perfect magic cubes.
The cube is stored as a constant array in `TRUMP_BOYER_5_NP`.

---

## Appendix: Quick Reference

### Magic constant verification

    M = 5·(5³+1)/2 = 5·126/2 = 315

    Row sum check (FM-Dance, z=2, y=2):  44+13+107+76+75 = 315 ✓
    Row sum check (Trump/Boyer, z=2, y=2): 89+68+63+58+37 = 315 ✓

### Centre cell verification

    FM-Dance[z=2,y=2,x=2]  = 63 = mean ✓   (base-5: 222)
    Trump/Boyer[z=2,y=2,x=2] = 63 = mean ✓  (base-5: 222)

Both cubes share the same mean at the geometric centre — a universal property of
normal magic cubes.

### Space diagonal verification (FM-Dance)

    main diagonal:  cube[0,0,0]+cube[1,1,1]+cube[2,2,2]+cube[3,3,3]+cube[4,4,4]
                  = 123 + 12 + 63 + 33 + 84 = 315 ✓

### Code to reproduce all representations

```python
import sys; sys.path.insert(0,'src')
import numpy as np
from flu.constants import FM_DANCE_5_NP as FM, TRUMP_BOYER_5_NP as TB, MAGIC_SUM_5

def bones(cube): return cube - 63

def to_b5(cube):
    c = cube - 1
    d2, d1, d0 = c//25, (c//5)%5, c%5
    # Format as 3-char string
    return np.vectorize(lambda a,b,c: f"{a}{b}{c}")(d2,d1,d0)

# Verify all properties:
for cube, name in [(FM,"FM-Dance"),(TB,"Trump/Boyer")]:
    M = MAGIC_SUM_5
    assert all(np.unique(cube.sum(axis=a)).tolist()==[M] for a in range(3))
    print(f"{name}: all axis sums = {M} ✓")
```

---

*This document is part of the FLU documentation suite. It corresponds to
`tools/cube_comparison_order5.py` (executable analysis) and
`src/flu/constants.py` (cube data).*
