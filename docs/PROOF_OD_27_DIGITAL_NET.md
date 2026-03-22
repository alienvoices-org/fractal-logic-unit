# Proof: FractalNetKinetic is a (m(D−1), mD, D)-net in base n

**Theorem ID:** OD-27-PROOF  
**Status:** PROVEN (V15.2, 2026-03-14)  
**Closes:** OD-27 (digital net t-value classification)  
**Clarifies:** FMD-NET (corrects the "(0,D,D)-net" label)

---

## Statement

Let n ≥ 3 be odd, D ≥ 1, m ≥ 1. Let FractalNetKinetic(n, D) generate the first
n^{mD} points using m super-depths. Then this point set is a **(t, mD, D)-net in
base n with t = m(D−1)**, in the sense of Niederreiter (1992, Definition 4.1):
every elementary interval

    E = ∏_{j=0}^{D-1} [ c_j / n^{d_j},  (c_j + 1) / n^{d_j} )

with non-negative integer d_j satisfying Σ_j d_j = mD − t = m, contains exactly
n^t = n^{m(D−1)} points.

The t-value m(D−1) is tight: the property fails for t − 1.

---

## Proof

### 0. Setup and notation

**T matrix (D × D, lower-triangular over Z_n):**

    T[0, 0] = −1,   T[i, j] = 1  for j ≤ i with i ≥ 1,   T[i, j] = 0  for j > i.

`path_coord` computes x = T · a (mod n), shifted to signed range. The unsigned
output is (T · a) mod n ∈ {0, …, n−1}^D. T ∈ GL(D, Z_n) because det(T) = −1,
which is a unit for odd n.

**Point construction.** Super-block size N = n^D. The r-th super-digit of index
K is the unique v_r ∈ {0, …, N−1} with K ≡ v_r (mod N^{r+1}) / N^r. Its base-n
digit vector is a^(r) = (a^(r)_0, …, a^(r)_{D-1}) ∈ {0,…,n−1}^D. Define the
unsigned output digit b^(r)_j = (T · a^(r))_j mod n. Point K has coordinate

    X(K)[j] = Σ_{r=0}^{m-1}  b^(r)_j / n^{r+1}.

As K ranges over {0, …, n^{mD}−1}, the tuple (a^(0), …, a^(m-1)) ranges over all
n^{mD} elements of (Z_n^D)^m in a bijective correspondence.

---

### 1. Digit-constraint formulation

Point K lies in E iff for each j: b^(r)_j = c_{j,r} for r = 0, …, d_j − 1,
where c_{j,r} ∈ {0,…,n−1} is the r-th base-n digit of c_j / n^{d_j}. (This
follows from the unique base-n expansion of each coordinate.)

The counting problem is therefore:

> Count tuples (a^(0), …, a^(m-1)) ∈ (Z_n^D)^m satisfying
> (T · a^(r))_j ≡ c_{j,r} (mod n) for all j ∈ J_r,
> where J_r = { j : d_j > r } is the set of dimensions still constrained at depth r.

---

### 2. Depth decoupling

**Claim:** the constraints at distinct depths r are independent.

**Proof:** a^(r) (the r-th super-digit) appears only in the depth-r constraints.
No constraint mixes a^(r) with a^(r') for r' ≠ r. Because the tuples (a^(0), …,
a^(m-1)) are in bijection with K, the number of solutions is the product

    #{K in E} = ∏_{r=0}^{m-1} S_r,

where S_r = |{ a^(r) ∈ Z_n^D : (T · a^(r))_j ≡ c_{j,r} (mod n) for all j ∈ J_r }|.

---

### 3. T-Rank Lemma

**Lemma.** For any non-empty J = {j_1 < j_2 < … < j_k} ⊆ {0,…,D−1}, the
submatrix T_J (rows j_1, …, j_k of T) has rank k over Z_n.

**Proof.** Let A be the k × k matrix formed by columns j_1, …, j_k of T_J (i.e.
A_{α,β} = T_{j_α, j_β} for 1 ≤ α, β ≤ k). Because j_1 < … < j_k and T is
lower-triangular:

- α < β ⟹ j_α < j_β ⟹ T_{j_α, j_β} = 0  (upper entry).  A is lower-triangular.
- A_{α,α} = T_{j_α, j_α} ∈ {−1, 1}.  Both are units in Z_n (n odd, gcd(1,n)=1,
  gcd(n−1,n)=1).

A lower-triangular matrix with units on the diagonal has det(A) = ∏_α A_{α,α} ∈
{+1, −1}. Both are units in Z_n for any n ≥ 2. Therefore A is invertible over Z_n,
and T_J has rank k. □

---

### 4. Per-depth solution count

By the T-Rank Lemma, the linear system T_{J_r} · a^(r) ≡ c_r (mod n) in the D
unknowns a^(r) has a k × k invertible subsystem (columns J_r). We can freely
choose the remaining D − k = D − |J_r| unknowns (those NOT in J_r), and the
constrained ones are then uniquely determined. Therefore

    S_r = n^{D − |J_r|}.

---

### 5. Count identity for Σ d_j = m

**Key observation.** When Σ_j d_j = m and all d_j ≥ 0:

    Σ_r |J_r|  =  Σ_r |{j : d_j > r}|  =  Σ_j |{r ∈ {0,…,m−1} : r < d_j}|
                =  Σ_j min(d_j, m).

Since Σ_j d_j = m and d_j ≥ 0, we have d_j ≤ m for all j (if any d_j > m
then the sum already exceeds m, as all terms are non-negative). Therefore
min(d_j, m) = d_j, and Σ_r |J_r| = Σ_j d_j = m.

**Conclusion:**

    #{K in E} = ∏_r n^{D − |J_r|} = n^{mD − Σ_r |J_r|} = n^{mD − m} = n^{m(D−1)}.

This holds for every elementary interval with Σ d_j = m, regardless of the
particular values c_j or the distribution of d_j across dimensions. □

---

### 6. Tightness: t cannot be m(D−1) − 1

For t to equal m(D−1) − 1, we would need every interval with Σ d_j = m + 1 to
contain exactly n^{m(D−1)−1} points.

For D ≥ 2, consider the interval with d_0 = m + 1 and d_j = 0 for j ≥ 1
(Σ d_j = m + 1). This specifies m + 1 digits for coordinate 0 and no constraint
on coordinates 1, …, D−1.

Coordinate 0 only has m significant digits (super-depths r = 0, …, m−1 contribute
one digit each; at depth r ≥ m, the contribution is the constant base_block[0][0] / n^{r+1}
which is the same for all K). The (m+1)-th digit of every point is therefore a fixed
constant δ = (T · 0)_0 mod n = 0.

- If c_{0,m} = 0 (interval consistent with δ): the first m digit-constraints on
  coordinate 0 yield m free variables in the other D−1 dimensions per depth, giving
  a total of n^{m(D−1)} solutions. But n^{m(D−1)} ≠ n^{m(D−1)−1} for n ≥ 3.
- If c_{0,m} ≠ 0: 0 solutions. Also ≠ n^{m(D−1)−1}.

In both cases the required count n^{m(D−1)−1} is not achieved, so t − 1 fails. □

---

## Corollary: FractalNet has the same t-value

FractalNet uses generator matrix T = I (identity) instead of the FM-Dance T. The
T-Rank Lemma holds trivially for I (any subset of rows of I is clearly full rank),
so FractalNet is also a (m(D−1), mD, D)-net for the first n^{mD} points.

---

## Clarification of FMD-NET

FMD-NET states that FractalNet is a "(0, D, D)-net" and proves that every interval
with ALL d_j = 1 (the balanced case, Σ d_j = D) contains exactly 1 point. This
is correct and is precisely the Latin hypercube property (T3).

However, the standard Niederreiter (0, D, D)-net definition requires uniformity for
ALL intervals with Σ d_j = D, including unbalanced cases such as d_0 = 2, d_1 = 0.
An interval with d_0 = 2 > m = 1 has all of its m+1 digit specifications beyond the
net's resolution, causing counts of 0 or n^{D−1} (as shown empirically). The correct
classification is t = D − 1 (i.e., m = 1, t = m(D−1) = D−1).

**FMD-NET should be understood as proving the Latin hypercube / balanced-partition
property (one point per balanced interval), not a full (0, D, D)-net in the standard
sense.** The theorem is not false; it is a property of FractalNet. The "(0,D,D)-net"
label is an overstatement relative to the standard definition.

---

## Implications for OD-27 and the discrepancy bound

The exact t-value is t = m(D−1), which grows with m (i.e., with log N). This
means the naive t-based Niederreiter discrepancy bound D*_N ≤ C(t,D) (log N)^{D−1}/N
would incorporate an m-dependent constant. However, this does not contradict T9:
the Faure conjugacy argument gives D*_N = O((log N)^D / N) independently of the
t-value computation, because it inherits the Faure sequence discrepancy bound via
algebraic conjugacy (T = S·P·S^{−1}).

The t-value characterises the net structure at each fixed scale n^{mD}, while the
discrepancy bound is an asymptotic statement about the entire infinite sequence.
They are complementary, not competing, results.

---

## Computational verification

```python
# Verified n=3, D∈{1,2,3}, m∈{1,2}: t = m(D-1) in all cases.
# Tightness confirmed: property fails for t-1 via the d_0=m+1, d_j=0 witness.
# See tests/test_core/test_od27_net.py for the full regression suite.
```

---

## References

- Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods.* SIAM.
  Definition 4.1 (digital nets), Theorem 4.10 (Faure sequences).
- T1 — n-ary Bijection (proved in fm_dance.py)
- T3 — Latin Hypercube Property
- T9 — FM-Dance Digital Sequence Theorem (Faure Conjugacy, PROVEN V15)
- FMD-NET — FractalNet balanced-partition property (clarified above)
- DISC-1 — FM-Dance Discrete Integral Identity (T[0,0] = −1)
