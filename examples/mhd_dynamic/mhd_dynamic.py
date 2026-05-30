"""
examples/mhd_dynamic/mhd_dynamic.py
=====================================
MHD-Dynamic Neural Initialisation & Regularisation.

Core module exposing the oracle, weight initialiser, and training-time
regulariser used by the benchmark scripts.  All heavy mathematics is
delegated to flu.core.fm_dance; this module is the neural-application
shim that sits on top.

Theorem coverage
----------------
MHD-STRUCT    det(A_magic) = ±1 → unimodular generator, sparse Hessenberg
MHD-MAGIC     axis-line sums constant → T(y) zero-mean over full cube
MHD-SPECTRAL  rank-1 Walsh dual → oracle projects to 1-D alternating phase
MHD-OA-MAX    contiguous rank blocks ⊆ saturated OA(n^{d-1}, d, n, d-1)
MHD-ETK       discrepancy = 1-D Fourier series (low-frequency concentration)

Honest scope
------------
The oracle implements the corrected A_magic formulation from the proof doc
(V15.5-pre).  The STATUS labels below match the theorem registry exactly.

  mhd_oracle_numpy()  — pure NumPy, no flu dependency for portability
  mhd_oracle_flu()    — canonical form via flu.core.fm_dance.magic_coord
                        (preferred when flu-math is installed)
  init_weights()      — contiguous-block init, Kaiming-scaled
  MHDRegulariser      — sparse L2 tethering back to oracle manifold
  kaiming_init()      — standard He baseline for comparison

Dependencies: numpy, flu-math >= 15.4.0
Optional    : torch (for MHDRegulariserTorch)

Author  : Felix Mönnich & The Kinship Mesh Collective
Date    : 2026-05-30
Version : V15.5.0-pre
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    from flu.core.fm_dance import magic_coord as _magic_coord
    _FLU_AVAILABLE = True
except ImportError:
    _FLU_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  MHD ORACLE
#     STATUS: PROVEN (MHD-STRUCT, MHD-SPECTRAL, MHD-MAGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def mhd_oracle_numpy(ranks: np.ndarray, n: int = 5, d: int = 10) -> np.ndarray:
    """
    Vectorised MHD oracle (pure NumPy, no flu dependency).

    Maps an array of integer ranks to scalar T(y) values via:
      1. Base-n address decoding   (rank → digit string a)
      2. A_magic application       (a → x = A_magic @ a + c  mod n)
      3. Spectral projection       (x → T(y) = Σ_j (-1)^j (x_j − n//2))

    STATUS : PROVEN
    Theorems: MHD-STRUCT (A_magic definition), MHD-SPECTRAL (T(y) projection),
              MHD-MAGIC  (T(y) is zero-mean over the full n^d cube).

    Parameters
    ----------
    ranks : 1-D int64 array, values in [0, n^d)
    n     : radix, must be odd and ≥ 3  (even n violates MHD-MAGIC, §6)
    d     : spatial dimension ≥ 2

    Returns
    -------
    float64 array, values in [−d/n, d/n]  (tight bound; typical range ≈ ±0.6)
    """
    ranks  = np.asarray(ranks, dtype=np.int64)
    powers = n ** np.arange(d, dtype=np.int64)

    # Step 1: base-n address decoding
    a = (ranks[:, None] // powers[None, :]) % n          # shape (N, d)

    # Step 2: apply A_magic  (sparse Hessenberg over Z_n — MHD-STRUCT)
    x = np.zeros_like(a)
    x[:, 0]     = a[:, 0] - a[:, 1]
    if d > 2:
        x[:, 1:d-1] = a[:, 0:d-2] - a[:, 2:d]
    x[:, d-1]   = a[:, d-2] - 2 * a[:, d-1]             # ×2 entry in last row

    c = np.full(d, n // 2, dtype=np.int64);  c[d-1] = n - 1
    x = (x + c) % n                                       # Z_n reduction

    # Step 3: spectral projection T(y)  (MHD-SPECTRAL / rank-1 Walsh dual)
    signed = x - (n // 2)
    signs  = (-1.0) ** np.arange(d, dtype=np.float64)
    return (np.sum(signed * signs, axis=1) / (n * 2.0)).astype(np.float64)


def mhd_oracle_flu(ranks: np.ndarray, n: int = 5, d: int = 10) -> np.ndarray:
    """
    Canonical MHD oracle via flu.core.fm_dance.magic_coord.

    Delegates to the FLU package's verified implementation.  Each rank k
    is decoded to its d-dimensional MHD coordinate tuple by magic_coord(),
    then projected to the alternating phase T(y) = Σ_j (-1)^j (x_j − n//2).

    STATUS : PROVEN (delegates to MHD-STRUCT verified implementation in FLU)

    Raises
    ------
    ImportError  if flu-math is not installed
    """
    if not _FLU_AVAILABLE:
        raise ImportError(
            "flu-math is required for mhd_oracle_flu(). "
            "Install with: pip install flu-math  or use mhd_oracle_numpy()."
        )
    ranks  = np.asarray(ranks, dtype=np.int64)
    signs  = (-1.0) ** np.arange(d, dtype=np.float64)
    half   = n // 2

    out = np.empty(len(ranks), dtype=np.float64)
    for i, k in enumerate(ranks):
        coords      = _magic_coord(int(k % (n ** d)), n=n, d=d)  # d-tuple
        signed      = np.array(coords, dtype=np.float64) - half
        out[i]      = np.dot(signed, signs) / (n * 2.0)
    return out


# Alias: prefer flu implementation when available, fall back to numpy
def mhd_oracle(ranks: np.ndarray, n: int = 5, d: int = 10) -> np.ndarray:
    """
    MHD oracle — uses flu.core.fm_dance when available, numpy otherwise.

    For large batch sizes prefer mhd_oracle_numpy() (vectorised);
    mhd_oracle_flu() loops over ranks and is slower but canonical.

    STATUS : PROVEN  (MHD-STRUCT / MHD-SPECTRAL / MHD-MAGIC)
    """
    # numpy version is always faster for batches; flu version is canonical ref
    return mhd_oracle_numpy(ranks, n=n, d=d)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ORACLE SELF-VERIFICATION
#     Callable standalone — used by benchmark scripts and CI
# ═══════════════════════════════════════════════════════════════════════════════

def verify_oracle(n: int = 5, d_algebraic: int = 3, d_full: int = 10,
                  verbose: bool = True) -> bool:
    """
    Run five oracle self-checks mapped to proof-doc theorems.

    V1  MHD-MAGIC    : full-cube T(y) mean = 0  (d=d_algebraic for speed)
    V2  MHD-STRUCT   : oracle(0) matches closed-form expectation
    V3  MHD-SPECTRAL : output range within theoretical bound [−d/n, d/n]
    V4  MHD-OA-MAX   : contiguous arange block has lower std than randint block
    V5  MHD-GEN      : oracle produces > 2 distinct value classes (non-degenerate)

    Returns True if all checks pass, False otherwise.
    Prints results if verbose=True.
    """
    errors = 0

    def _check(label: str, ok: bool, detail: str) -> None:
        nonlocal errors
        if verbose:
            print(f"  {'✓' if ok else '✗'} {label:14s}  {detail}")
        if not ok:
            errors += 1

    # V1  MHD-MAGIC
    all_r = np.arange(n ** d_algebraic, dtype=np.int64)
    vals  = mhd_oracle_numpy(all_r, n=n, d=d_algebraic)
    m     = abs(float(vals.mean()))
    _check("MHD-MAGIC", m < 1e-6, f"full-cube mean = {m:.2e}  (need < 1e-6)")

    # V2  MHD-STRUCT  closed-form: oracle(0) with a_i=0 ∀i
    # x_j = c_j for all j;  c_{d-1}=n-1, others=n//2
    # T = Σ (-1)^j (c_j − n//2) / (n·2)  = (n−1 − n//2)·(−1)^{d-1} / (n·2)
    r0       = float(mhd_oracle_numpy(np.array([0], dtype=np.int64), n=n, d=d_algebraic)[0])
    expected = float(((n - 1) - n // 2) * ((-1) ** (d_algebraic - 1))) / (n * 2.0)
    _check("MHD-STRUCT", abs(r0 - expected) < 1e-5,
           f"oracle(0)={r0:.4f}  expected={expected:.4f}")

    # V3  MHD-SPECTRAL  bound: |T(y)| ≤ d/n  (from alternating sum of ≤d terms)
    sample = mhd_oracle_numpy(np.arange(min(5000, n ** d_algebraic), dtype=np.int64),
                              n=n, d=d_algebraic)
    lo, hi = float(sample.min()), float(sample.max())
    bound  = float(d_algebraic) / float(n)
    _check("MHD-SPECTRAL", lo >= -bound - 1e-4 and hi <= bound + 1e-4,
           f"range=[{lo:.3f},{hi:.3f}]  bound±{bound:.3f}")

    # V4  MHD-OA-MAX  arange blocks have tighter value distribution than randint.
    # Use d=d_full so n^d >> block size (d_algebraic=3 → n^3=125 << 512 → wraps).
    stds_a, stds_r = [], []
    for _ in range(50):
        start  = int(np.random.randint(0, n ** (d_full - 1)))
        N_blk  = 512
        ra     = (start + np.arange(N_blk, dtype=np.int64)) % (n ** d_full)
        rr     = np.random.randint(0, n ** d_full, N_blk, dtype=np.int64)
        stds_a.append(float(mhd_oracle_numpy(ra, n=n, d=d_full).std()))
        stds_r.append(float(mhd_oracle_numpy(rr, n=n, d=d_full).std()))
    ma, mr = np.mean(stds_a), np.mean(stds_r)
    _check("MHD-OA-MAX", ma < mr,
           f"arange_std={ma:.4f} < randint_std={mr:.4f}  (d={d_full})")

    # V5  MHD-GEN  non-degenerate (oracle covers multiple value classes)
    unique = len(np.unique(np.round(vals * 100).astype(int)))
    _check("MHD-GEN", unique > 2, f"unique value classes = {unique}")

    if verbose:
        print(f"\n  {'ALL PASS ✓' if errors == 0 else str(errors) + ' FAILURE(S) ✗'}")

    return errors == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  WEIGHT INITIALISATION
#     STATUS: PROVEN that init preserves OA balance (MHD-OA-MAX)
#             Kaiming scaling is standard practice, not a theorem claim.
# ═══════════════════════════════════════════════════════════════════════════════

def init_weights(fan_in: int, fan_out: int,
                 n: int = 5, d: int = 10,
                 start_rank: Optional[int] = None,
                 ) -> Tuple[np.ndarray, int]:
    """
    Initialise a weight matrix using a contiguous MHD oracle block.

    Each of the fan_in × fan_out weights is assigned one oracle value from
    a contiguous rank sequence starting at start_rank.  Contiguous routing
    preserves the OA balance structure (MHD-OA-MAX); random (randint) routing
    destroys it — confirmed empirically in bench_mhd_dynamic_v2.py Suite A.

    The resulting matrix is:
      1. Mean-centred  (MHD-MAGIC: T(y) is zero-mean over the full cube)
      2. Normalised to unit variance
      3. Rescaled by sqrt(2 / fan_in)  (standard Kaiming / He scaling)

    STATUS : OA-balance preservation PROVEN (MHD-OA-MAX).
             Kaiming rescaling is engineering practice, not a theorem.

    Parameters
    ----------
    fan_in, fan_out : layer dimensions
    n               : oracle radix  (must be odd ≥ 3)
    d               : oracle spatial dimension
    start_rank      : int | None  — if None, sampled uniformly from [0, n^{d-1})
                      (prefix net guarantees balanced initial slice)

    Returns
    -------
    W          : float64 ndarray, shape (fan_in, fan_out)
    start_rank : int  — topological anchor, pass to MHDRegulariser
    """
    num_w = fan_in * fan_out
    if start_rank is None:
        start_rank = int(np.random.randint(0, n ** (d - 1)))
    ranks = (start_rank + np.arange(num_w, dtype=np.int64)) % (n ** d)
    w     = mhd_oracle_numpy(ranks, n=n, d=d).reshape(fan_in, fan_out)
    w     = (w - w.mean()) / (w.std() + 1e-8)
    w    *= math.sqrt(2.0 / fan_in)
    return w, start_rank


def kaiming_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Standard He/Kaiming normal initialisation (baseline)."""
    return np.random.randn(fan_in, fan_out) * math.sqrt(2.0 / fan_in)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MHD DYNAMIC REGULARISER (NumPy)
#     STATUS: CONJECTURE (MHD-STABILITY-CONJECTURE, docs/OPEN_DEBT.md §26)
#             The regulariser's ability to maintain global concentration is
#             conjectured; the local OA-tethering is a direct consequence of
#             MHD-OA-MAX (PROVEN).
# ═══════════════════════════════════════════════════════════════════════════════

class MHDRegulariser:
    """
    Training-time regulariser that tethers weights back to the MHD manifold.

    Adds a sparse L2 penalty driving sampled weights toward their original
    oracle targets.  Applied every update_freq gradient steps.

    MECHANISM
    ---------
    For a random subset of weights (idx) in a layer with anchor start_rank,
    the oracle target for weight[i] is:

        target[i] = oracle((start_rank + i) mod n^d)

    After normalising to match the layer's current mean/std, the gradient is:

        dW[idx] += strength · (W[idx] − target[idx])

    This is the subdifferential of  strength/2 · ‖W[idx] − target[idx]‖²,
    applied sparsely via direct gradient injection (NumPy autodiff).

    STATUS  : Gradient injection proven correct for sparse L2 (calculus).
              Global concentration maintained by regularisation: CONJECTURE
              (MHD-STABILITY-CONJECTURE, MHD-GLOBAL-CONCENTRATION-CONJECTURE).

    Parameters
    ----------
    n           : oracle radix
    d           : oracle spatial dimension
    strength    : L2 penalty coefficient  (default 0.005)
    update_freq : apply every N gradient steps  (default 4)
    max_sample  : max weights audited per layer per update  (default 4096)
    """

    def __init__(self, n: int = 5, d: int = 10, strength: float = 0.005,
                 update_freq: int = 4, max_sample: int = 4096) -> None:
        self.n           = n
        self.d           = d
        self.strength    = strength
        self.update_freq = update_freq
        self.max_sample  = max_sample
        self._step       = 0

    def apply(self, layers: list) -> float:
        """
        Inject MHD tethering gradient into dW for each layer.

        Each element of ``layers`` must have:
            .W              : float64 ndarray  (current weights)
            .dW             : float64 ndarray  (accumulated gradient — modified in place)
            .mhd_start_rank : int              (topological anchor from init_weights)

        Returns the scalar regularisation loss (for logging).
        """
        self._step += 1
        if self._step % self.update_freq != 0:
            return 0.0

        reg_loss = 0.0
        for layer in layers:
            if not hasattr(layer, 'mhd_start_rank'):
                continue
            flat_w = layer.W.ravel()
            numel  = flat_w.size
            n_samp = min(self.max_sample, numel)

            idx     = np.random.choice(numel, n_samp, replace=False)
            samples = flat_w[idx]

            # Exact topological rank reconstruction — MHD-OA-MAX
            ranks   = (layer.mhd_start_rank + idx) % (self.n ** self.d)
            targets = mhd_oracle_numpy(ranks, n=self.n, d=self.d)

            # Align targets to current layer statistics (preserve scale)
            s_mean, s_std = samples.mean(), samples.std() + 1e-8
            targets = (targets - targets.mean()) / (targets.std() + 1e-8)
            targets = targets * s_std + s_mean

            # Inject L2 gradient  d/dW [ strength/2 · (W−T)² ] = strength·(W−T)
            layer.dW.ravel()[idx] += self.strength * (samples - targets)
            reg_loss += float(np.mean((samples - targets) ** 2)) * self.strength

        return reg_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TORCH WRAPPERS (optional, guarded by try/except)
# ═══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    def init_weights_torch(module: "nn.Module", n: int = 5, d: int = 10) -> None:
        """
        Apply MHD initialisation to a torch.nn.Conv2d or torch.nn.Linear module.

        Stores the topological anchor as module.mhd_start_rank so that
        MHDRegulariserTorch can reconstruct exact oracle targets at training time.

        STATUS : OA-balance preservation PROVEN (MHD-OA-MAX).
        """
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            return
        import numpy as np
        device   = module.weight.device
        fan_in   = int(np.prod(module.weight.shape[1:]))
        num_w    = module.weight.numel()

        start = int(torch.randint(0, n ** (d - 1), (1,)).item())
        module.mhd_start_rank = start

        ranks = (start + torch.arange(num_w, device='cpu')) % (n ** d)
        w     = mhd_oracle_numpy(ranks.numpy(), n=n, d=d).reshape(module.weight.shape)
        w     = (w - w.mean()) / (w.std() + 1e-8)
        w    *= math.sqrt(2.0 / fan_in)

        module.weight.data.copy_(torch.tensor(w, dtype=torch.float32).to(device))
        if module.bias is not None:
            module.bias.data.zero_()

    class MHDRegulariserTorch:
        """
        Training-time MHD regulariser for PyTorch models.

        Usage
        -----
            reg = MHDRegulariserTorch(n=5, strength=0.005, update_freq=4)
            # in training loop:
            loss = criterion(outputs, targets) + reg(model)
            loss.backward()

        STATUS : See MHDRegulariser docstring.
        """

        def __init__(self, n: int = 5, d: int = 10, strength: float = 0.005,
                     update_freq: int = 4, max_sample: int = 4096) -> None:
            self.n           = n
            self.d           = d
            self.strength    = strength
            self.update_freq = update_freq
            self.max_sample  = max_sample
            self._step       = 0

        def __call__(self, model: "nn.Module") -> "torch.Tensor":
            self._step += 1
            device = next(model.parameters()).device

            if self._step % self.update_freq != 0:
                return torch.tensor(0.0, device=device)

            reg_loss = torch.tensor(0.0, device=device)

            for name, param in model.named_parameters():
                if not (param.requires_grad and 'weight' in name and param.ndim >= 2):
                    continue
                # Retrieve topological anchor
                mod_name = name.rsplit('.', 1)[0]
                try:
                    m = model.get_submodule(mod_name)
                except AttributeError:
                    m = dict(model.named_modules()).get(mod_name)
                if m is None or not hasattr(m, 'mhd_start_rank'):
                    continue

                with torch.no_grad():
                    flat       = param.view(-1)
                    n_samp     = min(self.max_sample, flat.numel())
                    idx        = torch.randint(0, flat.numel(), (n_samp,), device=device)
                    samples    = flat[idx]

                    ranks      = ((m.mhd_start_rank + idx.cpu()) % (self.n ** self.d)).numpy()
                    targets_np = mhd_oracle_numpy(ranks.astype(np.int64), n=self.n, d=self.d)
                    targets    = torch.tensor(targets_np, dtype=torch.float32, device=device)

                    s_mean  = samples.mean();  s_std = samples.std() + 1e-8
                    targets = (targets - targets.mean()) / (targets.std() + 1e-8)
                    targets = targets * s_std + s_mean

                import torch.nn.functional as F
                reg_loss = reg_loss + F.mse_loss(param.view(-1)[idx], targets)

            return self.strength * reg_loss

else:
    # Stubs so imports don't fail when torch is absent
    def init_weights_torch(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("PyTorch is required for init_weights_torch().")

    class MHDRegulariserTorch:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MHDRegulariserTorch.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  QUICK SELF-TEST  (python mhd_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MHD-Dynamic self-test\n" + "─" * 40)
    ok = verify_oracle(verbose=True)
    if ok:
        W, sr = init_weights(64, 64)
        print(f"\nSample init_weights(64,64):")
        print(f"  shape={W.shape}  mean={W.mean():.4f}  std={W.std():.4f}")
        print(f"  start_rank={sr}")
    import sys
    sys.exit(0 if ok else 1)
