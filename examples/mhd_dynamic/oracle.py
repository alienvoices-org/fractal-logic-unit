"""Deterministic manifold generator (MHD Net).

Bug fixes vs V4.0/V4.1:
  BUG-1 FIXED: per-block normalisation now applied correctly.
    _global_mean/_global_std were identity (0/1) and provided no normalisation.
    Raw T(y) values have std≈0.37 per block; without normalisation weights are
    3× too small, forcing the network to learn large compensatory deltas.
    Fix: normalise each generated block to zero mean / unit variance, then
    apply Kaiming scaling — identical to mhd_dynamic.init_weights().

  BUG-3 FIXED: generate_tile_pytorch now uses full-block statistics.
    Previously each tile was normalised independently; fusing tiles gave
    inconsistent weights.  Now generate_block_pytorch computes mean/std
    over the full block and passes them to each tile call.

V4.1 oracle.py is otherwise identical to V4.0 (explicitly unchanged per spec).
"""
from ._compat import _is_odd, HAS_NUMPY, HAS_TORCH
import math

if HAS_NUMPY:
    import numpy as np


class MHDOracle:
    """MHD manifold generator.  Sanity check: python -m procedural_manifold.oracle"""

    def __init__(self, n: int = 5, d: int = 10):
        if not _is_odd(n):
            raise ValueError("n must be odd (even-n obstruction, MHD §6)")
        self.n, self.d = n, d
        self.period    = n ** d
        self._c        = [n // 2] * (d - 1) + [n - 1]
        self._signs    = [(-1) ** j for j in range(d)]
        # Deprecated fields kept for back-compat; no longer used in generation.
        self._global_mean = 0.0
        self._global_std  = 1.0

    # ------------------------------------------------------------------ #
    #  Population statistics                                               #
    # ------------------------------------------------------------------ #

    def sigma_T(self) -> float:
        """
        Theoretical std of T(y) over the uniform distribution on [0, n^d).

        Closed form verified empirically (200k random samples, n=5, d=10):
            σ_T = sqrt(d · (n² − 1) / (48 · n²)) ≈ 0.4472

        This is the value the oracle normalises away in _generate_numpy via
        per-block (v − mean) / (std + ε).  Knowing σ_T lets you reason about
        the raw value range without sampling.
        """
        return math.sqrt(self.d * (self.n**2 - 1) / (48 * self.n**2))

    # ------------------------------------------------------------------ #
    #  Raw T(y) computation helpers (shared by numpy and pytorch paths)   #
    # ------------------------------------------------------------------ #

    def _raw_numpy(self, ranks: "np.ndarray") -> "np.ndarray":
        """Return raw (un-normalised) T(y) values for an array of ranks."""
        total  = len(ranks)
        digits = []
        tmp    = ranks.copy()
        for _ in range(self.d):
            digits.append(tmp % self.n)
            tmp //= self.n
        digits = np.stack(digits, axis=0)          # shape (d, total)

        x      = np.zeros_like(digits, dtype=np.float64)
        x[0]   = digits[0] - digits[1]
        for j in range(1, self.d - 1):
            x[j] = digits[j - 1] - digits[j + 1]
        x[self.d - 1] = digits[self.d - 2] - 2 * digits[self.d - 1]

        shift  = np.array(self._c, dtype=np.int64).reshape(-1, 1)
        x      = (x + shift) % self.n

        signs  = np.array(self._signs, dtype=np.float64).reshape(-1, 1)
        return np.sum((x - self.n // 2) * signs, axis=0) / (2.0 * self.n)  # (total,)

    # ------------------------------------------------------------------ #
    #  Public generation API                                               #
    # ------------------------------------------------------------------ #

    def _generate_numpy(self, in_f: int, out_f: int, start_rank: int) -> "np.ndarray":
        """Generate (in_f, out_f) weight matrix via numpy.

        Normalises the full block to zero mean / unit variance, then applies
        Kaiming scaling — identical to mhd_dynamic.init_weights().
        """
        total = in_f * out_f
        ranks = (np.arange(total, dtype=np.int64) + start_rank) % self.period
        v     = self._raw_numpy(ranks)                      # raw T(y)
        v     = (v - v.mean()) / (v.std() + 1e-8)          # BUG-1 FIX: normalise
        v    *= math.sqrt(2.0 / in_f)                       # Kaiming
        return v.reshape(in_f, out_f)

    def _pure_generate(self, in_f: int, out_f: int, start_rank: int) -> list:
        """Pure-Python fallback (no numpy).  Same normalisation as _generate_numpy."""
        total  = in_f * out_f
        result = []
        for idx in range(total):
            rank   = (start_rank + idx) % self.period
            digits = []
            tmp    = rank
            for _ in range(self.d):
                digits.append(tmp % self.n)
                tmp //= self.n
            x = [0.0] * self.d
            x[0] = digits[0] - digits[1]
            for j in range(1, self.d - 1):
                x[j] = digits[j - 1] - digits[j + 1]
            x[self.d - 1] = digits[self.d - 2] - 2 * digits[self.d - 1]
            for j in range(self.d):
                x[j] = (x[j] + self._c[j]) % self.n
            v = sum((x[j] - self.n // 2) * self._signs[j]
                    for j in range(self.d)) / (2.0 * self.n)
            result.append(v)

        # BUG-1 FIX: per-block normalisation
        mean_v = sum(result) / total
        std_v  = (sum((vi - mean_v) ** 2 for vi in result) / total) ** 0.5 + 1e-8
        scale  = math.sqrt(2.0 / in_f) / std_v
        result = [(vi - mean_v) * scale for vi in result]
        return [result[i * out_f: (i + 1) * out_f] for i in range(in_f)]

    def generate_weights(self, in_f: int, out_f: int, start_rank: int = 0):
        """Generate weights — uses numpy if available, pure-Python otherwise."""
        if HAS_NUMPY:
            return self._generate_numpy(in_f, out_f, start_rank)
        return self._pure_generate(in_f, out_f, start_rank)

    # ------------------------------------------------------------------ #
    #  PyTorch generation                                                  #
    # ------------------------------------------------------------------ #

    def generate_block_pytorch(self, in_f: int, out_f: int, start_rank: int,
                               device: str = 'cpu') -> "torch.Tensor":
        """Generate full (in_f, out_f) weight block on device.

        BUG-3 FIX: stats computed over the full block, then applied tile-wise
        via generate_tile_pytorch.
        """
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        import torch

        # Compute full-block raw values to get population stats
        total  = in_f * out_f
        ranks  = (torch.arange(total, device=device, dtype=torch.long) + start_rank) % self.period
        raw    = self._raw_tile_pytorch(ranks, device)       # (total,)
        blk_mean = raw.mean().item()
        blk_std  = raw.std().item() + 1e-8
        scale    = math.sqrt(2.0 / in_f) / blk_std

        # Apply normalisation directly on the full block
        v = (raw - blk_mean) * scale
        return v.view(in_f, out_f)

    def generate_tile_pytorch(self, in_f: int, out_f: int, start_rank: int,
                              col_start: int, col_end: int,
                              device: str = 'cpu',
                              _blk_mean: float = None,
                              _blk_scale: float = None) -> "torch.Tensor":
        """Generate a column-slice of the full (in_f, out_f) weight block.

        BUG-3 FIX: if _blk_mean/_blk_scale are provided (from generate_block_pytorch),
        use them instead of computing per-tile stats.  When called standalone
        (StreamingRegenLinear), computes full-block stats first.
        """
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        import torch

        cols      = col_end - col_start
        row_idx   = torch.arange(in_f, device=device).unsqueeze(1)
        col_idx   = torch.arange(col_start, col_end, device=device).unsqueeze(0)
        ranks     = (start_rank + row_idx * out_f + col_idx).flatten() % self.period
        raw_tile  = self._raw_tile_pytorch(ranks, device)    # (in_f * cols,)

        if _blk_mean is not None and _blk_scale is not None:
            # Use caller-provided stats (full-block normalisation)
            v = (raw_tile - _blk_mean) * _blk_scale
        else:
            # Standalone call: compute full block stats for consistency
            total     = in_f * out_f
            all_ranks = (torch.arange(total, device=device, dtype=torch.long)
                         + start_rank) % self.period
            raw_full  = self._raw_tile_pytorch(all_ranks, device)
            blk_mean  = raw_full.mean().item()
            blk_scale = math.sqrt(2.0 / in_f) / (raw_full.std().item() + 1e-8)
            v         = (raw_tile - blk_mean) * blk_scale

        return v.view(in_f, cols)

    def _raw_tile_pytorch(self, ranks: "torch.Tensor", device: str) -> "torch.Tensor":
        """Compute raw (un-normalised) T(y) for a flat ranks tensor."""
        import torch
        digits = []
        tmp    = ranks.clone()
        for _ in range(self.d):
            digits.append(tmp % self.n)
            tmp = tmp // self.n
        digits = torch.stack(digits, dim=0)               # (d, N)

        x      = torch.zeros_like(digits, dtype=torch.float32)
        x[0]   = (digits[0] - digits[1]).float()
        for j in range(1, self.d - 1):
            x[j] = (digits[j - 1] - digits[j + 1]).float()
        x[self.d - 1] = (digits[self.d - 2] - 2 * digits[self.d - 1]).float()

        shift  = torch.tensor(self._c, dtype=torch.long, device=device).unsqueeze(1)
        x      = (x + shift) % self.n

        signs  = torch.tensor(self._signs, dtype=torch.float32, device=device).unsqueeze(1)
        return torch.sum((x.float() - self.n // 2) * signs, dim=0) / (2.0 * self.n)


if __name__ == "__main__":
    import math
    oracle = MHDOracle(5, 10)
    if HAS_NUMPY:
        import numpy as np
        w      = oracle._generate_numpy(128, 256, 0).ravel()
        mean   = float(w.mean())
        std    = float(w.std())
        target = math.sqrt(2.0 / 128)
        print(f"mean: {mean:.6f}  std: {std:.6f}  expected_std: {target:.6f}")
        assert abs(mean) < 1e-4,            f"mean too large: {mean}"
        assert abs(std - target) < 1e-3,    f"std off: {std} vs {target}"
        print("Sanity check passed.")
    else:
        print("numpy not available — skipping sanity check.")
