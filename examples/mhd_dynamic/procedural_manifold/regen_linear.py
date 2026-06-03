"""RegenLinear - dense training layer (torch required)."""
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
import torch, torch.nn as nn
from .oracle import MHDOracle

class RegenLinear(nn.Module):
    def __init__(self, in_f, out_f, oracle, start_rank=0, contract=None):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.oracle = oracle
        self.start_rank = start_rank
        self.contract = contract
        self.deltas = nn.Parameter(torch.zeros(in_f, out_f))

    def get_fused_weight(self):
        base = self.oracle.generate_block_pytorch(self.in_f, self.out_f, self.start_rank)
        return base.to(self.deltas.device) + self.deltas

    def forward(self, x): return x @ self.get_fused_weight()
    def delta_count(self): return (self.deltas.abs() > 1e-7).sum().item()

    def center_deltas(self, mode: str = 'masked'):
        """
        V4.2 Manifold Projection — enforce zero-mean on deltas.

        mode='masked' (default, recommended):
            Shift only non-zero entries. Zeros are preserved exactly.
            REQUIRED when used alongside L1/proximal sparsity: full centering
            reawakens pruned weights by shifting zeros to -mean(Δ).
            Verified by bench_mhd_twins.py Phase 3 centering comparison.

        mode='full':
            Shift ALL entries by the global mean. Use only when no sparsity
            is expected (Phase 1/2 of the twin experiment, or unfrozen training).
        """
        if self.deltas is None or self.deltas.numel() == 0:
            return
        if mode == 'masked':
            mask = self.deltas.data.abs() > 1e-12
            if mask.any():
                self.deltas.data[mask] -= self.deltas.data[mask].mean()
        else:
            self.deltas.data -= self.deltas.data.mean()
