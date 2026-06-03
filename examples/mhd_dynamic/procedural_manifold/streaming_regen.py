"""StreamingRegenLinear - sparse execution with COO deltas."""
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
import torch, torch.nn as nn
from .oracle import MHDOracle
from collections import defaultdict

class StreamingRegenLinear(nn.Module):
    def __init__(self, in_f, out_f, oracle, start_rank=0, tile_size=256):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.oracle = oracle
        self.start_rank = start_rank
        self.tile_size = tile_size
        self.delta_rows, self.delta_cols, self.delta_vals = [], [], []
        self._buckets = None
        self._delta_dict = {}

    def add_delta(self, in_idx, out_idx, value):
        key = (in_idx, out_idx)
        if key in self._delta_dict: self._delta_dict[key] += value
        else: self._delta_dict[key] = value
        self._buckets = None
        self._rebuild_coo()

    def set_deltas(self, rows, cols, vals):
        self._delta_dict = {}
        for r,c,v in zip(rows, cols, vals):
            key = (r,c); self._delta_dict[key] = self._delta_dict.get(key,0.0)+v
        self._buckets = None
        self._rebuild_coo()

    def _rebuild_coo(self):
        self.delta_rows, self.delta_cols, self.delta_vals = [], [], []
        for (r,c),v in self._delta_dict.items():
            self.delta_rows.append(r); self.delta_cols.append(c); self.delta_vals.append(v)

    def _build_buckets(self):
        buckets = defaultdict(list)
        for in_idx, out_idx, val in zip(self.delta_rows, self.delta_cols, self.delta_vals):
            buckets[out_idx//self.tile_size].append((in_idx, out_idx, val))
        self._buckets = dict(buckets)

    def forward(self, x):
        *batch_dims, in_dim = x.shape
        assert in_dim == self.in_f
        x_flat = x.reshape(-1, self.in_f)
        M = x_flat.shape[0]
        out = torch.zeros(M, self.out_f, device=x.device, dtype=x.dtype)
        if self._buckets is None: self._build_buckets()
        for col_start in range(0, self.out_f, self.tile_size):
            col_end = min(col_start+self.tile_size, self.out_f)
            tile_id = col_start//self.tile_size
            W_tile = self.oracle.generate_tile_pytorch(self.in_f, self.out_f, self.start_rank,
                                                       col_start, col_end, device=x.device)
            for in_idx, out_idx, val in self._buckets.get(tile_id, []):
                W_tile[in_idx, out_idx-col_start] += val
            out[:, col_start:col_end] = x_flat @ W_tile
        return out.reshape(*batch_dims, self.out_f)

    def delta_count(self): return len(self._delta_dict)

    def center_deltas(self, mode: str = 'masked'):
        """
        V4.2 Manifold Projection — enforce zero-mean on sparse deltas.

        mode='masked' (default, recommended):
            Centres only the stored (non-zero) entries. Because this COO
            dictionary stores only non-zero deltas by construction, masked
            and full centering are equivalent here — but 'masked' is the
            correct semantic to signal intent and maintain API parity with
            RegenLinear, where the distinction matters.

        mode='full':
            Identical behaviour for COO dict. Kept for API symmetry.

        Invalidates tile cache after centering.
        """
        if not self._delta_dict:
            return
        mean_val = sum(self._delta_dict.values()) / len(self._delta_dict)
        for k in self._delta_dict:
            self._delta_dict[k] -= mean_val
        self._buckets = None
        self._rebuild_coo()
