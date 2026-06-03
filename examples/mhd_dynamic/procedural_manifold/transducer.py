import torch, torch.nn as nn
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .oracle import MHDOracle

class TernaryTransducer(nn.Module):
    def __init__(self, dim=128, window_length=9, symbol_map=None, oracle_n=5, oracle_d=10):
        super().__init__()
        self.dim = dim
        self.window_length = window_length
        self.oracle = MHDOracle(n=oracle_n, d=oracle_d)
        self.symbol_map = symbol_map or {}
        self.projection = nn.Linear(32, dim)

    def set_symbol_map(self, mapping): self.symbol_map = mapping

    def _to_ternary(self, symbols):
        seq = [self.symbol_map.get(s,1) for s in symbols]
        if len(seq) < self.window_length: seq += [1]*(self.window_length - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def _ternary_to_mhd(self, seq):
        coords = []
        for i in range(len(seq)-self.window_length+1):
            a = seq[i:i+self.window_length]
            d = len(a)
            x = torch.zeros(d)
            x[0] = a[0]-a[1]
            if d>2: x[1:d-1] = a[0:d-2]-a[2:d]
            x[d-1] = a[d-2]-2*a[d-1]
            c = torch.full((d,), self.oracle.n//2, dtype=torch.float32); c[d-1]=self.oracle.n-1
            x = (x+c)%self.oracle.n
            coords.append(x)
        return coords

    def forward(self, symbols):
        ternary = self._to_ternary(symbols)
        coords = self._ternary_to_mhd(ternary)
        if not coords: return torch.zeros(1, self.dim)
        coords = torch.stack(coords)
        mean_coord = coords.mean(0)
        t_y = torch.sum(torch.tensor([(-1)**j for j in range(len(mean_coord))])*mean_coord)
        feat = torch.cat([mean_coord, t_y.unsqueeze(0), torch.zeros(32-len(mean_coord)-1)])
        return self.projection(feat.unsqueeze(0))
