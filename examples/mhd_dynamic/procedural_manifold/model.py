import torch, torch.nn as nn, torch.nn.functional as F
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .transformer import RegenTransformer
from .transducer import TernaryTransducer

class KinshipMoE(nn.Module):
    def __init__(self, transformer, transducer=None, num_classes=10):
        super().__init__()
        self.transformer = transformer
        self.transducer = transducer
        self._dim = transformer.blocks[0].attn.dim
        self.gate = nn.Linear(self._dim, 2 if transducer else 1)
        self.head = nn.Linear(self._dim, num_classes)

    @property
    def dim(self): return self._dim

    def forward(self, x, symbols=None):
        l = self.transformer(x)
        if self.transducer is not None and symbols is not None:
            m = self.transducer(symbols)
            gate = F.softmax(self.gate(l), dim=-1)
            g0, g1 = gate[...,0:1], gate[...,1:2]
            B, T = l.size(0), l.size(1)
            if m.dim()==2:
                if m.size(0)==1 and B>1: m = m.expand(B,-1)
                m = m.unsqueeze(1).expand(-1,T,-1)
            elif m.size(1)==1: m = m.expand(-1,T,-1)
            fused = g0*l + g1*m
        else: fused = l
        return self.head(fused)
