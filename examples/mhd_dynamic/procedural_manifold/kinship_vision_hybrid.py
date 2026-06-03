import torch, torch.nn as nn
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .attention import MultiHeadAttention
from .oracle import MHDOracle

class KinshipVisionHybrid(nn.Module):
    def __init__(self, num_conv=4, num_trans=64, dim=128, num_heads=8, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Linear(3*4*4, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1,64,dim)*0.02)
        self.conv_blocks = nn.ModuleList([self._conv(dim) for _ in range(num_conv)])
        self.trans_blocks = nn.ModuleList([MultiHeadAttention(dim, num_heads) for _ in range(num_trans)])
        self.head = nn.Linear(dim, num_classes)

    @staticmethod
    def _conv(dim): return nn.Sequential(nn.Conv1d(dim,dim,3,padding=1), nn.ReLU(), nn.Conv1d(dim,dim,3,padding=1), nn.ReLU())

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2,4,4).unfold(3,4,4).reshape(B,64,-1)
        x = self.patch_embed(x) + self.pos_encoding
        for b in self.conv_blocks: x = b(x.transpose(1,2)).transpose(1,2) + x
        for b in self.trans_blocks: x = b(x) + x*0.012
        return self.head(x.mean(1))
