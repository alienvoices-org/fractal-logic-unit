import torch, torch.nn as nn, math
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .streaming_regen import StreamingRegenLinear
from .oracle import MHDOracle

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, oracle=None):
        super().__init__()
        assert dim%num_heads==0
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim//num_heads
        self.oracle = oracle or MHDOracle()
        self.W_q = StreamingRegenLinear(dim, dim, self.oracle)
        self.W_k = StreamingRegenLinear(dim, dim, self.oracle)
        self.W_v = StreamingRegenLinear(dim, dim, self.oracle)
        self.W_o = StreamingRegenLinear(dim, dim, self.oracle)

    def forward(self, x):
        B,T,C = x.shape
        q = self.W_q(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = self.W_k(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = self.W_v(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        attn = (q @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        return self.W_o(out.transpose(1,2).contiguous().view(B,T,C))
