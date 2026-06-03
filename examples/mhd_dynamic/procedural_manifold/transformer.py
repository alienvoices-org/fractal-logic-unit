import torch, torch.nn as nn, math
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .attention import MultiHeadAttention
from .streaming_regen import StreamingRegenLinear
from .oracle import MHDOracle

class RegenTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, base_scale=0.012, block_index=1,
                 activation='leaky_relu', oracle=None):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, oracle)
        self.ffn  = StreamingRegenLinear(dim, dim, oracle or MHDOracle())
        # V4.1: Stochastic Residual Scaling — later blocks contribute softer nudges
        self.res_factor = base_scale / math.sqrt(block_index)
        # V4.1: Activation choice
        self.act = torch.nn.functional.leaky_relu if activation == 'leaky_relu' else torch.relu

    def forward(self, x):
        x = x + self.res_factor * self.attn(x)
        return x + self.res_factor * self.act(self.ffn(x))

class RegenTransformer(nn.Module):
    def __init__(self, num_blocks, dim, num_heads=8, base_scale=0.012,
                 activation='leaky_relu'):
        super().__init__()
        self.blocks = nn.ModuleList([
            RegenTransformerBlock(dim, num_heads, base_scale, i+1, activation)
            for i in range(num_blocks)
        ])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x
