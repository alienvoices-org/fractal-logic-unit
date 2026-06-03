import torch, torch.nn as nn
from ._compat import HAS_TORCH, HAS_PL
if not HAS_TORCH: raise ImportError("torch required")
from .builder import KinshipBuilder

class LightningWrapper:
    def __init__(self, config, lr=1e-3):
        if not HAS_PL: raise ImportError("pytorch_lightning required")
        import pytorch_lightning as pl
        self.model = KinshipBuilder.build_moe(config); self.lr = lr

class HuggingFaceWrapper(nn.Module):
    """Wraps KinshipMoE for HuggingFace pipelines.  Expects input embeddings tensor."""
    def __init__(self, config):
        super().__init__()
        self.model = KinshipBuilder.build_moe(config)

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        # inputs_embeds is already a tensor of shape (batch, seq, dim)
        return self.model(inputs_embeds)

def export_to_onnx(model, filepath, sample_input):
    torch.onnx.export(model, sample_input, filepath, input_names=['input'], output_names=['output'], opset_version=14)
