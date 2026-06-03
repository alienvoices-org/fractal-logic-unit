"""Procedural Manifold Parameter System V4.0 - Public API."""
from ._compat import HAS_TORCH, HAS_NUMPY
from .oracle import MHDOracle
from .delta_knowledge import DeltaKnowledge, DeltaComparer, DeltaRegistry, DeltaExtractor

if HAS_TORCH:
    from .regen_linear import RegenLinear
    from .streaming_regen import StreamingRegenLinear
    from .attention import MultiHeadAttention
    from .transformer import RegenTransformerBlock, RegenTransformer
    from .transducer import TernaryTransducer
    from .delta_stream import DeltaStream
    from .inner_communion import SharedDeltaPool
    from .model import KinshipMoE
    from .builder import KinshipBuilder
    from .agnostic_converter import convert_checkpoint
    from .agnostic_regen import patch_model
    from .kinship_vision_hybrid import KinshipVisionHybrid
    from .token_generator import KinshipTokenGenerator
    from .hooks import LightningWrapper, HuggingFaceWrapper, export_to_onnx
    from .benchmarks import (benchmark_stability, benchmark_streaming,
                             benchmark_token_generation, benchmark_delta_knowledge)

__version__ = "4.0.0"
