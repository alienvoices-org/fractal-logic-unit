import json
try:
    import yaml
except ImportError:
    yaml = None
from ._compat import HAS_TORCH
if not HAS_TORCH: raise ImportError("torch required")
from .transformer import RegenTransformer
from .transducer import TernaryTransducer
from .model import KinshipMoE

class KinshipBuilder:
    @staticmethod
    def load_mapping(path):
        with open(path) as f:
            if path.endswith(('.yaml','.yml')):
                if yaml is None: raise ImportError("PyYAML required for YAML configs")
                return yaml.safe_load(f)
            return json.load(f)

    @staticmethod
    def build_transformer(config):
        return RegenTransformer(
            num_blocks=config.get('num_blocks', 64),
            dim=config.get('dim', 128),
            num_heads=config.get('num_heads', 8),
            base_scale=config.get('base_scale', config.get('res_factor', 0.012)),
            activation=config.get('activation', 'leaky_relu'),
        )

    @staticmethod
    def build_transducer(config):
        mapping = {}
        if 'transducer_mapping' in config:
            mapping = KinshipBuilder.load_mapping(config['transducer_mapping'])
        return TernaryTransducer(
            dim=config.get('dim', 128),
            window_length=config.get('window_length', 9),
            symbol_map=mapping,
            oracle_n=config.get('oracle_n', 5),
            oracle_d=config.get('oracle_d', 10),
        )

    @staticmethod
    def build_moe(config):
        t = KinshipBuilder.build_transformer(config)
        d = KinshipBuilder.build_transducer(config) if config.get('use_transducer', True) else None
        return KinshipMoE(t, d, config.get('num_classes', 10))
