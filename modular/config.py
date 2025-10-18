from dataclasses import dataclass
from flax import nnx

@dataclass
class LRConfig:
    init_value: float = 1e-4
    peak_value: float = 1e-3
    warmup_steps: int = 1000 # A default, should be overridden
    decay_steps: int = 9000 # A default, should be overridden
    end_value: float = 1e-5
    exponent: float = 1.0

@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.5
    context_size: int | None = None
    rngs: nnx.Rngs = nnx.Rngs(0)

@dataclass
class DataSplitRatios:
    train: float = 0.9
    valid: float = 0.1
    test : float | None = None

    def __call__(self, dataset_length: int):
        train_end = int(self.train * dataset_length)
        valid_end = train_end + int(self.valid * dataset_length)
        return train_end, valid_end
