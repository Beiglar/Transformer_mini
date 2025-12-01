from typing import TypeAlias, Union
import jax
import numpy as np
import optax
from flax import nnx

from modular.model import TinyTransformerLM
from modular.dataloader import MemmapDataLoader, jax_np_collet
from modular.config import ModelConfig, LRConfig

Array: TypeAlias = Union[jax.Array, np.ndarray]

def create_model_and_optimizer(
        model_config: ModelConfig,
        lr_config: LRConfig,
        grad_clip: float = 1.0,
        rng_seed: int = 0):
    rngs = nnx.Rngs(params=rng_seed, dropout=rng_seed + 1)
    model_kwargs = model_config.__dict__.copy()
    model_kwargs.pop('rngs', None)
    model = TinyTransformerLM(**model_kwargs, rngs=rngs)

    schedule = optax.warmup_cosine_decay_schedule(**lr_config.__dict__)

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule)
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return model, optimizer, schedule

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: tuple[Array, Array]):
    """
    Single training step: computes grads and updates the model and optimizer.
    """
    batch_inputs, batch_targets = batch

    def loss_fn(model):
        logits, _ = model(batch_inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_targets).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def compute_val_loss(model: TinyTransformerLM, dataloader: MemmapDataLoader):
    model.eval()
    @nnx.jit
    def compute_val_loss_(model: TinyTransformerLM, batch: tuple[Array, Array]):
        batch_inputs, batch_targets = batch
        logits, _ = model(batch_inputs) # type: ignore
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_targets).mean()
    
    total_loss = 0
    count = 0
    for val_batch in dataloader:
        total_loss += compute_val_loss_(model, jax_np_collet(val_batch)).item()
        count += 1
    return total_loss / count if count > 0 else 0
