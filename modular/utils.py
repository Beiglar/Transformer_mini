import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from matplotlib import pyplot as plt
from typing import NamedTuple, TypeAlias, Union
from functools import partial

from modular.model import TinyTransformerLM
from modular.dataloader import MemmapDataLoader, jax_np_collet
from modular.tokenizer import MiniCharTok
from modular.training import train_step

Array = jax.Array
PRNGKey: TypeAlias = Union[jaxlib.xla_client.ArrayImpl, jax.random.PRNGKey] # type: ignore


class History(NamedTuple):
    loss: list = []
    LR: list = []
    val_loss: list = []

    def __call__(self, loss: float, lr: float, val_loss: float|None=None):
        self.loss.append(loss)
        self.LR.append(lr)
        
        if val_loss is not None:
            self.val_loss.append(val_loss)
        
    def report(self, last_n_steps=10) -> str:
        loss_mean = np.mean(self.loss[-last_n_steps:])
        perplexity = np.exp(loss_mean)
        list_to_fit = [
            f"Loss: {loss_mean:.4f}",
            f"Perplexity: {perplexity:.4f}",
            f"LR: {np.mean(self.LR[-last_n_steps:]):.8f}"]
        if len(self.val_loss) != 0:
            list_to_fit.append(f"Val Loss: {self.val_loss[-1]:.4f}")
        return " | ".join(list_to_fit)

def find_learning_rate(
    model: nnx.Module,
    dataloader: MemmapDataLoader,
    min_lr: float = 1e-7,
    max_lr: float = 10.0,
    num_steps: int = 100
) -> tuple[list, list]:
    """Learning rate range test to find optimal learning rate"""
    lrs = jnp.logspace(jnp.log10(min_lr), jnp.log10(max_lr), num_steps)
    schedule = lambda count: lrs[count]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0), 
        optax.sgd(learning_rate=schedule)
        )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    losses = []
    
    for i, (batch, lr) in enumerate(zip(dataloader, lrs)):
        if i >= num_steps:
            break

        loss = train_step(model, optimizer, jax_np_collet(batch))
        losses.append(float(loss))

        if i % 10 == 0:
            print(f"LR test step {i}/{num_steps}, LR: {lr:.2e}, Loss: {loss:.4f}")
        
    return lrs.tolist(), losses

def plot_lr_find(lrs: list, losses: list, save_path: str | None = None):
    """Plot learning rate finder results"""
    plt.figure(figsize=(10, 6), frameon=False)
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if len(losses) > 10:
        gradients = np.gradient(losses)
        best_idx = np.argmin(gradients)
        best_lr = lrs[best_idx]
        plt.axvline(x=best_lr, color='red', linestyle='--', 
                   label=f'Suggested LR: {best_lr:.2e}')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

@partial(nnx.jit, static_argnames=['max_new_tokens', 'top_p', 'temperature'])
def jit_generate(
    model: TinyTransformerLM,
    initial_ids: Array,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    rng: PRNGKey,) -> Array:
    """JIT-compiled generation function with KV caching option."""
    return model.generate(initial_ids, max_new_tokens, top_p, temperature, rng)

def sample_from_model(
    model: TinyTransformerLM,
    initial_text: str,
    char_tokenizer: MiniCharTok,
    rng: PRNGKey,
    max_new_tokens: int = 100,
    top_p: float = 0.8,
    temperature: float = 1.0,
) -> str:
    """
    Generates text from a model given an initial prompt with optional KV caching.
    """
    assert 0 < top_p <= 1, "`top_p` should be in range (0, 1]"
    assert temperature != 0, "`temperature` of zero divides by zero, can't do that around here."
    initial_ids = jnp.array([char_tokenizer.Encode(initial_text)], dtype=jnp.int32)
    model.eval()
    generated_ids = jit_generate(model, initial_ids, max_new_tokens, top_p, temperature, rng)
    return char_tokenizer.Decode(generated_ids[0].tolist())

def write_sample_to_file(file_path: str, text: str) -> None:
    """Appends text to a file, followed by a separator."""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + '\n' + ('-—' * 40) + '\n')

def format_report(epoch: int, step: int, history_report: str, sampled_text: str) -> str:
    """Formats a string for logging training progress."""
    header = f"Epoch {epoch:2}, Step {step} " + history_report
    return f"{header}\nSampled: {sampled_text}"
