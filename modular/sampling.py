import jax
import jax.numpy as jnp
import jaxlib
from functools import partial
from typing import TypeAlias, Union

Array = jax.Array
PRNGKey: TypeAlias = Union[jaxlib.xla_client.ArrayImpl, jax.random.PRNGKey] # type: ignore


def top_p_filtering_probs(probs: jnp.ndarray, top_p: float):
    """
    probs: (..., V) array of probabilities
    top_p: float in (0, 1]
    returns: (sorted_indices, filtered_probs)
      sorted_indices: (..., V) indices of tokens sorted by prob descending
      filtered_probs: (..., V) probabilities after top-p filtering and renormalization
    """
    # probs: (..., V)
    sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)
    # keep tokens up to and including the first where cumulative >= top_p
    cutoff_mask = cumulative <= top_p
    # always keep at least the top token
    cutoff_mask = cutoff_mask.at[..., 0].set(True)
    # zero-out rest and renormalize
    filtered = jnp.where(cutoff_mask, sorted_probs, 0.0)
    filtered = filtered / (jnp.sum(filtered, axis=-1, keepdims=True) + 1e-12)
    return sorted_indices, filtered

@partial(jax.jit, static_argnames=['top_p', 'temperature'])
def top_p_sample(key: PRNGKey, logits: jax.Array, top_p: float = 0.9, temperature: float = 1.0):
    """ 
    Sample from the top-p distribution.
        key: PRNGKey for randomness
        logits: (B, V) unnormalized log-probabilities
        top_p: float in (0, 1]
        temperature: float > 0
        returns: (B,) sampled token IDs
    """
    logits = logits / temperature
    probs = jax.nn.softmax(logits, axis=-1)
    sorted_indices, filtered = top_p_filtering_probs(probs, top_p)
    sample_idx = jax.random.categorical(key, jnp.log(filtered))
    sampled_token = jnp.take_along_axis(sorted_indices, sample_idx[..., None], axis=-1)
    return jnp.squeeze(sampled_token, axis=-1)