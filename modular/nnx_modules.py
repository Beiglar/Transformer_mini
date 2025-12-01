import jax
import jax.numpy as jnp
from flax import nnx
import jaxlib
from typing import Optional, Tuple, TypeAlias, Union, Callable

Array = jax.Array
PRNGKey: TypeAlias = Union[jaxlib.xla_client.ArrayImpl, jax.random.PRNGKey] # type: ignore

class RoPE(nnx.Module):
    """Rotary Positional Embedding (RoPE)"""
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = theta ** -(jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
        # store as a Variable so state serialization works nicely
        self.inv_freq = nnx.Variable(inv_freq)

    def get_sin_cos(self, seq_len: int, offset: int = 0) -> Tuple[Array, Array]:
        """Get sin/cos with optional position offset for caching"""
        t = jnp.arange(offset, offset + seq_len, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, head_dim/2)
        sin = jnp.sin(freqs)[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        cos = jnp.cos(freqs)[None, :, None, :]
        return sin, cos

    def apply_rotary(self, x: Array, sin: Array, cos: Array):
        # x shape: (..., seq_len, num_heads, head_dim)
        *prefix, seq_len, num_heads, head_dim = x.shape
        x2 = x.reshape(*prefix, seq_len, num_heads, head_dim // 2, 2)
        x1 = x2[..., 0] * cos - x2[..., 1] * sin
        x2b = x2[..., 1] * cos + x2[..., 0] * sin
        return jnp.stack([x1, x2b], axis=-1).reshape(*prefix, seq_len, num_heads, head_dim)

    def __call__(self, x: Array, seq_len: int, offset: int = 0) -> Array:
        sin, cos = self.get_sin_cos(seq_len, offset)
        return self.apply_rotary(x, sin, cos)


class Attention(nnx.Module):
    """Multi-head Self Attention with RoPE, Causal Masking, and KV caching"""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, *, rngs: nnx.Rngs):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        weight_initializer = nnx.initializers.variance_scaling(scale=1e-3, mode='fan_avg', distribution='uniform')
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=weight_initializer, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=weight_initializer, rngs=rngs)
        self.rope = RoPE(self.head_dim)

    def __call__(self, x: Array, cache_k: Array | None = None, cache_v: Array | None = None):
        B, N, C = x.shape
        
        # Compute qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.moveaxis(qkv, 2, 0)  # (3, B, N, H, Dh)
        q, k, v = [t.squeeze(0) for t in jnp.split(qkv, 3, axis=0)]  # (B,N,H,Dh)
        
        # Determine position offset for RoPE when using cache
        pos_offset = cache_k.shape[1] if cache_k is not None else 0
        
        # Apply RoPE to q and k
        q = self.rope(q, N, offset=pos_offset)
        k = self.rope(k, N, offset=pos_offset)
        
        # Handle KV caching
        if cache_k is not None and cache_v is not None:
            # Concatenate current keys/values with cached ones
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)
        
        # Create causal mask
        total_seq_len = k.shape[1]  # Total sequence length including cache
        mask = nnx.make_causal_mask(jnp.ones((B, total_seq_len)))
        
        # For generation with caching, we typically only need attention for the last token
        if N == 1 and total_seq_len > 1:
            mask = mask[:, :, -1:, :]  # (B, 1, 1, total_seq_len)
            
        # Compute attention
        attn = nnx.dot_product_attention(q, k, v, mask=mask, broadcast_dropout=False)
        out = attn.reshape(B, N, C)
        out = self.proj(out)
        
        # Return output and updated cache
        return out, k, v


class SiLU(nnx.Module): # For use within sequential blocks (Not in use)
    def __call__(self, x: Array) -> Array:
        return nnx.silu(x)


class DecayDropout(nnx.Module):
    """
    A Dropout layer where the rate decays from `start_rate` to `end_rate`
    based on the number of forward passes (steps).
    """
    def __init__(
        self,
        start_rate: float = 0.9,
        end_rate: float = 0.1,
        decay_scale: int = 1000,
        *,
        rngs: Optional[Union[nnx.Rngs, nnx.RngStream]] = None,
        rng_collection: str = 'dropout',
        deterministic: bool = False
    ):
        self.start_rate = nnx.Variable(jnp.array(start_rate))
        self.end_rate = nnx.Variable(jnp.array(end_rate))
        self.decay_scale = nnx.Variable(jnp.array(decay_scale, dtype=jnp.int32))
        self.rng_collection = rng_collection
        self.deterministic = deterministic

        # 1. State Management:
        # We use nnx.Variable to store the step. This ensures it is treated
        # as a JAX array, can be updated inside JIT, and is included in checkpoints.
        self.step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

        # 2. RNG Handling:
        # We setup an RngStream. This allows us to just call self.rngs()
        # later without manual splitting.
        if isinstance(rngs, nnx.Rngs):
            self.rngs = rngs[self.rng_collection].fork()
        elif isinstance(rngs, nnx.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            # Handle case where no RNG is provided (e.g. for inference-only init)
            self.rngs = nnx.RngStream(rng_collection, tag='decay_dropout').fork()
        else:
            raise TypeError(f"rngs must be nnx.Rngs or nnx.RngStream, got {type(rngs)}")

    def __call__(self, x: jax.Array, *, deterministic: Optional[bool] = None) -> jax.Array:
        # 3. Argument Precedence:
        # Check call-time argument first, then fall back to class attribute
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="No `deterministic` arg provided."
        )

        # If deterministic (Eval mode), return input as-is and DO NOT increment step
        if deterministic:
            return x

        # 4. Logic & Update:
        # Read the value from the variable
        current_step = self.step.get_value()

        # Calculate current rate
        # Formula: rate = (start - end) / (step / scale + 1) + end
        rate = (self.start_rate - self.end_rate) / \
               (current_step / self.decay_scale + 1) + self.end_rate

        # Update the step counter (In-place update logic handled by NNX)
        self.step.value += 1

        # 5. Apply Dropout:
        # Use the RngStream to generate a new key
        key = self.rngs()
        keep_prob = 1.0 - rate

        # 6. Safety Checks:
        # Handle edge cases to prevent NaNs or unnecessary computation
        safe_keep_prob = jnp.where(keep_prob > 0., keep_prob, 1.0)

        # Standard dropout logic
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        return jax.lax.select(mask, x / safe_keep_prob, jnp.zeros_like(x))


class GLU(nnx.Module):
    """Gated Linear Unit (GLU) with sigmoid activation and dropout"""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float|dict, *, rngs: nnx.Rngs):
        self.fc_gate_value = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, out_features, rngs=rngs)
        if isinstance(dropout, dict):
            self.dropout = DecayDropout(**dropout, rngs=rngs)
        else:
            self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        gate_val_proj = self.fc_gate_value(x)
        gated_values, gate_input = jnp.split(gate_val_proj, 2, axis=-1)
        gated_activation = nnx.sigmoid(gated_values) * gate_input
        gated_activation = self.dropout(gated_activation)
        out = self.fc_out(gated_activation)
        return out


class SwiGLU(nnx.Module):
    """SwiGLU activation: x * SiLU(Wx + b) * (Vx + c)"""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float|dict, *, rngs: nnx.Rngs):
        self.wv_proj = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_features, rngs=rngs)
        if isinstance(dropout, dict):
            self.dropout = DecayDropout(**dropout, rngs=rngs)
        else:
            self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        wv = self.wv_proj(x)
        w, v = jnp.split(wv, 2, axis=-1)
        # SwiGLU: x * SiLU(Wx + b) * (Vx + c)
        activated = w * nnx.silu(v)
        activated = self.dropout(activated)
        return self.out_proj(activated)


class GeneralGLU(nnx.Module):
    """General Gated Linear Unit (GLU) with chosen activation and dropout"""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, 
                 dropout: float|dict, gate_activation: Callable, *, rngs: nnx.Rngs):
        self.wv_proj = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_features, rngs=rngs)
        if isinstance(dropout, dict):
            self.dropout = DecayDropout(**dropout, rngs=rngs)
        else:
            self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.gate_fn = gate_activation

    def __call__(self, x: Array) -> Array:
        wv = self.wv_proj(x)
        w, v = jnp.split(wv, 2, axis=-1)
        # General GLU: x * Gate_fn(Wx + b) * (Vx + c)
        activated = w * self.gate_fn(v)
        activated = self.dropout(activated)
        return self.out_proj(activated)


class TransformerBlock(nnx.Module):
    """Transformer Block with Multi-head Self Attention, RoPE, GLU MLP, LayerNorm, and Residual Connections"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float|dict, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = SwiGLU(in_features=dim, hidden_dim=mlp_hidden_dim, out_features=dim, dropout=dropout, rngs=rngs)

    def __call__(self, x: Array, cache_k: Array | None = None, cache_v: Array | None = None):
        # Attention with potential caching
        attn_out, new_k, new_v = self.attn(self.norm1(x), cache_k=cache_k, cache_v=cache_v)
        x = x + attn_out
        # Feed-forward network
        x = x + self.ffn(self.norm2(x))
        return x, new_k, new_v