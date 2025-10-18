import jax
import jax.numpy as jnp
from flax import nnx
import jaxlib
from typing import Tuple, TypeAlias, Union, Callable

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

    def get_sin_cos(self, seq_len: int) -> Tuple[Array, Array]:
        t = jnp.arange(seq_len, dtype=jnp.float32)
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

    def __call__(self, x: Array, seq_len: int) -> Array:
        sin, cos = self.get_sin_cos(seq_len)
        return self.apply_rotary(x, sin, cos)


class Attention(nnx.Module):
    """Multi-head Self Attention with RoPE and Causal Masking"""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, *, rngs: nnx.Rngs):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        weight_initializer = nnx.initializers.variance_scaling(scale=1e-3, mode='fan_avg', distribution='uniform')
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, kernel_init=weight_initializer, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=weight_initializer, rngs=rngs)
        self.rope = RoPE(self.head_dim)

    def __call__(self, x: Array):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.moveaxis(qkv, 2, 0)  # (3, B, N, H, Dh)
        q, k, v = [t.squeeze(0) for t in jnp.split(qkv, 3, axis=0)]  # (B,N,H,Dh)

        # Apply RoPE to q and k (safe: seq_len=N)
        q = self.rope(q, N)
        k = self.rope(k, N)

        # Create and apply the causal mask
        mask = nnx.make_causal_mask(jnp.ones((B, N)))

        # nnx.dot_product_attention picks optimized path when dropout/deterministic isn't used.
        attn = nnx.dot_product_attention(q, k, v, mask=mask, broadcast_dropout=False)
        out = attn.reshape(B, N, C)
        out = self.proj(out)
        return out


class SiLU(nnx.Module): # For use within sequential blocks (Not in use)
    def __call__(self, x: Array) -> Array:
        return nnx.silu(x)


class GLU(nnx.Module):
    """Gated Linear Unit (GLU) with sigmoid activation and dropout"""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float, *, rngs: nnx.Rngs):
        self.fc_gate_value = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, out_features, rngs=rngs)
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
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float, *, rngs: nnx.Rngs):
        self.wv_proj = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_features, rngs=rngs)
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
                 dropout: float, gate_activation: Callable, *, rngs: nnx.Rngs):
        self.wv_proj = nnx.Linear(in_features, 2 * hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_features, rngs=rngs)
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
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = SwiGLU(in_features=dim, hidden_dim=mlp_hidden_dim, out_features=dim, dropout=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x