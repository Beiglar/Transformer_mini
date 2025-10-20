from modular.nnx_modules import *
from modular.sampling import *
import flax

class TinyTransformerLM(nnx.Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.5,
            context_size: int | None = None,
            *, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if dropout < 0 or dropout > 1:
            raise ValueError(f"dropout ({dropout}) must be between 0 and 1")

        self.vocab_size = vocab_size
        self.dim = dim
        self.context_size = context_size if context_size is not None else 1024  # default context size
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.pos_linear = None  # not using learned positional embeddings; we use RoPE in attention
        layers = [TransformerBlock(dim, num_heads, mlp_ratio, dropout, rngs=rngs) for _ in range(num_layers)]
        self.layers = nnx.List(layers) if flax.__version__ >= '0.12.0' else layers # type: ignore

        # main LM head(s)
        self.head = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
            nnx.Linear(
                dim, vocab_size,
                kernel_init=nnx.initializers.uniform(scale=1e-3), # Uniform [-scale, scale]
                bias_init=nnx.initializers.zeros,
                rngs=rngs))  # optionally tie to embed

    def __call__(self, tokens: jax.Array) -> Array:
        """
        tokens: (B, T) integer token IDs
        returns dict with:
          'logits': Array of (B, T, V) for predicting t+1..t+H (list len)
        """
        B, T = tokens.shape
        x = self.embed(tokens)  # (B, T, dim)
        # pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)  # (B, T, V)
        return logits

    def generate(
            self, 
            initial_tokens: jax.Array, 
            max_new_tokens: int, 
            top_p: float, 
            temperature: float, 
            rng: PRNGKey) -> jax.Array:
        """
        initial_tokens: (B, T) integer token IDs
        returns: (B, T + max_new_tokens) integer token IDs
        """
        self.eval()
        B, T = initial_tokens.shape
        assert T >= 1, "Need at least one initial token to start generation"
        tokens = initial_tokens

        for _ in range(max_new_tokens):
            # Only process the last N tokens to maintain efficiency
            context = tokens[:, -self.context_size:] if hasattr(self, 'context_size') else tokens
            logits = self(context)
            next_token_logits = logits[:, -1, :]
            rng, subkey = jax.random.split(rng)
            next_token = top_p_sample(subkey, next_token_logits, top_p=top_p, temperature=temperature)  # (B,)
            next_token = next_token[:, None]  # (B, 1)
            tokens = jnp.concatenate([tokens, next_token], axis=1)  # (B, T+1)
        return tokens
