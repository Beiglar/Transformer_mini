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
        layers = [TransformerBlock(dim, num_heads, mlp_ratio, dropout, rngs=rngs) for _ in range(num_layers)]
        self.layers = nnx.List(layers) if flax.__version__ >= '0.12.0' else layers # type: ignore

        # main LM head(s)
        self.head = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
            nnx.Linear(
                dim, vocab_size,
                kernel_init=nnx.initializers.uniform(scale=1e-3),
                bias_init=nnx.initializers.zeros,
                rngs=rngs))

    def __call__(self, tokens: jax.Array, caches: list[tuple[Array, Array]] | None = None):
        """
        tokens: (B, T) integer token IDs
        caches: List of (key_cache, value_cache) tuples for each layer
        returns:
          logits: Array of (B, T, V) for predicting next tokens
          new_caches: Updated caches for each layer
        """
        B, T = tokens.shape
        x = self.embed(tokens)  # (B, T, dim)
        
        new_caches = []
        current_caches = caches if caches is not None else [None] * len(self.layers)
        
        # pass through transformer blocks
        for i, layer in enumerate(self.layers):
            cache = current_caches[i]
            cache_k, cache_v = cache if cache is not None else (None, None)
            x, new_k, new_v = layer(x=x, cache_k=cache_k, cache_v=cache_v)
            new_caches.append((new_k, new_v))
            
        logits = self.head(x)  # (B, T, V)
        return logits, new_caches

    def generate(
            self, 
            initial_tokens: jax.Array, 
            max_new_tokens: int, 
            top_p: float, 
            temperature: float, 
            rng: PRNGKey) -> jax.Array:
        """
        Generate text with KV caching for efficient inference
        """
        self.eval()
        B, T = initial_tokens.shape
        assert T >= 1, "Need at least one initial token to start generation"
        tokens = initial_tokens
        
        # Initialize caches for each layer
        caches = None
        
        for step in range(max_new_tokens):
            if step == 0:
                # First step: process all context tokens
                context = tokens[:, -self.context_size:] if self.context_size is not None else tokens
                logits, caches = self(context)
                next_token_logits = logits[:, -1, :]
            else:
                # Subsequent steps: only process the last token with caching
                last_token = tokens[:, -1:]
                logits, caches = self(last_token, caches=caches)
                next_token_logits = logits[:, 0, :]  # Only one token position
                
            rng, subkey = jax.random.split(rng)
            next_token = top_p_sample(subkey, next_token_logits, top_p=top_p, temperature=temperature)
            next_token = next_token[:, None]  # (B, 1)
            tokens = jnp.concatenate([tokens, next_token], axis=1)
            
        return tokens