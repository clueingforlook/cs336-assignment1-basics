import torch
from torch import nn
from einops import rearrange, einsum
from cs336_basics.MyTransformerBlock import MyTransformerBlock
from cs336_basics.MyEmbedding import MyEmbedding
from cs336_basics.MyScaledDotProductAttention import softmax
from cs336_basics.MyRMSNorm import MyRMSNorm
class MyTransformerLM(nn.Module):
    def __init__(self,vocab_size: int, 
                 context_length:int, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 rope_theta: float,
                 weights: dict[str, torch.Tensor]
                 ):
        """
        Build a pre-norm Transformer language model.

        Args:
            vocab_size: Vocabulary size of token IDs.
            context_length: Maximum sequence length supported by this model.
            num_layers: Number of Transformer blocks.
            d_model: Hidden dimension.
            num_heads: Number of attention heads. `d_model % num_heads == 0`.
            d_ff: Feed-forward hidden dimension.
            rope_theta: RoPE theta.
            weights: Flat state-dict-like tensor mapping used to initialize all modules.
                Required keys:
                - "token_embeddings.weight": (vocab_size, d_model)
                - "ln_final.weight": (d_model,)
                - "lm_head.weight": (vocab_size, d_model)
                - For each layer i in [0, num_layers):
                  "layers.{i}.attn.q_proj.weight": (d_model, d_model)
                  "layers.{i}.attn.k_proj.weight": (d_model, d_model)
                  "layers.{i}.attn.v_proj.weight": (d_model, d_model)
                  "layers.{i}.attn.output_proj.weight": (d_model, d_model)
                  "layers.{i}.ln1.weight": (d_model,)
                  "layers.{i}.ln2.weight": (d_model,)
                  "layers.{i}.ffn.w1.weight": (d_ff, d_model)
                  "layers.{i}.ffn.w2.weight": (d_model, d_ff)
                  "layers.{i}.ffn.w3.weight": (d_ff, d_model)

        Notes:
            - `weights` is treated as an initialization source. After initialization,
              `forward` uses module parameters directly (standard PyTorch behavior).
        """
        super().__init__()
        self.vocab_size = vocab_size    
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.embedding = MyEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MyTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            ) for _ in range(num_layers)
        ])
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))

        self._load_weights(weights)
        
    def _load_weights(self, weights: dict[str, torch.Tensor]):
        """Load top-level and per-layer tensors from a flat weight dictionary."""
        with torch.no_grad():
            self.embedding.weight.copy_(weights['token_embeddings.weight'])
            self.ln_final.weight.copy_(weights['ln_final.weight'])
            self.lm_head.copy_(weights['lm_head.weight'])

        for i,block in enumerate(self.blocks):
            layer_prefix = f'layers.{i}.'
            layer_weights = {k[len(layer_prefix):]:v for k, v in weights.items() if k.startswith(layer_prefix)}
            block.load_weights(layer_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (batch, sequence_length).
        Returns:
            Logits of shape (batch, sequence_length, vocab_size).
        """
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        # Final normalization + tied output projection.
        x = self.ln_final(x)
        x = einsum(x, self.lm_head, "... d_model, vocab_size d_model -> ... vocab_size")
        return x

