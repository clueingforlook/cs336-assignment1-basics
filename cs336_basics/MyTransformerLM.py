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
        Parameters:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
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
        self.ln_final.weight.data = self.weights['ln_final.weight']
        self.lm_head = self.weights['lm_head.weight']

        self._load_weights(weights)
        
    def _load_weights(self, weights: dict[str, torch.Tensor]):
        self.embedding.weight.data = weights['token_embeddings.weight']
        self.ln_final.weight.data = weights['ln_final.weight']
        self.lm_head.data = weights['lm_head.weight']

        for i,block in enumerate(self.blocks):
            layer_prefix = f'layers.{i}.'
            layer_weights = {k[len(layer_prefix):]:v for k, v in weights.items() if k.startswith(layer_prefix)}
            block.load_weights(layer_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embedding layer
        self.embedding.weight.data = self.weights['token_embeddings.weight']
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)
        x = einsum(x, self.lm_head, "... d_model, vocab_size d_model -> ... vocab_size")
        # x = softmax(x, dim=-1)
        return x


