import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class MyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
    ):
        """
        Parameters:
            - d_model: Dimensionality of the Transformer block inputs.
            - num_heads: Number of attention heads.
            - d_ff: Dimensionality of the feed-forward inner layer.
            - max_seq_len: Maximum sequence length for RoPE precomputation.
            - theta: RoPE theta.
            - eps: Epsilon for RMSNorm.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        from cs336_basics.MyRMSNorm import MyRMSNorm
        from cs336_basics.MySwiGLU import MySwiGLU

        self.ln1 = MyRMSNorm(d_model, eps=eps)
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.ln2 = MyRMSNorm(d_model, eps=eps)
        self.ffn = MySwiGLU(d_model, d_ff)

    def _normalize_ffn_weights(self, w1: Tensor, w2: Tensor, w3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Internal MySwiGLU layout is (d_ff, d_model), (d_model, d_ff), (d_ff, d_model).
        internal_layout = (
            (self.d_ff, self.d_model),
            (self.d_model, self.d_ff),
            (self.d_ff, self.d_model),
        )
        # Some adapters/documentation use the transposed layout.
        transposed_layout = (
            (self.d_model, self.d_ff),
            (self.d_ff, self.d_model),
            (self.d_model, self.d_ff),
        )
        current_layout = (tuple(w1.shape), tuple(w2.shape), tuple(w3.shape))

        if current_layout == internal_layout:
            return w1, w2, w3
        if current_layout == transposed_layout:
            return w1.transpose(0, 1), w2.transpose(0, 1), w3.transpose(0, 1)

        raise ValueError(
            "Unexpected FFN weight shapes. "
            f"Got w1/w2/w3={current_layout}, expected either {internal_layout} or {transposed_layout}."
        )

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        q = weights.get("attn.q_proj.weight", weights.get("q_proj_weight"))
        k = weights.get("attn.k_proj.weight", weights.get("k_proj_weight"))
        v = weights.get("attn.v_proj.weight", weights.get("v_proj_weight"))
        o = weights.get("attn.output_proj.weight", weights.get("o_proj_weight"))
        ln1_w = weights.get("ln1.weight", weights.get("ln1_weight"))
        ln2_w = weights.get("ln2.weight", weights.get("ln2_weight"))
        w1 = weights.get("ffn.w1.weight", weights.get("ffn_w_1"))
        w2 = weights.get("ffn.w2.weight", weights.get("ffn_w_2"))
        w3 = weights.get("ffn.w3.weight", weights.get("ffn_w_3"))

        missing = [
            key
            for key, value in {
                "q_proj": q,
                "k_proj": k,
                "v_proj": v,
                "o_proj": o,
                "ln1": ln1_w,
                "ln2": ln2_w,
                "ffn.w1": w1,
                "ffn.w2": w2,
                "ffn.w3": w3,
            }.items()
            if value is None
        ]
        if missing:
            raise KeyError(f"Missing required keys in weights: {missing}")

        w1, w2, w3 = self._normalize_ffn_weights(w1, w2, w3)

        with torch.no_grad():
            self.q_proj_weight.copy_(q)
            self.k_proj_weight.copy_(k)
            self.v_proj_weight.copy_(v)
            self.o_proj_weight.copy_(o)
            self.ln1.weight.copy_(ln1_w)
            self.ln2.weight.copy_(ln2_w)
            self.ffn.w_1.copy_(w1)
            self.ffn.w_2.copy_(w2)
            self.ffn.w_3.copy_(w3)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"] | None = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        from cs336_basics.runCausalMultiHeadSelfAttention import run_multihead_self_attention_with_rope

        x = in_features
        attn_out = run_multihead_self_attention_with_rope(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            o_proj_weight=self.o_proj_weight,
            in_features=self.ln1(x),
            token_positions=token_positions,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x
