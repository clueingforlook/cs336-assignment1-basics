import torch
import torch.nn.functional as F
from torch import nn
from einops import einsum


class MySiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w_2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = einsum(x, self.w_1, "batch seq d_model, d_ff d_model -> batch seq d_ff")
        hidden = F.silu(hidden)
        return einsum(hidden, self.w_2, "batch seq d_ff, d_model d_ff -> batch seq d_model")
