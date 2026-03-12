import torch
from torch import nn
from einops import einsum


class MyRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Parameters:
            - theta: float value for the RoPE
            - d_k: int dimension of query and key vectors
            - max_seq_len: int maximum sequence length that will be inputted
            - device: torch.device | None = None device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = 1 / theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        positions = torch.arange(max_seq_len, device=device)

        self.angles = einsum(positions, freqs, "i, j -> i j")
        self.register_buffer("cos_cached", torch.cos(self.angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(self.angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor x.

        Args:
            - x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            - token_positions (torch.Tensor): Tensor of shape (..., seq_len) indicating token positions.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, shape same as x.
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        q1, q2 = x[..., ::2], x[..., 1::2]

        # Allow token_positions to omit broadcastable dimensions such as heads.
        while cos.ndim < q1.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        q1_rotated = q1 * cos - q2 * sin
        q2_rotated = q1 * sin + q2 * cos

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = q1_rotated
        x_rotated[..., 1::2] = q2_rotated
        return x_rotated
