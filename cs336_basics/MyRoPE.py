import torch
from torch import nn
from einops import einsum
class MyRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Parameters:
            - theta: float Θ value for the RoPE
            - d_k: int dimension of query and key vectors
            - max_seq_len: int Maximum sequence length that will be inputted
            - device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = 1 / theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        positions = torch.arange(max_seq_len, device=device)

        # angles:(max_seq_len, d_k / 2)
        self.angles = einsum(positions, freqs, "i, j -> i j")
        cos_cached = torch.cos(self.angles)
        sin_cached = torch.sin(self.angles)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to the input tensor x.

        Args:
            - x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            - token_positions (torch.Tensor): Tensor of shape (..., seq_len) indicating token positions.

        Returns:
            torch.Tensor: Output tensor with RoPE applied, shape same as x.
        """
        # 抽出来的 cos 和 sin 形状将是 (..., seq_len, d_k / 2)
        cos = self.cos_cached[token_positions]  
        sin = self.sin_cached[token_positions]  
        q1, q2 = x[..., ::2], x[..., 1::2]  # 将 x 分成两部分，q1 和 q2
        q1_rotated = q1 * cos - q2 * sin  # 旋转后的 q1
        q2_rotated = q1 * sin + q2 * cos  # 旋转后的 q2

        # 将旋转后的 q1 和 q2 交错在一起，恢复原来的形状
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = q1_rotated
        x_rotated[..., 1::2] = q2_rotated
        return x_rotated

        
