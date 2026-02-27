import torch
from torch import nn
from einops import einsum

class MySwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        """
        Parameters:
            - w_1: Linear layer that maps from d_model to d_ff, (d_ff * d_model) 
            - w_2: Linear layer that maps from d_ff to d_model, (d_model * d_ff) 
            - w_3: Linear layer that maps from d_model to d_ff, (d_ff * d_model) 
        """
        self.w_1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w_2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w_3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        parameters:
            - x: Float[Tensor, "... d_model"] The input tensor to be processed.

        Returns:
            Float[Tensor, "... d_model"]: The output tensor after applying the SwiGLU transformation.
        """
        w1_out = einsum(x, self.w_1.data, "batch seq d_model, d_ff d_model -> batch seq d_ff")
        """
        SiLU(x) = x * sigmoid(x)
         = x / (1 + exp(-x))
         = 1 / (1 + exp(-x)) * x
         = sigmoid(x) * x
         = w1_out * sigmoid(w1_out)
         = w1_out / (1 + exp(-w1_out))
        """
        gate = w1_out / (1 + torch.exp(-w1_out))
        w3_out = einsum(x, self.w_3.data, "batch seq d_model, d_ff d_model -> batch seq d_ff")
        result = einsum(self.w_2, (gate * w3_out), "d_model d_ff, batch seq d_ff -> batch seq d_model")
        return result