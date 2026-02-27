import torch
from torch import nn

class MyRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        处理一个形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。

        parameters:
            - d_model: int Hidden dimension of the model
            - eps: float = 1e-5 Epsilon value for numerical stability
            - device: torch.device | None = None Device to store the parameters on
            - dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        parameters:
            - x: Float[Tensor, "... d_model"] The input tensor to be normalized.

        Returns:
            Float[Tensor, "... d_model"]: The normalized output tensor.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim = True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
        