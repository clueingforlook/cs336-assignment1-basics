import torch
from torch import nn
from einops import rearrange, einsum

class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Constructs a linear transformation module without bias.

        Args:
            - in_features (int): Size of each input sample.
            - out_features (int): Size of each output sample.
            - device (torch.device | None): Device to store the parameters on.
            d- type (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features,self.in_features, device=self.device, dtype=self.dtype))

        mu = 0.0
        std = 2/(self.in_features + self.out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=mu, std=std, a=-3*std, b=3*std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return einsum(x,self.W,"... d_in, ... d_out d_in -> ... d_out")