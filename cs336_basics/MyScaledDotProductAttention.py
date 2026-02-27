import torch
from torch import nn
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified dimension.
    Args:
        - x (torch.Tensor): The input tensor.
        - dim (int): The dimension along which to compute the softmax.
    Returns:
        - torch.Tensor: The softmax of the input tensor along the specified dimension.
    """
    # Subtract the maximum value from each element for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim = dim, keepdim=True)
    return x_exp / x_sum

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    Args:
        - Q (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len_q, d_k).
        - K (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len_k, d_k).
        - V (torch.Tensor): Value tensor of shape (batch_size, ..., d_v).
        - mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len_q, seq_len_k). Defaults to None.
    Returns:
        - torch.Tensor: The output of the attention mechanism of shape (batch_size, ..., seq_len_q, d_v).
    """
    d_k = Q.size(dim=-1)
    attention = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / d_k ** 0.5

    if mask is not None:
        attention = attention.masked_fill_(mask == False, float('-inf'))
    attention = softmax(attention, dim=-1)
    out = einsum(attention, V, "... seq_len_q seq_len_k , ... seq_len_k d_v -> ... seq_len_q d_v")
    return out