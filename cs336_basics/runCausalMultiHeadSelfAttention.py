import torch
from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum, rearrange
from cs336_basics.MyScaledDotProductAttention import attention
from cs336_basics.MyRoPE import MyRoPE


def run_multihead_self_attention(
                 d_model: int,
                 num_heads: int,
                 q_proj_weight: Float[Tensor, " d_out d_model"],
                 k_proj_weight: Float[Tensor, " d_out d_model"],
                 v_proj_weight: Float[Tensor, " d_out d_model"],
                 o_proj_weight: Float[Tensor, " d_model d_v"],
                 in_features: Float[Tensor, " ... sequence_length d_model"]
                ) -> Float[Tensor, " ... sequence_length d_model"]:
    """
            Initialize the MyCausalMultiHeadSelfAttention module.
            Args:
                d_model (int): The dimension of the model.
                num_heads (int): The number of attention heads.
    """        
    
    # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    Q = einsum(in_features, q_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")
    K = einsum(in_features, k_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")
    V = einsum(in_features, v_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")

    
    # 重构为多头模式
    Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=num_heads)
    K = rearrange(K, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=num_heads)
    V = rearrange(V, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads=num_heads)

   
    
    # 构造mask
    seq_len = in_features.size(dim=-2)
    casual_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device))
    
    out = attention(Q, K, V, mask=casual_mask)

    out = rearrange(out, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")
    final_out = einsum(out, o_proj_weight, "... sequence_length d_in, d_out d_in -> ... sequence_length d_out")
    
    
    return final_out

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    实现带RoPE的多头自注意力机制，基于 run_scaled_dot_product_attention

    Args:
        d_model (int): 模型的维度
        num_heads (int): 注意力头的数量
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q投影权重，包含所有头的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K投影权重，包含所有头的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V投影权重，包含所有头的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 输入特征
        rope_theta (float): RoPE参数
        token_positions (Int[Tensor, "... sequence_length"] | None): token位置信息

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: 多头自注意力的输出
    """
    batch_shape = in_features.shape[:-2]  # 除了最后两个维度 (sequence_length, d_model)
    seq_len = in_features.shape[-2]
    
    

    # 计算每个头的维度
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    Q = einsum(in_features, q_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")
    K = einsum(in_features, k_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")
    V = einsum(in_features, v_proj_weight, "... sequence_length d_model, d_out d_model -> ... sequence_length d_out")

    # 重构为多头模式
    Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=num_heads)
    K = rearrange(K, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=num_heads)
    V = rearrange(V, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads=num_heads)

    

    # 计算Q K的RoPE位置编码
    if token_positions is None:
        token_positions = torch.arange(seq_len, dtype=torch.long, device=in_features.device)
        token_positions = token_positions.expand(*batch_shape, seq_len)
    rope = MyRoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_features.device)
    Q = rope(Q, token_positions)
    K = rope(K, token_positions)

  

    # 构造mask
    casual_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device))
   

    out_heads = attention(Q, K, V, mask=casual_mask)
    out = rearrange(out_heads, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")

    out = einsum(out, o_proj_weight, "... sequence_length d_in, d_out d_in -> ... sequence_length d_out")
 
    return out
