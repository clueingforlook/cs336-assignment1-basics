import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
from typing import Iterable

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



def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """
    计算交叉熵损失函数 (支持任意维度的 batch_shape)
    
    Args:
        logits: 模型预测的原始未归一化分数，形状为 (*batch_shape, vocab_size)
        targets: 真实的下一个词的 Token ID，形状为 (*batch_shape)
        
    Returns:
        loss: 标量，整个批次的平均损失
    """
    max_logits = torch.max(logits, dim = -1, keepdim=True).values   # ... 1
    # 注意这一步 sum 之后最后一维消失了，形状变成了 [...]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits),dim = -1)) 

    # ...
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(dim=-1)).squeeze(dim=-1)
    loss = -target_logits + max_logits.squeeze(dim=-1) + log_sum_exp
    return loss.mean()

def lr_cosine_schedule(
    t:int,
    max_lr:float,
    min_lr:float,
    t_warm_up:int,
    t_c:int,
) -> float:
    """
    计算学习率的余弦退火调度值。
    args:
        t: 当前的训练步数
        max_lr: 预热阶段结束时的最大学习率
        min_lr: 退火阶段结束时的最小学习率
        t_warm_up: 预热阶段的步数
        t_c: 退火阶段的总步数（不包括预热阶段）
    """
    if t_c <= t_warm_up:
        raise ValueError("t_c must be greater than t_warm_up")
    # 预热阶段
    if t < t_warm_up:
        return t * max_lr / t_warm_up
    # 退火阶段
    elif t < t_c:
        # 去掉 torch.tensor，把 torch.cos 换成 math.cos，math.pi 是自带的 π
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos((t - t_warm_up) / (t_c - t_warm_up) * math.pi))
    return min_lr

def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    """
    实现全局梯度裁剪 (Global Gradient Clipping)。
    
    参数:
        parameters: 一个包含模型参数的列表或迭代器 (通常是 model.parameters())
        max_norm: 梯度的最大 L2 范数
        eps: 用于数值稳定性的微小值，默认为 1e-6
    """
    # 1. 过滤掉没有梯度的参数 (例如被冻结的层)
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if not params_with_grad:
        return

    # 2. 计算全局 L2 范数 ||g||_2
    # 将所有参数的梯度的平方求和
    total_norm_sq = 0.0
    for p in params_with_grad:
        # 使用 .detach() 是因为我们不需要对裁剪操作本身求导
        total_norm_sq += p.grad.detach().pow(2).sum().item()
        
    # 开根号得到全局范数
    total_norm = total_norm_sq ** 0.5

    # 3. 判断是否超过最大阈值 M
    if total_norm > max_norm:
        # 4. 计算缩放因子: M / (||g||_2 + eps)
        scale_factor = max_norm / (total_norm + eps)
        
        # 5. 原位 (in-place) 修改梯度
        for p in params_with_grad:
            # 在 PyTorch 中，带有下划线的方法 (如 .mul_()) 表示原位操作
            p.grad.detach().mul_(scale_factor)