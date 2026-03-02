import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from cs336_basics.MyTransformerLM import MyTransformerLM
from cs336_basics.MyAdamW import MyAdamW
import os
import argparse

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

def get_batch(data:np.ndarray, batch_size, context_length,device='cuda:0'):
    """
    从输入序列 data 中随机采样一个批次的训练数据。
    
    参数:
        data(np.array 或 np.memmap): 输入序列，形状为 (seq_length,)
        batch_size(int): 批次大小
        context_length(int): 上下文长度，即模型输入的序列长度
        device(str): 设备字符串或 torch.device 对象
    返回:
        (x, y): 均为形状 (batch_size, context_length) 的 torch.Tensor，已移动到设备
    """
    # 1. 修复边界差一错误：randint 的上限是不包含 (exclusive) 的
    # 最大允许的起始索引 i 必须满足：i + 1 + context_length <= len(data)
    # 所以传入 randint 的 high 应该是 len(data) - context_length
    ix = np.random.randint(0, len(data) - context_length, size=batch_size)
    
    # 2. 修复内存加载问题：针对 np.memmap 放弃高级索引，改用连续切片 (Slicing)
    # 连续切片能最大化利用磁盘顺序读取和 OS 缓存，避免内存瞬间撑爆
    x_list = [data[i : i + context_length].astype(np.int64) for i in ix]
    y_list = [data[i + 1 : i + 1 + context_length].astype(np.int64) for i in ix]
    
    # 3. 转换为 Tensor、组合成 Batch 并移动到指定的 Device
    x = torch.stack([torch.from_numpy(arr) for arr in x_list]).to(device)
    y = torch.stack([torch.from_numpy(arr) for arr in y_list]).to(device)

    return x, y

def save_checkpoint(model, optimizer, iteration, out):
    """
    保存模型检查点 (Checkpoint)。
    参数:
        model: 要保存的模型对象
        optimizer: 模型对应的优化器对象
        iteration: 当前的训练迭代次数 (int)
        out: 保存文件的路径 (str)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    """
    从检查点文件加载模型和优化器状态。
    参数:
        src: 检查点文件的路径 (str)
        model: 要加载状态的模型对象
        optimizer: 要加载状态的优化器对象
    返回:
        iteration: 从检查点恢复的训练迭代次数 (int)
    """
    checkpoint = torch.load(src, map_location='cpu')  # 加载到 CPU 上，后续再移动到正确设备
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def train():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    # ==========================================
    # 1. 模型架构超参数 (Model Hyperparameters)
    # ==========================================
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度 (m)")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer 隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer 块的数量")
    parser.add_argument("--num_heads", type=int, default=16, help="多头注意力的头数")

    # ==========================================
    # 2. 训练超参数 (Training Hyperparameters)
    # ==========================================
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="最大学习率")
    parser.add_argument("--max_iters", type=int, default=5000, help="最大训练迭代步数")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备 (cuda:0, cpu, mps)")

    # ==========================================
    # 3. 数据与 Checkpoint (5.2节的核心)
    # ==========================================
    parser.add_argument("--data_path", type=str, required=True, help="预训练数据的路径 (.npy 或 .bin)")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="保存 checkpoint 的文件夹路径")
    parser.add_argument("--resume_from", type=str, default=None, help="如果中断了，指定一个 .pt 文件的路径来恢复训练")
    parser.add_argument("--save_interval", type=int, default=500, help="每隔多少步保存一次 checkpoint")
    model = MyTransformerLM(
        vocab_size=10000,
        context_length=128,
        num_layers=4,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        rope_theta=100000.0,
        weights={}
    )
