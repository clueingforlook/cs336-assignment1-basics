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
    """
    End-to-end training entry for Assignment 1 section 5.3.

    This function wires together:
    - MyTransformerLM: language model forward pass -> logits
    - cross_entropy: logits + next-token labels -> scalar loss
    - MyAdamW: parameter update
    - lr_cosine_schedule: per-step learning-rate schedule
    - gradient_clipping: global gradient norm clipping
    - get_batch: random (x, y) language-model batch sampling
    - save_checkpoint / load_checkpoint: training state persistence
    """
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Data and checkpointing
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"])
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    # Deterministic-ish initialization for reproducibility.
    # (Full determinism may still require additional backend flags.)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection:
    # - Use requested device when available
    # - Fallback to CPU if CUDA/MPS is unavailable
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False):
        print("MPS not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.num_heads <= 0 or args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    min_learning_rate = args.min_learning_rate
    if min_learning_rate is None:
        min_learning_rate = args.learning_rate * 0.1

    # Dataset interface expected by get_batch:
    # - 1D integer token array (np.ndarray or np.memmap)
    # - each element is a token id
    ext = os.path.splitext(args.data_path)[1].lower()
    if ext == ".npy":
        # npy supports shape metadata, load as memory-mapped when possible.
        data = np.load(args.data_path, mmap_mode="r")
    elif ext == ".bin":
        # raw bin requires explicit dtype; memory-map avoids loading whole corpus.
        dtype_map = {
            "uint16": np.uint16,
            "int32": np.int32,
            "int64": np.int64,
        }
        data = np.memmap(args.data_path, dtype=dtype_map[args.data_dtype], mode="r")
    else:
        raise ValueError("data_path must end with .npy or .bin")

    if data.ndim != 1:
        raise ValueError("Training data must be a 1D token id array.")
    if len(data) <= args.context_length:
        raise ValueError("Training data is too short for the configured context_length.")

    d_ff = 4 * args.d_model

    # Parameter initialization helper for matrix weights.
    # Tensor shape contracts must match MyTransformerLM/MyTransformerBlock expectations.
    def init_matrix(shape: tuple[int, ...]) -> Tensor:
        t = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(t)
        return t

    # Flat weight dictionary consumed by MyTransformerLM(weights=...).
    # Required top-level keys:
    # - token_embeddings.weight: (vocab_size, d_model)
    # - ln_final.weight: (d_model,)
    # - lm_head.weight: (vocab_size, d_model)
    init_weights: dict[str, Tensor] = {
        "token_embeddings.weight": init_matrix((args.vocab_size, args.d_model)),
        "ln_final.weight": torch.ones(args.d_model, dtype=torch.float32),
        "lm_head.weight": torch.nn.Parameter(init_matrix((args.vocab_size, args.d_model))),
    }
    # Required per-layer keys for MyTransformerBlock.load_weights(...):
    # - attn.{q,k,v,output}_proj.weight
    # - ln1.weight, ln2.weight
    # - ffn.w{1,2,3}.weight
    for layer_idx in range(args.num_layers):
        prefix = f"layers.{layer_idx}."
        init_weights[prefix + "attn.q_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.k_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.v_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.output_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "ln1.weight"] = torch.ones(args.d_model, dtype=torch.float32)
        init_weights[prefix + "ln2.weight"] = torch.ones(args.d_model, dtype=torch.float32)
        init_weights[prefix + "ffn.w1.weight"] = init_matrix((d_ff, args.d_model))
        init_weights[prefix + "ffn.w2.weight"] = init_matrix((args.d_model, d_ff))
        init_weights[prefix + "ffn.w3.weight"] = init_matrix((d_ff, args.d_model))

    model = MyTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        weights=init_weights,
    )
    if not isinstance(model.lm_head, torch.nn.Parameter):
        model.lm_head = torch.nn.Parameter(model.lm_head)
    model = model.to(device)

    # Important for this MyTransformerLM implementation:
    # forward() reads from model.weights["token_embeddings.weight"].
    # Keep those entries pointing to real Parameters so gradients/updates are preserved.
    model.weights["token_embeddings.weight"] = model.embedding.weight
    model.weights["ln_final.weight"] = model.ln_final.weight
    model.weights["lm_head.weight"] = model.lm_head

    # Optimizer interface:
    # MyAdamW(params, lr, betas, weight_decay, eps)
    optimizer = MyAdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    start_iter = 0
    if args.resume_from is not None:
        # load_checkpoint restores:
        # - model.state_dict()
        # - optimizer.state_dict()
        # - iteration (returned)
        start_iter = load_checkpoint(args.resume_from, model, optimizer) + 1
        # If checkpoint was loaded on CPU, move optimizer state tensors to target device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        print(f"Resumed from {args.resume_from}, starting at step {start_iter}.")

    model.train()
    for step in range(start_iter, args.max_iters):
        # Per-step LR from linear warmup + cosine decay schedule.
        lr = lr_cosine_schedule(
            t=step,
            max_lr=args.learning_rate,
            min_lr=min_learning_rate,
            t_warm_up=args.warmup_iters,
            t_c=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # get_batch returns:
        # - x: (batch_size, context_length), token ids
        # - y: (batch_size, context_length), next-token labels
        x, y = get_batch(
            data=data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )
        # MyTransformerLM(x) -> logits: (batch_size, context_length, vocab_size)
        logits = model(x)
        # cross_entropy supports arbitrary batch shape [..., vocab_size] + [...]
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        # Global gradient clipping before optimizer step.
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_interval == 0 or step == start_iter:
            print(f"step {step + 1}/{args.max_iters} | loss {loss.item():.6f} | lr {lr:.8f}")

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.max_iters:
            # save_checkpoint serializes model state, optimizer state, and current iteration.
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_step_{step + 1}.pt")
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
