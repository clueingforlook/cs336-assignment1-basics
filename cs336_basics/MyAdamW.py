import torch
from typing import Optional
from collections.abc import Callable, Iterable
class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), weight_decay=0.01, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历这组里面的每一个具体参数矩阵 param
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param.data)
                    state["v"] = torch.zeros_like(param.data)
                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]


                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) *grad * grad
                lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                # Update the parameters
                param.data = param.data - lr_t * m / ((v + eps) ** 0.5)
                # Apply weight decay
                param.data = param.data - lr * weight_decay * param.data
                state["m"], state["v"] = m, v
        return loss




        