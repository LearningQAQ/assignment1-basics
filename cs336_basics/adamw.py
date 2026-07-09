from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdaW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                
                # 获取参数对应的状态字典
                state = self.state[p]
                # 获取迭代计数 t；若不存在则初始化为 0
                t = state.get('t', 0)
                # 获取梯度数据
                grad = p.grad.data
                
                # 初始化一阶矩和二阶矩（仅在首次迭代时执行）
                if 'm' not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  # 一阶矩
                    state["v"] = torch.zeros_like(p.data)  # 二阶矩
                
                m, v = state['m'], state['v']
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # 更新有偏一阶矩估计：m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新有偏二阶矩估计：v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正, 解决动量估计在训练初期（冷启动阶段）向零偏移的问题
                bias_correction1 = 1 - beta1 ** (t + 1)
                bias_correction2 = 1 - beta2 ** (t + 1)
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # 参数更新：p = p - lr * m_hat / (sqrt(v_hat) + eps)
                denom = v_hat.sqrt().add_(eps)
                p.data.addcdiv_(m_hat, denom, value=-lr)

                # 更新状态：t ← t + 1
                state['t'] = t + 1

        return loss
    

def cosine_annealing_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    elif it > cosine_cycle_iters:
        return min_learning_rate
    
    else:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosin_delay = (1 + math.cos(progress * math.pi)) / 2
        return min_learning_rate + cosin_delay * (max_learning_rate - min_learning_rate)
    

def gradient_clip(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_norm_sq = sum(p.grad.data.norm(2).pow(2) for p in params_with_grad)
    total_norm = torch.sqrt(total_norm_sq)

    if max_l2_norm < total_norm:
        clip = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(clip)