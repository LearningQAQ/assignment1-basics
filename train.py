"""
Transformer 语言模型训练脚本

用法:
    python train.py --train-data <训练数据路径> --val-data <验证数据路径> --out-dir <输出目录> [其他超参数...]

示例:
    python train.py
    # 所有超参数均有合理默认值，无需手动指定
"""
import os
import argparse
import time
import json
import math
import numpy as np
import torch
from torch import nn
from typing import Optional
from cs336_basics.model import Transformer
from cs336_basics.adamw import AdaW, cosine_annealing_lr
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.loss import cross_entropy

# 尝试导入 swanlab，如果不可用则使用简单的本地日志
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("警告: swanlab 未安装，将使用本地日志记录")
    print("安装命令: pip install swanlab")


def load_dataset(data_path: str) -> np.memmap:
    """
    使用 np.memmap 内存高效地加载数据集。

    Args:
        data_path: 数据文件路径（应该是 .bin 格式的 token ID 数组）

    Returns:
        np.memmap: 内存映射的数据集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 假设数据是以 uint16 或 int32 格式存储的 1D 数组
    # 尝试 uint16 先（较小的词汇表）
    try:
        dataset = np.memmap(data_path, dtype=np.uint16, mode='r')
    except Exception:
        # 如果失败，尝试 int32
        dataset = np.memmap(data_path, dtype=np.int32, mode='r')

    return dataset


def estimate_loss(
    model: nn.Module,
    train_dataset: np.memmap,
    val_dataset: Optional[np.memmap],
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int = 100,
) -> dict:
    """
    估计训练集和验证集的损失。

    Args:
        model: 模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集（可选）
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备
        eval_iters: 评估迭代次数

    Returns:
        dict: 包含训练损失和验证损失（如果有验证集）
    """
    out = {}
    model.eval()

    with torch.no_grad():
        # 训练集损失
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(train_dataset, batch_size, context_length, device)
            logits = model(x)
            # 展平 logits 和 targets
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = y.view(B * T)
            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())
        out['train_loss'] = np.mean(losses)

        # 验证集损失
        if val_dataset is not None:
            val_losses = []
            for _ in range(eval_iters):
                x, y = get_batch(val_dataset, batch_size, context_length, device)
                logits = model(x)
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                targets_flat = y.view(B * T)
                loss = cross_entropy(logits_flat, targets_flat)
                val_losses.append(loss.item())
            out['val_loss'] = np.mean(val_losses)

    model.train()
    return out


def train(
    # 数据参数
    train_data_path: str,
    val_data_path: Optional[str],
    vocab_size: int,

    # 模型参数（根据作业要求设置默认值）
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,  # 约 8/3 * d_model，64 的倍数
    rope_theta: float = 10000.0,

    # 优化器参数
    batch_size: int = 32,  # 批次大小
    learning_rate: float = 3e-4,  # 最大学习率
    min_learning_rate: float = 3e-5,  # 最小学习率（最大学习率的 1/10）
    weight_decay: float = 0.1,  # 权重衰减
    beta1: float = 0.9,
    beta2: float = 0.95,  # AdamW beta2
    eps: float = 1e-8,
    max_grad_norm: float = 1.0,  # 梯度裁剪

    # 学习率调度参数
    warmup_iters: int = 200,  # MPS 低资源：预热迭代次数减少（原 2000）

    # 训练控制参数
    # MPS 低资源：减少总 token 数到 40,000,000
    # batch_size(32) × max_iters(5000) × context_length(256) = 40,960,000 tokens
    max_iters: int = 5000,
    eval_interval: int = 500,
    eval_iters: int = 200,
    log_interval: int = 10,

    # 检查点和输出参数
    out_dir: str = './checkpoints',
    resume_from_checkpoint: Optional[str] = None,
    save_checkpoint_interval: int = 1000,

    # SwanLab 日志参数
    use_swanlab: bool = True,
    swanlab_project: str = 'CS336-assignment1',

    # 设备
    device: str = 'mps',

    # MPS 低资源训练配置
    target_val_loss: float = 2.00,  # MPS 目标验证损失（原 1.45）
):
    """
    训练 Transformer 语言模型。

    所有参数都可以通过命令行参数配置。
    """

    # 计算总 token 数
    # MPS 低资源: batch_size(32) × max_iters(5000) × context_length(256) = 40,960,000 tokens
    total_tokens = batch_size * max_iters * context_length
    print("=" * 80)
    print("训练配置 (MPS 低资源模式):")
    print("=" * 80)
    print(f"数据路径: {train_data_path}")
    print(f"验证数据路径: {val_data_path}")
    print(f"词汇表大小: {vocab_size}")
    print(f"上下文长度: {context_length}")
    print(f"模型维度 d_model: {d_model}")
    print(f"前馈层维度 d_ff: {d_ff}")
    print(f"层数: {num_layers}")
    print(f"注意力头数: {num_heads}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"最大迭代次数: {max_iters}")
    print(f"总 token 数: {total_tokens:,}")
    print(f"目标验证损失: {target_val_loss}")
    print(f"设备: {device}")
    if device == 'mps':
        print("注意: MPS 设备不使用 TF32 内核，避免不稳定的训练")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 保存训练配置
    config = {
        'train_data_path': train_data_path,
        'val_data_path': val_data_path,
        'vocab_size': vocab_size,
        'context_length': context_length,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'rope_theta': rope_theta,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'max_grad_norm': max_grad_norm,
        'warmup_iters': warmup_iters,
        'max_iters': max_iters,
        'eval_interval': eval_interval,
        'eval_iters': eval_iters,
        'log_interval': log_interval,
        'out_dir': out_dir,
        'device': device,
        'total_tokens': total_tokens,
        'target_val_loss': target_val_loss,
    }
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"配置已保存至: {config_path}")

    # 初始化 SwanLab
    swanlab_run = None
    if use_swanlab and SWANLAB_AVAILABLE:
        try:
            swanlab_run = swanlab.init(
                project=swanlab_project,
                workspace="123456",
                config=config,
            )
            print(f"SwanLab 已初始化，实验链接: {swanlab_run.url}")
        except Exception as e:
            print(f"SwanLab 初始化失败: {e}")
            print("将使用本地日志记录")
            swanlab_run = None

    # 加载数据集（使用内存映射）
    print("\n加载数据集...")
    train_dataset = load_dataset(train_data_path)
    print(f"训练集大小: {len(train_dataset):,} tokens")

    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = load_dataset(val_data_path)
        print(f"验证集大小: {len(val_dataset):,} tokens")

    # MPS 安全设置：禁止 TF32 内核，避免不稳定的训练
    if device == 'mps':
        # 确保不使用 TF32（torch.set_float32_matmul_precision('high') 在 MPS 上可能导致损坏的内核）
        torch.set_float32_matmul_precision('highest')

    # 初始化模型
    print("\n初始化模型...")
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        use_rope=True,
        weights=None,  # 从头开始训练
        device=device,
        dtype=torch.float32,
    )
    model.to(device)

    # 修正 positions 为 1D（model.py 中 expand 到 2D 会导致 RoPE 广播失败）
    def patched_forward(token_ids):
        batch_size, seq_len = token_ids.shape
        x = model.embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)  # 1D，不 expand
        for block in model.blocks:
            x = block(x, token_positions=positions)
        x = model.ln_final(x)
        logits = model.lm_head(x)
        return logits
    model.forward = patched_forward

    # 编译模型以加速训练
    print("正在编译模型 (backend=aot_eager)...")
    model = torch.compile(model, backend="aot_eager")
    print("模型编译完成")

    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    # 非嵌入参数量（排除 embedding 和 lm_head）
    non_embedding_params = num_params - sum(
        p.numel() for name, p in model.named_parameters()
        if 'embedding' in name
    )
    print(f"模型总参数量: {num_params:,}")
    print(f"非嵌入参数量 (non-embedding): {non_embedding_params:,} (目标 ≈ 17M)")

    # 初始化优化器
    optimizer = AdaW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    # 从检查点恢复（如果指定）
    start_iter = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"\n从检查点恢复: {resume_from_checkpoint}")
        start_iter = load_checkpoint(resume_from_checkpoint, model, optimizer)
        print(f"从迭代 {start_iter} 继续训练")

    # 训练循环
    print("\n开始训练...")
    model.train()
    t0 = time.time()
    running_loss = 0.0
    step_times = []  # 记录每步时间用于计算吞吐量

    for iter in range(start_iter, max_iters):
        iter_start_time = time.time()

        # 获取批次数据
        x, y = get_batch(train_dataset, batch_size, context_length, device)

        # 前向传播
        logits = model(x)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = y.view(B * T)

        # 计算损失
        loss = cross_entropy(logits_flat, targets_flat)

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        grad_norm = 0.0
        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 更新学习率（余弦退火 + 线性预热）
        lr = cosine_annealing_lr(
            it=iter,
            max_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 参数更新
        optimizer.step()

        # 记录损失和时间
        running_loss += loss.item()
        iter_time = time.time() - iter_start_time
        step_times.append(iter_time)

        # 日志输出和 SwanLab 记录
        if iter % log_interval == 0 and iter > 0:
            t1 = time.time()
            dt = t1 - t0
            avg_loss = running_loss / log_interval
            running_loss = 0.0
            t0 = t1

            # 计算吞吐量（tokens/秒）
            tokens_per_step = batch_size * context_length
            avg_step_time = np.mean(step_times[-log_interval:])
            tokens_per_sec = tokens_per_step / avg_step_time

            print(f"Iter {iter:6d} | loss: {avg_loss:.4f} | lr: {lr:.2e} | grad_norm: {grad_norm:.3f} | time: {dt*1000:.2f}ms | tokens/s: {tokens_per_sec:.0f}")

            # SwanLab 日志记录
            if swanlab_run is not None:
                swanlab.log({
                    'train/loss': avg_loss,
                    'train/learning_rate': lr,
                    'train/grad_norm': grad_norm if isinstance(grad_norm, (int, float)) else grad_norm.item(),
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/step_time_ms': avg_step_time * 1000,
                    'train/iter': iter,
                    'train/total_tokens_processed': (iter - start_iter) * batch_size * context_length,
                })

            step_times = []  # 重置

        # 评估
        if iter % eval_interval == 0 and iter > 0:
            out = estimate_loss(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_iters=eval_iters,
            )
            print(f"\n{'='*80}")
            print(f"Step {iter}: train_loss = {out['train_loss']:.4f}", end="")
            if 'val_loss' in out:
                print(f", val_loss = {out['val_loss']:.4f}", end="")
            print(f"\n{'='*80}\n")

            # SwanLab 评估日志
            if swanlab_run is not None:
                log_dict = {
                    'eval/train_loss': out['train_loss'],
                    'eval/iter': iter,
                }
                if 'val_loss' in out:
                    log_dict['eval/val_loss'] = out['val_loss']
                    # 计算困惑度
                    log_dict['eval/val_perplexity'] = math.exp(out['val_loss'])
                swanlab.log(log_dict)

            # 保存最佳模型（如果有验证集）
            if 'val_loss' in out:
                best_val_loss_path = os.path.join(out_dir, 'best_val_loss.txt')
                if not os.path.exists(best_val_loss_path):
                    best_val_loss = float('inf')
                else:
                    with open(best_val_loss_path, 'r') as f:
                        best_val_loss = float(f.read())

                if out['val_loss'] < best_val_loss:
                    best_val_loss = out['val_loss']
                    with open(best_val_loss_path, 'w') as f:
                        f.write(str(best_val_loss))
                    # 保存最佳模型
                    best_model_path = os.path.join(out_dir, 'best_model.pt')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"新的最佳验证损失: {best_val_loss:.4f}, 困惑度: {math.exp(best_val_loss):.2f}, 已保存模型至: {best_model_path}")

                # 检查是否达到目标验证损失
                if best_val_loss <= target_val_loss:
                    print(f"已达到目标验证损失 {target_val_loss}! 当前最佳: {best_val_loss:.4f}")

        # 保存检查点
        if iter % save_checkpoint_interval == 0 and iter > 0:
            checkpoint_path = os.path.join(out_dir, f'checkpoint_iter_{iter}.pt')
            save_checkpoint(model, optimizer, iter, checkpoint_path)
            print(f"检查点已保存至: {checkpoint_path}")

            # 同时保存一个最新的检查点
            latest_checkpoint_path = os.path.join(out_dir, 'checkpoint_latest.pt')
            save_checkpoint(model, optimizer, iter, latest_checkpoint_path)

    # 保存最终模型
    final_checkpoint_path = os.path.join(out_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, max_iters, final_checkpoint_path)
    print(f"\n训练完成！最终检查点已保存至: {final_checkpoint_path}")

    # 关闭 SwanLab
    if swanlab_run is not None:
        swanlab.finish()
        print("SwanLab 实验已结束")


def main():
    parser = argparse.ArgumentParser(description="Transformer 语言模型训练脚本")

    # 数据参数
    parser.add_argument('--train-data', type=str, default='bpe_output/train.bin', dest='train_data_path',
                        help='训练数据路径（.bin 格式的 token ID 数组）')
    parser.add_argument('--val-data', type=str, default='bpe_output/val.bin', dest='val_data_path',
                        help='验证数据路径（可选）')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='词汇表大小')

    # 模型参数（根据作业要求设置默认值）
    parser.add_argument('--context-length', type=int, default=256,
                        help='上下文长度（默认: 256）')
    parser.add_argument('--d-model', type=int, default=512,
                        help='模型维度（默认: 512）')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Transformer 层数（默认: 4）')
    parser.add_argument('--num-heads', type=int, default=16,
                        help='注意力头数（默认: 16）')
    parser.add_argument('--d-ff', type=int, default=1344,
                        help='前馈层维度（默认: 1344）')
    parser.add_argument('--rope-theta', type=float, default=10000.0,
                        help='RoPE theta 参数（默认: 10000.0）')

    # 优化器参数
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小（默认: 32）')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='最大学习率（默认: 3e-4）')
    parser.add_argument('--min-learning-rate', type=float, default=3e-5,
                        help='最小学习率（默认: 3e-5）')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='权重衰减（默认: 0.1）')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='AdamW beta1（默认: 0.9）')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='AdamW beta2（默认: 0.95）')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='AdamW epsilon（默认: 1e-8）')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='梯度裁剪最大范数（默认: 1.0）')

    # 学习率调度参数
    parser.add_argument('--warmup-iters', type=int, default=200,
                        help='学习率预热迭代次数（MPS 低资源默认: 200）')
    parser.add_argument('--max-iters', type=int, default=5000,
                        help='最大训练迭代次数（MPS 低资源默认: 5000，总 tokens ≈ 40,960,000）')

    # 训练控制参数
    parser.add_argument('--eval-interval', type=int, default=500,
                        help='评估间隔（MPS 低资源默认: 500）')
    parser.add_argument('--eval-iters', type=int, default=200,
                        help='评估迭代次数（默认: 200）')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='日志输出间隔（默认: 10）')

    # 检查点和输出参数
    parser.add_argument('--out-dir', type=str, default='./checkpoints',
                        help='输出目录（默认: ./checkpoints）')
    parser.add_argument('--resume', type=str, default=None, dest='resume_from_checkpoint',
                        help='从检查点恢复训练的路径')
    parser.add_argument('--save-checkpoint-interval', type=int, default=1000,
                        help='保存检查点间隔（MPS 低资源默认: 1000）')

    # SwanLab 日志参数
    parser.add_argument('--use-swanlab', action='store_true', default=True,
                        help='使用 SwanLab 记录实验日志（默认: True）')
    # 设备
    parser.add_argument('--device', type=str, default='mps',
                        help='设备（cpu, mps 或 cuda，默认: mps）')

    # MPS 低资源配置
    parser.add_argument('--target-val-loss', type=float, default=2.00,
                        help='目标验证损失（MPS 低资源默认: 2.00）')

    args = parser.parse_args()

    # 调用训练函数（argparse 参数名与 train() 签名一致，直接展开即可）
    train(**vars(args))


if __name__ == "__main__":
    main()
