import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 数据集长度必须足以容纳至少一个完整样本 (context_length + 1)
    min_required_len = context_length + 1
    if len(dataset) < min_required_len:
        raise ValueError(
            f"dataset length ({len(dataset)}) is too short for "
            f"context_length={context_length}; need at least {min_required_len}"
        )
    
    # 随机采样 batch_size 个起始索引
    max_start = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start, size=batch_size)

    # 构建 (batch_size, context_length+1) 的索引矩阵并提取数据
    offsets = np.arange(context_length + 1)
    indices = start_indices[:, None] + offsets[None, :]  # (batch_size, context_length+1)
    chunk = dataset[indices]  # numpy array, shape: (batch_size, context_length+1)

    # 转换为 torch tensor 并拆分输入和标签
    tensor = torch.from_numpy(chunk).long()
    x = tensor[:, :context_length]   # 输入: 前 context_length 个 token
    y = tensor[:, 1:context_length + 1]  # 标签: 后 context_length 个 token（右移一位）

    return x.to(device), y.to(device)
    