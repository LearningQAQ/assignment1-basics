import torch
from torch import nn
from torch.nn import init
import math
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    """
    Args:
        num_embeddings: 词表大小
        embedding_dim: 嵌入向量维度
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # (batch_size, sequence_length, d_model)
        RMS = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / RMS * self.weight
        return x.to(in_dtype)
    

def silu(x: torch.Tensor): return x * torch.sigmoid(x)
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs) # 升维
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs) # 降维
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs) # 升维
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch seq d_model)
        w1x = self.linear1(x)
        w3x = self.linear3(x)
        return self.linear2(silu(w1x) * w3x)



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE 要求 d_k 必须为偶数。")
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 计算频率基： 1 / theta ** (2i / d_k)          (i = 0,1,…,d_k/2-1)
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        positions = torch.arange(max_seq_len, device=device)
        freqs = torch.outer(positions, freq)  # shape: (max_seq_len, d_k // 2)

        # 缓存 cos 和 sin 编码
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape: (..., sequence_length, d_k)
        cos_pos = self.cos_cached[token_positions]  # shape: (sequence_length, d_k // 2)
        sin_pos = self.sin_cached[token_positions]  # shape: (sequence_length, d_k // 2)

        # 分割输入为偶数和奇数维度: x = [x_even, x_odd]. 
        # shape: (..., sequence_length, d_k // 2)
        x_even = x[..., ::2]   # 偶数索引: 0, 2, 4, ...   
        x_odd  = x[..., 1::2]  # 奇数索引: 1, 3, 5, ...

        """
        应用旋转公式, 对每个相邻维度对 (x_even, x_odd) 施加二维旋转:
            ┌      ┐   ┌            ┐ ┌        ┐
            │  x'  │ = │ cosθ -sinθ │ │ x_even │
            │  y'  │   │ sinθ  cosθ │ │ x_odd  │
            └      ┘   └            ┘ └        ┘
        """
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # 交错合并回原始维度顺序
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

