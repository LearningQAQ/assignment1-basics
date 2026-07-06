import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from .layers import *


def softmax_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = d_k ** 0.5
    
    def forward(
        self,   
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... queries d_v"]:
        
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        attn_weights = softmax_stable(scores, dim = -1) # 对每一个 query, 在所有 keys 上做归一化
        
        return einsum(attn_weights, V, "... queries keys, ... keys d_v ->  ... queries d_v")
    

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float=10000.0,
                 use_rope: bool=True, device=None,dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被num_heads ({num_heads}) 整除")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        self.attn = ScaledDotProductAttention(self.d_k)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
            
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None,
                ) -> Float[Tensor, " ... sequence_length d_model"]:
        batch_size, seq_len, _ = x.shape
        
        # 1. QKV 投影 + reshape
        q, k, v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads)
                    for proj in [self.q_proj, self.k_proj, self.v_proj]]
        
        # 2. RoPE
        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        
        # 3. 动态切割掩码 & Attention 计算
        out = self.attn(q, k, v, self.causal_mask[..., :seq_len, :seq_len])

        # 4. 合并多头 & 输出投影
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


