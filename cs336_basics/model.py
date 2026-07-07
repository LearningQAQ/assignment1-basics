import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum
from .layers import *
from .attention import *


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, 
                rope_theta: float=10000.0, use_rope: bool=True, device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.norm1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model,
                                                 num_heads=num_heads,
                                                 max_seq_len=max_seq_len,
                                                 rope_theta=rope_theta,
                                                 use_rope=use_rope,
                                                 **factory_kwargs)
        
        self.norm2 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, **factory_kwargs)
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"],
                ) -> Float[Tensor, " batch sequence_length d_model"]:
        
        # --- Sub-layer 1: Causal Multi-Head Self-Attention + Residual ---
        norm1_out = self.norm1(x)
        attn_out = self.attn(norm1_out, token_positions)
        x = x + attn_out

        # --- Sub-layer 2: SwiGLU FFN + Residual ---
        norm2_out = self.norm2(x)
        ffn_out = self.ffn(norm2_out)
        x = x + ffn_out

        return x



class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, weights: dict[str, Tensor], rope_theta: float=10000.0,
                 use_rope: bool=True, device=None,dtype=None):
        super().__init__()
        self.context_length = context_length
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=use_rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)
    
        
    def forward(self, token_ids: Int[Tensor, " batch_size sequence_length"]
                ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        batch_size, seq_len = token_ids.shape
        
        x = self.embedding(token_ids)
        
        positions = torch.arange(seq_len).expand(batch_size, seq_len)
        for block in self.blocks:
            x = block(x, token_positions=positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
