import torch
import torch.nn as nn


class CoarseFineInteraction(nn.Module):
    """Cross-attention from fine to coarse tokens.

    Args:
        dim (int): embedding dimension.
        num_heads (int): number of attention heads.
        qkv_bias (bool): include bias for q, k, v projections.
        attn_drop (float): dropout rate on attention weights.
        proj_drop (float): dropout rate on output projection.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        """Compute cross-attention.

        Args:
            query (Tensor): fine tokens of shape (B, N_q, C).
            context (Tensor): coarse tokens of shape (B, N_k, C).
        Returns:
            Tensor: cross-attended features of shape (B, N_q, C).
        """
        B, N_q, C = query.shape
        N_k = context.shape[1]

        q = self.q(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N_k, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
