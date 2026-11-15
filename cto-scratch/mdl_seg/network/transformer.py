"""Module longue portée : un petit Transformer classique appliqué sur x4.

x4 est la carte la plus profonde (8x8 pour une image 256x256), soit seulement 64 jetons :
une attention complète entre toutes les positions y coûte très peu, et chaque position
peut regarder toutes les autres (vraie longue portée).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv1x1


class MultiHeadSelfAttention(nn.Module):
    """Attention multi-têtes classique : softmax(Q K^T / sqrt(d)) V."""

    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim doit être divisible par heads"
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)  # une seule couche calcule Q, K et V
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):  # x : (B, N, dim), N = nombre de jetons (positions)
        b, n, dim = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # chacun : (B, têtes, N, dim/têtes)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, dim)
        return self.proj(out)


class TransformerLayer(nn.Module):
    """Couche pré-norm : x + Attention(LN(x)), puis x + MLP(LN(x))."""

    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalTransformer(nn.Module):
    """x4 (B, C, h, w) -> features globales (B, dim, h, w)."""

    def __init__(self, in_ch, dim=256, depth=2, heads=8, mlp_ratio=4.0, dropout=0.1, grid=8):
        super().__init__()
        self.reduce = Conv1x1(in_ch, dim)  # 2048 -> 256 canaux avant l'attention
        # Encodage de position appris, défini pour une grille grid x grid
        # (interpolé si l'image n'est pas en 256x256).
        self.pos = nn.Parameter(torch.zeros(1, dim, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x4):
        x = self.reduce(x4)
        b, c, h, w = x.shape
        pos = self.pos
        if pos.shape[-2:] != (h, w):
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        x = (x + pos).flatten(2).transpose(1, 2)  # (B, h*w, dim) : un jeton par position
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(b, c, h, w)
