"""Fusion du flux CNN (x4) et du flux Transformer, au fond de l'encodeur.

Deux têtes au choix :
- "da"    : tête à double attention (position + canal).
- "light" : deux convolutions, beaucoup plus légère.
"""
import torch
import torch.nn as nn

from .blocks import ConvBNR, Conv1x1


class PositionAttention(nn.Module):
    """Attention entre positions (DANet)."""

    def __init__(self, ch):
        super().__init__()
        self.conv_q = nn.Conv2d(ch, ch // 8, 1)
        self.conv_k = nn.Conv2d(ch, ch // 8, 1)
        self.conv_v = nn.Conv2d(ch, ch, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.conv_q(x).flatten(2).transpose(1, 2).float()  # (B, N, c/8)
        k = self.conv_k(x).flatten(2).float()                  # (B, c/8, N)
        v = self.conv_v(x).flatten(2).float()                  # (B, c, N)
        # Produits matriciels en float32 : ils peuvent déborder en précision mixte.
        with torch.autocast(device_type=x.device.type, enabled=False):
            attn = torch.softmax(torch.bmm(q, k), dim=-1)      # (B, N, N)
            out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.alpha * out.to(x.dtype) + x


class ChannelAttention(nn.Module):
    """Attention entre canaux (DANet)."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        feat = x.flatten(2).float()                           # (B, c, N)
        with torch.autocast(device_type=x.device.type, enabled=False):
            attn = torch.bmm(feat, feat.transpose(1, 2))      # (B, c, c)
            attn = attn.max(dim=-1, keepdim=True).values - attn
            attn = torch.softmax(attn, dim=-1)
            out = torch.bmm(attn, feat).view(b, c, h, w)
        return self.beta * out.to(x.dtype) + x


class DAHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inter = in_ch // 4
        self.conv_p1 = ConvBNR(in_ch, inter)
        self.conv_c1 = ConvBNR(in_ch, inter)
        self.pam = PositionAttention(inter)
        self.cam = ChannelAttention()
        self.conv_p2 = ConvBNR(inter, inter)
        self.conv_c2 = ConvBNR(inter, inter)
        self.out = nn.Sequential(nn.Dropout(0.1), Conv1x1(inter, out_ch))

    def forward(self, x):
        feat_p = self.conv_p2(self.pam(self.conv_p1(x)))
        feat_c = self.conv_c2(self.cam(self.conv_c1(x)))
        return self.out(feat_p + feat_c)


class LightFusion(nn.Sequential):
    def __init__(self, in_ch, out_ch, mid_ch=256):
        super().__init__(Conv1x1(in_ch, mid_ch), ConvBNR(mid_ch, out_ch))


def build_fusion(kind, in_ch, out_ch):
    if kind == "da":
        return DAHead(in_ch, out_ch)
    if kind == "light":
        return LightFusion(in_ch, out_ch)
    raise ValueError(f"fusion_head inconnue : {kind}")
