"""Assemblage du réseau CTO.

    image -> Res2Net-50 -> x1, x2, x3, x4
    x4 -> Transformer global -> concat avec x4 -> tête de fusion -> d4 (début du décodeur)
    x1, x4 -> Sobel + BEM -> carte de contours
    d4 -> BIM(x3) -> BIM(x2) -> BIM(x1) -> segmentation

Le réseau renvoie (liste des sorties de segmentation, logits des contours).
Les sorties de segmentation vont de la plus fine à la plus grossière : [o1, o2, o3, o4],
toutes remises à la taille de l'image (supervision profonde). La prédiction finale est o1.
"""
import torch
import torch.nn as nn

from .backbone import Res2NetBackbone
from .blocks import Conv1x1, resize_to
from .boundary import BoundaryExtraction
from .decoder import BIM
from .fusion import build_fusion
from .transformer import GlobalTransformer


class CTO(nn.Module):
    def __init__(
        self,
        num_class=2,
        backbone="res2net50_26w_4s",
        pretrained=True,
        trans_dim=256,
        trans_depth=2,
        trans_heads=8,
        fusion_head="da",
        bim_attention="uncertainty",
        dec_ch=64,
    ):
        super().__init__()
        assert num_class >= 2, "num_class compte le fond : 2 pour une segmentation binaire"
        # --- Encodeur : flux CNN + flux Transformer ---
        self.backbone = Res2NetBackbone(backbone, pretrained)
        c1, c2, c3, c4 = self.backbone.channels
        self.transformer = GlobalTransformer(c4, trans_dim, trans_depth, trans_heads)
        self.fusion = build_fusion(fusion_head, c4 + trans_dim, dec_ch)

        # --- Contours ---
        self.bem = BoundaryExtraction(c1, c4)

        # --- Décodeur ---
        self.reduce1 = Conv1x1(c1, dec_ch)
        self.reduce2 = Conv1x1(c2, dec_ch)
        self.reduce3 = Conv1x1(c3, dec_ch)
        self.bim3 = BIM(dec_ch, bim_attention)
        self.bim2 = BIM(dec_ch, bim_attention)
        self.bim1 = BIM(dec_ch, bim_attention)
        self.predictor4 = nn.Conv2d(dec_ch, num_class, 1)
        self.predictor3 = nn.Conv2d(dec_ch, num_class, 1)
        self.predictor2 = nn.Conv2d(dec_ch, num_class, 1)
        self.predictor1 = nn.Conv2d(dec_ch, num_class, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x1, x2, x3, x4 = self.backbone(x)

        # Carte de contours, à la résolution de x1
        edge_logits = self.bem(x1, x4)
        edge_prob = torch.sigmoid(edge_logits)

        # Fond de l'encodeur : fusion CNN + Transformer
        trans = self.transformer(x4)
        d4 = self.fusion(torch.cat([x4, trans], dim=1))
        o4 = self.predictor4(d4)

        # Décodeur : chaque BIM reçoit la prédiction du niveau précédent
        d3 = self.bim3(self.reduce3(x3), edge_prob, d4, o4)
        o3 = self.predictor3(d3)
        d2 = self.bim2(self.reduce2(x2), edge_prob, d3, o3)
        o2 = self.predictor2(d2)
        d1 = self.bim1(self.reduce1(x1), edge_prob, d2, o2)
        o1 = self.predictor1(d1)

        seg_outputs = [resize_to(o, size) for o in (o1, o2, o3, o4)]
        return seg_outputs, resize_to(edge_logits, size)
