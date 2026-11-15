"""Flux CNN de l'encodeur : Res2Net-50 pré-entraîné sur ImageNet, chargé via timm."""
import timm
import torch.nn as nn


class Res2NetBackbone(nn.Module):
    """Renvoie les 4 cartes de features x1..x4.

    Pour une image 256x256 et res2net50 :
        x1 : 256 canaux, 64x64   (1/4)
        x2 : 512 canaux, 32x32   (1/8)
        x3 : 1024 canaux, 16x16  (1/16)
        x4 : 2048 canaux, 8x8    (1/32)
    """

    def __init__(self, name="res2net50_26w_4s", pretrained=True):
        super().__init__()
        self.body = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        self.channels = self.body.feature_info.channels()  # ex. [256, 512, 1024, 2048]

    def forward(self, x):
        return self.body(x)
