"""Contours : filtre de Sobel fixe + module d'extraction des contours (BEM)."""
import torch
import torch.nn as nn

from .blocks import ConvBNR, Conv1x1, resize_to


class SobelFilter(nn.Module):
    """Norme du gradient de Sobel, canal par canal. Les noyaux sont fixes (non appris)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))
        self.eps = eps

    def forward(self, x):
        c = x.shape[1]
        # Calcul en float32 même en précision mixte : les carrés débordent vite en float16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            gx = nn.functional.conv2d(x, self.kx.expand(c, 1, 3, 3), padding=1, groups=c)
            gy = nn.functional.conv2d(x, self.ky.expand(c, 1, 3, 3), padding=1, groups=c)
            # eps évite un gradient infini de la racine carrée en 0 (source de NaN)
            return torch.sqrt(gx * gx + gy * gy + self.eps)


class BoundaryExtraction(nn.Module):
    """BEM : prédit une carte de contours (logits, 1 canal) à la résolution de x1.

    1. Sobel sur x1 (détails fins) et x4 (sémantique).
    2. F_e = F * sigmoid(BN(Sobel(F))) : on renforce les features là où le gradient est fort.
       La BatchNorm centre la norme du gradient, pour que la sigmoïde couvre bien ]0, 1[.
    3. Fusion des deux niveaux puis prédiction, supervisée par les contours vérité terrain.
    """

    def __init__(self, ch1, ch4, mid1=64, mid4=256):
        super().__init__()
        self.sobel = SobelFilter()
        self.norm1 = nn.BatchNorm2d(ch1)
        self.norm4 = nn.BatchNorm2d(ch4)
        self.reduce1 = Conv1x1(ch1, mid1)
        self.reduce4 = Conv1x1(ch4, mid4)
        self.block = nn.Sequential(
            ConvBNR(mid1 + mid4, 256),
            ConvBNR(256, 256),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x1, x4):
        fe1 = x1 * torch.sigmoid(self.norm1(self.sobel(x1)))
        fe4 = x4 * torch.sigmoid(self.norm4(self.sobel(x4)))
        r1 = self.reduce1(fe1)
        r4 = resize_to(self.reduce4(fe4), r1)
        return self.block(torch.cat([r1, r4], dim=1))  # logits (B, 1, H/4, W/4)
