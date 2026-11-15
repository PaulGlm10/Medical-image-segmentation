"""Petites briques de base réutilisées partout dans le réseau."""
import torch.nn as nn
import torch.nn.functional as F


class ConvBNR(nn.Sequential):
    """Convolution 3x3 -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Conv1x1(ConvBNR):
    """Convolution 1x1 -> BatchNorm -> ReLU. Sert à changer le nombre de canaux."""

    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1)


def resize_to(x, ref):
    """Redimensionne x (bilinéaire) à la taille spatiale de ref (tenseur ou tuple (h, w))."""
    size = ref.shape[-2:] if hasattr(ref, "shape") else ref
    if x.shape[-2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
