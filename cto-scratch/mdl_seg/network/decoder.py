"""Décodeur guidé par les contours : modules d'injection des contours (BIM)."""
import torch
import torch.nn as nn

from .blocks import ConvBNR, resize_to


def prediction_attention(logits, mode="uncertainty"):
    """Carte d'attention (B, 1, h, w) calculée à partir de la prédiction du niveau précédent.

    - "background"  : probabilité du fond, soit 1 - p(avant-plan).
    - "uncertainty" : vaut 1 là où la prédiction hésite, 0 là où elle est sûre.
                      En binaire, c'est exactement 1 - |2p - 1|.
                      Les pixels hésitants sont en général près des frontières.
    """
    prob = torch.softmax(logits.float(), dim=1)
    if mode == "background":
        return prob[:, :1]
    if mode == "uncertainty":
        c = prob.shape[1]
        p_max = prob.max(dim=1, keepdim=True).values  # entre 1/c et 1
        return 1.0 - (c * p_max - 1.0) / (c - 1.0)
    raise ValueError(f"bim_attention inconnue : {mode}")


class BIM(nn.Module):
    """Module d'injection des contours, à deux chemins.

    Entrées :
        f_c         features de l'encodeur à ce niveau (ch canaux)
        edge_prob   carte de contours prédite par le BEM (1 canal, entre 0 et 1)
        d_prev      features du niveau précédent du décodeur (ch canaux)
        logits_prev prédiction de segmentation du niveau précédent
    Chemin 1 (avant-plan) : concat[F_b, F_c, D_prev] -> convolutions, avec F_b = F_c * contours.
    Chemin 2 (attention)  : convolutions(attention * F_c).
    Sortie : fusion de concat[chemin 1, chemin 2, D_prev].
    """

    def __init__(self, ch=64, attention="uncertainty"):
        super().__init__()
        self.attention = attention
        self.fg = nn.Sequential(ConvBNR(3 * ch, ch), ConvBNR(ch, ch))
        self.bg = nn.Sequential(ConvBNR(ch, ch), ConvBNR(ch, ch), ConvBNR(ch, ch))
        self.fuse = ConvBNR(3 * ch, ch)

    def forward(self, f_c, edge_prob, d_prev, logits_prev):
        d_prev = resize_to(d_prev, f_c)
        edge = resize_to(edge_prob, f_c).to(f_c.dtype)
        att = prediction_attention(resize_to(logits_prev, f_c), self.attention).to(f_c.dtype)

        f_b = f_c * edge  # features renforcées sur les contours
        f_fg = self.fg(torch.cat([f_b, f_c, d_prev], dim=1))
        f_bg = self.bg(att * f_c)
        return self.fuse(torch.cat([f_fg, f_bg, d_prev], dim=1))
