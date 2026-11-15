"""Métriques : Dice et IoU par image, moyennés sur les classes d'avant-plan."""
import torch


@torch.no_grad()
def dice_iou(pred, target, num_class=2):
    """pred, target : (B, H, W) entiers. Renvoie deux tenseurs (B,) : Dice et IoU par image."""
    dices, ious = [], []
    for c in range(1, num_class):
        p, t = pred == c, target == c
        inter = (p & t).sum(dim=(1, 2)).float()
        total = p.sum(dim=(1, 2)).float() + t.sum(dim=(1, 2)).float()
        union = total - inter
        # Si la classe est absente de la prédiction et de la vérité terrain : score parfait.
        dices.append(torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(inter)))
        ious.append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter)))
    return torch.stack(dices).mean(dim=0), torch.stack(ious).mean(dim=0)
