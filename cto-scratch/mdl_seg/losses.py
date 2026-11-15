"""Losses : segmentation (CE + IoU, supervision profonde) et contours (Dice, ou BCE + Dice)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_from_mask(mask, num_class=2, width=4):
    """Contours vérité terrain.

    mask : (B, H, W) entiers. Renvoie (B, 1, H, W) avec 1 sur un anneau de `width` pixels
    à cheval sur la frontière (dilatation - érosion, faites avec un max-pooling).
    """
    onehot = F.one_hot(mask.long(), num_class).permute(0, 3, 1, 2).float()[:, 1:]  # sans le fond
    r_in = width // 2
    r_out = width - r_in
    dilated = F.max_pool2d(onehot, 2 * r_out + 1, stride=1, padding=r_out) if r_out > 0 else onehot
    eroded = -F.max_pool2d(-onehot, 2 * r_in + 1, stride=1, padding=r_in) if r_in > 0 else onehot
    return (dilated - eroded).amax(dim=1, keepdim=True)


def seg_loss(logits, target, boundary_weight=False, smooth=1.0):
    """Cross-entropie + IoU douce.

    boundary_weight=True :
    les pixels proches d'une frontière comptent jusqu'à 6 fois plus.
    """
    logits = logits.float()
    num_class = logits.shape[1]
    target = target.long()
    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_class).permute(0, 3, 1, 2).float()

    if boundary_weight:
        fg = (target > 0).float().unsqueeze(1)
        weight = 1 + 5 * torch.abs(F.avg_pool2d(fg, 31, stride=1, padding=15) - fg)
    else:
        weight = torch.ones_like(prob[:, :1])

    ce = F.cross_entropy(logits, target, reduction="none").unsqueeze(1)
    ce = (ce * weight).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3))

    inter = (prob * onehot * weight).sum(dim=(2, 3))
    union = ((prob + onehot - prob * onehot) * weight).sum(dim=(2, 3))
    iou = 1 - (inter[:, 1:] + smooth) / (union[:, 1:] + smooth)  # classes d'avant-plan

    return ce.mean() + iou.mean()


def dice_loss(logits, target, smooth=1.0):
    """Dice binaire (éq. 9), appliquée sur la sigmoïde des logits."""
    prob = torch.sigmoid(logits.float())
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


class CTOLoss(nn.Module):
    """L = somme sur les niveaux de (CE + IoU)  +  alpha * loss de contours.

    edge_loss : "dice", "bce_dice" (variante) ou "none" (pas de supervision des contours).
    """

    def __init__(self, num_class=2, alpha=3.0, edge_loss="dice", edge_width=4, boundary_weight=False):
        super().__init__()
        assert edge_loss in ("dice", "bce_dice", "none")
        self.num_class = num_class
        self.alpha = alpha
        self.edge_loss = edge_loss
        self.edge_width = edge_width
        self.boundary_weight = boundary_weight

    def forward(self, seg_outputs, edge_logits, target):
        loss_seg = sum(seg_loss(o, target, self.boundary_weight) for o in seg_outputs)
        parts = {"seg": loss_seg.item()}
        loss = loss_seg

        if edge_logits is not None and self.edge_loss != "none":
            edge_gt = boundary_from_mask(target, self.num_class, self.edge_width)
            loss_edge = dice_loss(edge_logits, edge_gt)
            if self.edge_loss == "bce_dice":
                loss_edge = loss_edge + F.binary_cross_entropy_with_logits(edge_logits.float(), edge_gt)
            parts["edge"] = loss_edge.item()
            loss = loss + self.alpha * loss_edge

        return loss, parts
