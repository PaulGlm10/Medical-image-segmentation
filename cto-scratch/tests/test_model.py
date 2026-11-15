"""Tests rapides, sur CPU et sans données :  python -m pytest tests -q"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdl_seg.dataset import get_split_indices
from mdl_seg.losses import CTOLoss, boundary_from_mask, dice_loss
from mdl_seg.metrics import dice_iou
from mdl_seg.network.cto_net import CTO
from mdl_seg.network.decoder import prediction_attention
from mdl_seg.network.unet import UNet


def make_batch(b=2, size=128):
    image = torch.randn(b, 3, size, size)
    mask = torch.zeros(b, size, size, dtype=torch.long)
    mask[:, size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1
    return image, mask


def test_cto_output_shapes():
    for head in ("da", "light"):
        net = CTO(pretrained=False, fusion_head=head)
        image, _ = make_batch()
        seg_outputs, edge_logits = net(image)
        assert len(seg_outputs) == 4
        for out in seg_outputs:
            assert out.shape == (2, 2, 128, 128)
        assert edge_logits.shape == (2, 1, 128, 128)


def test_unet_output_shapes():
    seg_outputs, edge_logits = UNet()(make_batch()[0])
    assert seg_outputs[0].shape == (2, 2, 128, 128) and edge_logits is None


def test_edge_loss_trains_the_boundary_branch():
    """Le défaut du code officiel : la loss de contours avait un gradient nul."""
    net = CTO(pretrained=False, fusion_head="light")
    image, mask = make_batch()
    _, edge_logits = net(image)
    edge_gt = boundary_from_mask(mask, 2, 4)
    losses = {
        "dice": dice_loss(edge_logits, edge_gt),
        "bce_dice": dice_loss(edge_logits, edge_gt)
        + torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_gt),
    }
    for loss in losses.values():
        net.zero_grad()
        loss.backward(retain_graph=True)
        grad = net.bem.block[-1].weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_total_loss_backward():
    net = CTO(pretrained=False, fusion_head="light")
    image, mask = make_batch()
    loss, parts = CTOLoss()(*net(image), mask)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"seg", "edge"}
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_prediction_attention():
    # trois pixels : sûr du fond, hésitant, sûr de l'avant-plan
    logits = torch.tensor([[[[10.0, 0.0, -10.0]], [[-10.0, 0.0, 10.0]]]])
    unc = prediction_attention(logits, "uncertainty").flatten()
    bg = prediction_attention(logits, "background").flatten()
    assert torch.allclose(unc, torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)
    assert torch.allclose(bg, torch.tensor([1.0, 0.5, 0.0]), atol=1e-4)
    # en binaire : 1 - |2p - 1|
    p = torch.softmax(torch.randn(1, 2, 8, 8), dim=1)
    expected = 1 - (2 * p[:, 1:] - 1).abs()
    assert torch.allclose(prediction_attention(p.log(), "uncertainty"), expected, atol=1e-5)


def test_boundary_from_mask_width():
    _, mask = make_batch(b=1, size=64)
    for width in (2, 4, 6):
        edge = boundary_from_mask(mask, 2, width)[0, 0]
        assert edge.shape == (64, 64) and set(edge.unique().tolist()) == {0.0, 1.0}
        # le long d'une ligne qui traverse le carré : deux frontières de `width` pixels
        assert edge[32].sum().item() == 2 * width


def test_splits_are_disjoint_and_complete():
    n = 2594
    for fold in range(5):
        parts = [set(get_split_indices(n, fold, ph).tolist()) for ph in ("train", "val", "test")]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert len(parts[0] | parts[1] | parts[2]) == n  # aucune image perdue
    tests = [set(get_split_indices(n, f, "test").tolist()) for f in range(5)]
    assert sum(len(t) for t in tests) == len(set().union(*tests))  # les 5 jeux de test ne se recouvrent pas


def test_dice_iou():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1  # 8 pixels
    pred = torch.zeros_like(target)
    pred[0, :1] = 1    # 4 pixels, tous corrects
    dice, iou = dice_iou(pred, target)
    assert np.isclose(dice.item(), 2 * 4 / 12) and np.isclose(iou.item(), 4 / 8)
