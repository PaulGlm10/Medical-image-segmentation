"""Outils : graine aléatoire, taux d'apprentissage, checkpoints, comptage des paramètres."""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_lr(optimizer, init_lr, epoch, max_epoch, power=0.9):
    """Décroissance polynomiale, comme le code officiel."""
    lr = init_lr * (1 - epoch / max_epoch) ** power
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def save_checkpoint(state, path):
    """Écrit d'abord un fichier temporaire : pas de checkpoint corrompu si Colab coupe."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parameters_by_module(model):
    """Nombre de paramètres de chaque sous-module direct."""
    return {name: count_parameters(child) for name, child in model.named_children()}


def count_flops(model, input_size=256, device="cpu"):
    """FLOPs indicatifs (en G) si le paquet thop est installé, sinon None."""
    try:
        from thop import profile
    except ImportError:
        return None

    class _FirstOutput(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[0][0]

    x = torch.randn(1, 3, input_size, input_size, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        flops, _ = profile(_FirstOutput(model), inputs=(x,), verbose=False)
    model.train(was_training)
    # thop laisse des tampons dans le modèle : on les retire
    for module in model.modules():
        for key in ("total_ops", "total_params"):
            if key in module._buffers:
                del module._buffers[key]
    return flops / 1e9
