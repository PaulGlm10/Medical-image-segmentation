import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_split_indices(n, fold, phase, n_folds=5, val_ratio=0.2):
    """
    Le fold k réserve la k-ième tranche de 1/n_folds pour le test.
    Dans le reste, les premiers val_ratio servent à la validation, le reste à l'entraînement.
    """
    idx = np.arange(n)
    if phase == "all":  # jeu externe : on évalue sur toutes les images
        return idx
    n_test = n // n_folds
    test_idx = idx[fold * n_test:(fold + 1) * n_test]
    rest = np.concatenate([idx[:fold * n_test], idx[(fold + 1) * n_test:]])
    n_val = int(len(rest) * val_ratio)
    if phase == "test":
        return test_idx
    if phase == "val":
        return rest[:n_val]
    if phase == "train":
        return rest[n_val:]
    raise ValueError("phase doit valoir train, val, test ou all")


def get_transform(phase):
    norm = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    if phase != "train":
        return A.Compose(norm)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        *norm,
    ])


class ISICDataset(Dataset):
    """Lit img.npy (N, H, W, 3), mask.npy (N, H, W) et filename.npy dans data_dir."""

    def __init__(self, data_dir, phase, fold=0, transform=None):
        self.imgs = np.load(os.path.join(data_dir, "img.npy"), mmap_mode="r")
        self.masks = np.load(os.path.join(data_dir, "mask.npy"), mmap_mode="r")
        self.names = np.load(os.path.join(data_dir, "filename.npy"))
        self.indices = get_split_indices(len(self.names), fold, phase)
        self.transform = transform if transform is not None else get_transform(phase)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        out = self.transform(image=np.array(self.imgs[j]), mask=np.array(self.masks[j]))
        mask = torch.as_tensor(out["mask"]).long()
        return {"image": out["image"], "label": mask, "name": str(self.names[j])}
