import argparse
import glob
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="dossier ISIC2018_Task1-2_Training_Input")
    p.add_argument("--masks", required=True, help="dossier ISIC2018_Task1_Training_GroundTruth")
    p.add_argument("--out", default="data/isic2018_256")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--limit", type=int, default=0, help="ne garder que N images (essais rapides)")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "ISIC_*.jpg")))
    assert paths, f"aucune image ISIC_*.jpg dans {args.images}"
    order = np.random.RandomState(args.seed).permutation(len(paths))
    paths = [paths[i] for i in order]
    if args.limit:
        paths = paths[:args.limit]

    n, s = len(paths), args.size
    imgs = np.zeros((n, s, s, 3), dtype=np.uint8)
    masks = np.zeros((n, s, s), dtype=np.uint8)
    names = []
    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(args.masks, f"{name}_segmentation.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"image illisible : {path}"
        assert mask is not None, f"masque introuvable : {mask_path}"
        imgs[i] = cv2.cvtColor(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        masks[i] = (cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        names.append(name)
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"{i + 1}/{n}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "img.npy"), imgs)
    np.save(os.path.join(args.out, "mask.npy"), masks)
    np.save(os.path.join(args.out, "filename.npy"), np.array(names))
    print(f"=> {n} images enregistrées dans {args.out}")


if __name__ == "__main__":
    main()
