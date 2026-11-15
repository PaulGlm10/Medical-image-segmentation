import argparse
import glob
import os

import cv2
import numpy as np

EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg")


def read_image(path, flag):
    img = cv2.imread(path, flag)
    if img is None:  # certains BMP (1 bit par pixel) sont mieux lus par Pillow
        from PIL import Image
        pil = Image.open(path).convert("L" if flag == cv2.IMREAD_GRAYSCALE else "RGB")
        img = np.array(pil)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def find_pairs(root):
    """Renvoie la liste triée des (identifiant, chemin image, chemin masque)."""
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if f.lower().endswith(EXTENSIONS)]
    by_stem = {}
    for f in files:
        by_stem.setdefault(os.path.splitext(os.path.basename(f))[0], []).append(f)
    pairs = []
    for stem, paths in by_stem.items():
        if stem.endswith("_lesion") and stem[:-len("_lesion")] in by_stem:
            name = stem[:-len("_lesion")]
            pairs.append((name, by_stem[name][0], paths[0]))
    return sorted(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dossier où PH2Dataset.rar a été décompressé")
    p.add_argument("--out", default="data/ph2_256")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    pairs = find_pairs(args.root)
    assert pairs, f"aucune paire image / masque (*_lesion) trouvée sous {args.root}"
    n, s = len(pairs), args.size
    imgs = np.zeros((n, s, s, 3), dtype=np.uint8)
    masks = np.zeros((n, s, s), dtype=np.uint8)
    for i, (name, img_path, mask_path) in enumerate(pairs):
        img = read_image(img_path, cv2.IMREAD_COLOR)
        mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        imgs[i] = cv2.cvtColor(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        mask = (mask > mask.max() / 2).astype(np.uint8)  # masque en 0/255 ou en 0/1 selon le fichier
        masks[i] = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "img.npy"), imgs)
    np.save(os.path.join(args.out, "mask.npy"), masks)
    np.save(os.path.join(args.out, "filename.npy"), np.array([name for name, _, _ in pairs]))
    print(f"=> {n} images enregistrées dans {args.out} (PH2 complet : 200)")
    print(f"   part moyenne de lésion : {100 * masks.mean():.1f} % des pixels")


if __name__ == "__main__":
    main()
