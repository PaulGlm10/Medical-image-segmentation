"""Options en ligne de commande, partagées par train.py et test.py."""
import argparse
import json
import os


def build_parser():
    p = argparse.ArgumentParser(description="CTO from scratch")
    # --- général ---
    p.add_argument("--name", default="cto", help="nom du run (dossier de résultats)")
    p.add_argument("--model", default="cto", choices=["cto", "unet"])
    p.add_argument("--data-dir", default="data/isic2018_256", help="dossier contenant img.npy, mask.npy, filename.npy")
    p.add_argument("--save-dir", default="runs", help="dossier racine des résultats")
    p.add_argument("--fold", type=int, default=0, help="0 à 4")
    p.add_argument("--num-class", type=int, default=2, help="nombre de classes, fond compris")
    p.add_argument("--seed", type=int, default=2022)
    # --- réseau ---
    p.add_argument("--backbone", default="res2net50_26w_4s", help="nom timm du backbone")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="poids ImageNet pour le backbone")
    p.add_argument("--trans-dim", type=int, default=256, help="dimension des jetons du Transformer")
    p.add_argument("--trans-depth", type=int, default=2, help="nombre de couches du Transformer sur x4")
    p.add_argument("--fusion-head", default="da", choices=["da", "light"],
                   help="da : tête du code officiel (lourde) ; light : deux convolutions")
    p.add_argument("--bim-attention", default="uncertainty", choices=["uncertainty", "background"],
                   help="uncertainty : pixels où la prédiction hésite ; background : comme le papier")
    # --- loss ---
    p.add_argument("--alpha", type=float, default=3.0, help="poids de la loss de contours")
    p.add_argument("--edge-loss", default="dice", choices=["dice", "bce_dice", "none"])
    p.add_argument("--edge-width", type=int, default=4, help="épaisseur (pixels) des contours vérité terrain")
    p.add_argument("--boundary-weight", action=argparse.BooleanOptionalAction, default=False,
                   help="pondère la loss de segmentation près des frontières (code officiel)")
    # --- entraînement ---
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.5, help="seuil d'écrêtage des gradients (0 = désactivé)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="précision mixte (GPU)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="reprend depuis last.pth s'il existe")
    return p


def run_dir(opt):
    return os.path.join(opt.save_dir, opt.name, f"fold_{opt.fold}")


def save_options(opt):
    os.makedirs(run_dir(opt), exist_ok=True)
    with open(os.path.join(run_dir(opt), "options.json"), "w") as f:
        json.dump(vars(opt), f, indent=2)


def load_options(directory):
    """Recharge les options d'un run : le test reconstruit exactement le réseau entraîné."""
    with open(os.path.join(directory, "options.json")) as f:
        return argparse.Namespace(**json.load(f))
