"""Évalue un run sur le jeu de test.

Exemples :
    python test.py --run-dir runs/cto/fold_0
    python test.py --run-dir runs/cto/fold_0 --external-dir data/ph2_256 --external-name ph2
"""
import argparse

from mdl_seg.inference import CTOInference
from mdl_seg.options import load_options


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="dossier du run, ex. runs/cto/fold_0")
    p.add_argument("--data-dir", default=None, help="remplace le dossier de données enregistré")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--tta", action="store_true", help="moyenne sur les retournements de l'image")
    p.add_argument("--save-pred", action="store_true", help="enregistre les masques prédits")
    p.add_argument("--external-dir", default=None,
                   help="évalue le modèle sur un autre jeu de données (toutes ses images), ex. data/ph2_256")
    p.add_argument("--external-name", default=None, help="nom du jeu externe (défaut : nom du dossier)")
    args = p.parse_args()

    opt = load_options(args.run_dir)  # mêmes options qu'à l'entraînement : même réseau
    if args.data_dir:
        opt.data_dir = args.data_dir
    CTOInference(opt, args.checkpoint, args.tta, args.save_pred, args.external_dir, args.external_name).run()


if __name__ == "__main__":
    main()
