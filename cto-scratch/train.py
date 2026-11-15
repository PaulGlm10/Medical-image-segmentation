"""Entraîne un modèle.  Exemple : python train.py --name cto --data-dir data/isic2018_256"""
from mdl_seg.options import build_parser
from mdl_seg.trainer import CTOTrainer


def main():
    opt = build_parser().parse_args()
    CTOTrainer(opt).run()


if __name__ == "__main__":
    main()
