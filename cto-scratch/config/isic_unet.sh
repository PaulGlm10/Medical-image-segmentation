#!/bin/sh
# U-Net de référence. À lancer depuis la racine du projet : sh config/isic_unet.sh
DATA=${DATA:-data/isic2018_256}
RUNS=${RUNS:-runs}
python train.py --name unet --model unet --data-dir "$DATA" --save-dir "$RUNS" --fold 0
