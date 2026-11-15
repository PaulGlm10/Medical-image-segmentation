#!/bin/sh
# Ablations : chaque run change une seule chose par rapport à config/isic_cto.sh.
DATA=${DATA:-data/isic2018_256}
RUNS=${RUNS:-runs}
COMMON="--data-dir $DATA --save-dir $RUNS --fold 0"

# Sans supervision des contours (comportement réel du code officiel)
python train.py --name cto_no_edge $COMMON --edge-loss none
# BIM avec l'attention sur l'arrière-plan, comme le papier
python train.py --name cto_background $COMMON --bim-attention background
# Sans couche Transformer sur x4
python train.py --name cto_no_trans $COMMON --trans-depth 0
# Tête de fusion légère (environ 30 M de paramètres en moins)
python train.py --name cto_light $COMMON --fusion-head light
